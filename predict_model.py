# predict.py

import joblib
import pandas as pd
import json
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# --- 1. LOAD MODEL, COLUMNS, AND THE TEST SET "MENU" ---
print("Loading model, column blueprint, and test set...")
try:
    model = joblib.load("traffic_model.joblib")
    
    with open('model_columns.json', 'r') as f:
        model_columns = json.load(f)
    
    test_set_df = pd.read_csv("user_test_set.csv")
    
except FileNotFoundError:
    print("\n--- ERROR ---")
    print("One or more files ('traffic_model.joblib', 'model_columns.json', 'user_test_set.csv') not found.")
    print("Please run train_model.py first to create these files.")
    exit()

print("Model and test set loaded successfully.")


# --- 2. PREDICTION FUNCTION ---
def process_and_predict(selected_row_series):
    """
    Takes a single row (as a pandas Series) from the *original* data,
    processes it, and returns a prediction.
    """
    
    # Start with our "empty" row, all columns set to 0
    data_row = {col: 0 for col in model_columns}

    # A. Process date_time
    try:
        dt = pd.to_datetime(selected_row_series['date_time'], dayfirst=True)
        data_row['hour'] = dt.hour
        data_row['day_of_week'] = dt.dayofweek
        data_row['month'] = dt.month
        data_row['year'] = dt.year
    except Exception as e:
        print(f"Error processing date_time: {e}")
        return None

    # B. Fill in simple numerical values
    data_row['temp'] = selected_row_series['temp']
    data_row['rain_1h'] = selected_row_series['rain_1h']
    data_row['snow_1h'] = selected_row_series['snow_1h']
    data_row['clouds_all'] = selected_row_series['clouds_all']

    # C. Fill in one-hot encoded "categorical" values
    holiday_col = f"holiday_{selected_row_series['holiday']}"
    weather_main_col = f"weather_main_{selected_row_series['weather_main']}"
    weather_desc_col = f"weather_description_{selected_row_series['weather_description']}"
    
    if holiday_col in data_row:
        data_row[holiday_col] = 1
    if weather_main_col in data_row:
        data_row[weather_main_col] = 1
    if weather_desc_col in data_row:
        data_row[weather_desc_col] = 1
        
    # D. Make the prediction
    try:
        data_df = pd.DataFrame([data_row], columns=model_columns)
        prediction = model.predict(data_df)
        return prediction[0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None


# --- 3. INTERACTIVE PREDICTION LOOP ---
if __name__ == "__main__":
    
    print("\n--- Interactive Traffic Prediction ---")
    print("This will use data from the 20% 'test set' (data the model has *not* seen).")
    
    print("\nHere are the first 5 rows from the test set ('user_test_set.csv'):")
    test_set_df = test_set_df.reset_index(drop=True)
    
    try:
        # Try to print the nice markdown table
        print(test_set_df.head().to_markdown(numalign="left", stralign="left"))
    except ImportError:
        # If tabulate is not installed, print a basic table
        print("\nNOTE: 'tabulate' package not found. Printing a basic table.")
        print("You can install it with 'pip install tabulate' for a nicer view.\n")
        print(test_set_df.head())
    
    total_rows = len(test_set_df)
    print(f"\n(Total rows in test set: {total_rows})")

    while True:
        try:
            row_num_str = input(f"\nEnter a row number (0 to {total_rows - 1}) to predict (or 'q' to quit): ")
            
            if row_num_str.lower() == 'q':
                break
                
            row_num = int(row_num_str)
            
            if not (0 <= row_num < total_rows):
                print(f"Error: Please enter a number between 0 and {total_rows - 1}.")
                continue
                
            selected_row = test_set_df.iloc[row_num]
            actual_traffic = selected_row['traffic_volume']
            predicted_traffic = process_and_predict(selected_row)
            
            if predicted_traffic is not None:
                print("\n--- Prediction Comparison ---")
                print(f"Data for row:     {row_num}")
                print(f"Time:             {selected_row['date_time']}")
                print(f"Weather:          {selected_row['weather_description']}")
                
                difference = predicted_traffic - actual_traffic

                if actual_traffic == 0:
                    percent_error_str = "N/A (Actual was 0)"
                else:
                    percent_error = (difference / actual_traffic) * 100
                    percent_error_str = f"{percent_error:.2f}%"
                
                print("\nRESULTS:")
                print(f" > Actual Traffic:    {actual_traffic}")
                print(f" > Predicted Traffic: {predicted_traffic:.0f}")
                print(f" > Difference:        {difference:.0f} vehicles")
                print(f" > Percent Error:     {percent_error_str}")

        except ValueError:
            print("Error: Invalid input. Please enter a number.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    print("\nExiting program.")