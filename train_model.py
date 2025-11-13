# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import json
import time

# --- Configuration ---
DATA_FILE = "Metro_Interstate_Traffic_Volume.csv"
TEST_SET_FILE = "user_test_set.csv"
MODEL_FILE = "traffic_model.joblib"
COLUMNS_FILE = "model_columns.json"

def load_data(file_path):
    """Loads and validates the initial CSV data."""
    print(f"[Step 1/8] Loading data from '{file_path}'...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"\n--- ERROR ---")
        print(f"File not found: '{file_path}'. Please check the file name.")
        return None
    print("           ...Data loaded successfully.")
    return df

def engineer_features(df):
    """Engineers new time-based features from the 'date_time' column."""
    print("[Step 2/8] Engineering features from 'date_time'...")
    try:
        df['date_time'] = pd.to_datetime(df['date_time'], dayfirst=True)
    except ValueError as e:
        print(f"\n--- ERROR ---")
        print(f"Date conversion error: {e}")
        return None
        
    df['hour'] = df['date_time'].dt.hour
    df['day_of_week'] = df['date_time'].dt.dayofweek
    df['month'] = df['date_time'].dt.month
    df['year'] = df['date_time'].dt.year
    df = df.drop('date_time', axis=1)
    print("           ...Time features (hour, day_of_week, etc.) created.")
    return df

def preprocess_data(df):
    """Applies one-hot encoding to categorical features."""
    print("[Step 3/8] Applying one-hot encoding to categorical data...")
    categorical_cols = ['holiday', 'weather_main', 'weather_description']
    df = pd.get_dummies(df, columns=categorical_cols)
    print("           ...Categorical columns converted to numeric.")
    return df

def save_prediction_artifacts(X, test_indices):
    """Saves the column blueprint and the user-facing test set."""
    print(f"[Step 4/8] Saving column blueprint to '{COLUMNS_FILE}'...")
    column_list = list(X.columns)
    with open(COLUMNS_FILE, 'w') as f:
        json.dump(column_list, f)

    print(f"[Step 5/8] Saving 'unseen' test set to '{TEST_SET_FILE}'...")
    df_original = pd.read_csv(DATA_FILE)
    df_test_set_for_user = df_original.iloc[test_indices]
    df_test_set_for_user.to_csv(TEST_SET_FILE, index=False)

def main():
    """Main function to orchestrate the model training pipeline."""
    print("=" * 60)
    print("--- PROFESSIONAL MODEL TRAINING PIPELINE ---")
    print("=" * 60)
    
    # 1. Load Data
    df = load_data(DATA_FILE)
    if df is None: return

    # 2. Feature Engineering
    df = engineer_features(df)
    if df is None: return

    # 3. Preprocessing
    df_processed = preprocess_data(df)

    # 4. Define Features (X) and Target (y)
    print("[Step 6/8] Separating features (X) and target (y)...")
    y = df_processed['traffic_volume']
    X = df_processed.drop('traffic_volume', axis=1)

    # 5. Train/Test Split
    print("[Step 7/8] Splitting data into training (80%) and testing (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. Save Artifacts for Prediction Script
    save_prediction_artifacts(X, X_test.index)

    # 7. Model Training
    print("\n[Step 8/8] TRAINING MODEL: GradientBoostingRegressor")
    print("           This is a complex model and will take 5-10 minutes.")
    print("           Please be patient...")
    start_time = time.time()
    
    model = GradientBoostingRegressor(
        n_estimators=1500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.7,
        random_state=42,
        verbose=1  # This will print progress
    )
    
    model.fit(X_train, y_train)
    
    end_time = time.time()
    print(f"\nTraining complete. Total time: {(end_time - start_time) / 60:.2f} minutes")

    # 8. Model Evaluation
    print("\n--- PERFORMANCE EVALUATION ---")
    print("Evaluating model performance on the 'unseen' test set...")
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f" > Model Mean Absolute Error (MAE): {mae:.2f}")
    print(f" > R-squared (R^2):               {r2:.4f} (or {r2*100:.2f}%)")

    # 9. Save the Final Model
    print(f"\nSaving the final trained model to '{MODEL_FILE}'...")
    joblib.dump(model, MODEL_FILE)

    print("\n" + "=" * 60)
    print("--- PIPELINE COMPLETE ---")
    print(f"All files ({MODEL_FILE}, {COLUMNS_FILE}, {TEST_SET_FILE}) are ready.")
    print("=" * 60)

if __name__ == "__main__":
    main()