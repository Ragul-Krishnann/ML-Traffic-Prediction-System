# view_importance.py

import joblib
import pandas as pd
import json
import matplotlib.pyplot as plt

# --- 1. LOAD MODEL AND COLUMN BLUEPRINT ---
print("Loading model and column blueprint...")
try:
    model = joblib.load("traffic_model.joblib")
    
    with open('model_columns.json', 'r') as f:
        model_columns = json.load(f)
        
except FileNotFoundError:
    print("\n--- ERROR ---")
    print("Files ('traffic_model.joblib', 'model_columns.json') not found.")
    print("Please run train_model.py first to create these files.")
    exit()

print("Model loaded successfully.")

# --- 2. EXTRACT AND MAP FEATURE IMPORTANCES ---
try:
    # Get the array of importance scores from the trained model
    importances = model.feature_importances_
    
    # Create a DataFrame to map names to scores
    feature_importance_df = pd.DataFrame({
        'Feature': model_columns,
        'Importance': importances
    })

except AttributeError:
    print("\n--- ERROR ---")
    print("The saved model is not a tree-based model (like Random Forest or Gradient Boosting).")
    print("Cannot extract feature importances.")
    exit()

# --- 3. AGGREGATE ONE-HOT ENCODED FEATURES ---
print("Aggregating feature importances...")

# Define the base features that were NOT one-hot encoded
base_features = [
    'temp', 'rain_1h', 'snow_1h', 'clouds_all', 
    'hour', 'day_of_week', 'month', 'year'
]

# Define the features that WERE one-hot encoded
categorical_prefixes = ['holiday', 'weather_main', 'weather_description']

# Create a dictionary to hold the aggregated scores
aggregated_importances = {}

# Add the base features directly
for feature in base_features:
    score = feature_importance_df[feature_importance_df['Feature'] == feature]['Importance'].sum()
    aggregated_importances[feature] = score

# Sum up the scores for the categorical features
for prefix in categorical_prefixes:
    # Find all columns that start with this prefix (e.g., "holiday_")
    categorical_features = feature_importance_df['Feature'].str.startswith(f"{prefix}_")
    
    # Sum their importance scores
    total_importance = feature_importance_df[categorical_features]['Importance'].sum()
    
    # Store the total score
    aggregated_importances[prefix] = total_importance

# --- 4. CREATE AND SAVE THE PLOT ---
print("Generating plot...")

# Convert the dictionary to a pandas Series for easy plotting
final_importance_series = pd.Series(aggregated_importances)

# Sort from lowest to highest (for a nice horizontal bar chart)
final_importance_series = final_importance_series.sort_values(ascending=True)

# Create the plot
plt.figure(figsize=(10, 8))  # Set figure size
final_importance_series.plot(kind='barh', color='skyblue') # 'barh' = horizontal bar

plt.title('Feature Importance for Traffic Volume Prediction')
plt.xlabel('Importance Score (Higher is More Important)')
plt.ylabel('Feature Group')

# Ensure layout is not cramped
plt.tight_layout()

# Save the plot to a file
output_filename = "feature_importance.png"
plt.savefig(output_filename)

print("\n" + "="*50)
print(f"SUCCESS! Chart saved as: {output_filename}")
print("="*50)
print("Open the 'feature_importance.png' file to see your graph.")