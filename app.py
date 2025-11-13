# app.py
#
# A simple Flask web server to host the traffic prediction model.
# v5.2: "Glassmorphism" Dark Mode UI with error rate display.
#
# --- SETUP (Terminal) ---
# 1. Install required libraries:
#    pip install -r requirements.txt
#
# --- HOW TO RUN ---
# 1. Make sure 'traffic_model.joblib', 'model_columns.json', 
#    and 'Metro_Interstate_Traffic_Volume.csv' are in the same folder.
# 2. Run this script in your terminal:
#    python app.py
# 3. Open your web browser and go to:
#    http://127.0.0.1:5000

import joblib
import json
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from datetime import datetime

# --- 1. INITIALIZE THE APP ---
app = Flask(__name__)

# --- 2. LOAD MODELS AND DATA ON STARTUP ---
print("Loading model, columns, and default values...")
try:
    # Load the trained model
    model = joblib.load("traffic_model.joblib")
    
    # Load the column blueprint
    with open('model_columns.json', 'r') as f:
        model_columns = json.load(f)
    
    # Load the *original data* to calculate averages AND thresholds
    data_df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")
    
    # Calculate averages to use as defaults for optional fields
    DEFAULT_VALUES = {
        "temp": data_df['temp'].mean(),
        "rain_1h": data_df['rain_1h'].mean(),
        "snow_1h": data_df['snow_1h'].mean(),
        "clouds_all": data_df['clouds_all'].mean()
    }
    # Convert temp from Kelvin to Celsius for user display
    DEFAULT_TEMP_KELVIN = DEFAULT_VALUES['temp']
    DEFAULT_TEMP_CELSIUS = DEFAULT_TEMP_KELVIN - 273.15
    
    # --- Calculate Traffic Thresholds ---
    TRAFFIC_THRESHOLD_LIGHT = data_df['traffic_volume'].quantile(0.33)
    TRAFFIC_THRESHOLD_HEAVY = data_df['traffic_volume'].quantile(0.67)
    
    print(f"Default Temp (K): {DEFAULT_TEMP_KELVIN:.2f}")
    print(f"Default Rain (mm): {DEFAULT_VALUES['rain_1h']:.2f}")
    print(f"--- Traffic Thresholds ---")
    print(f"  Light: < {TRAFFIC_THRESHOLD_LIGHT:.0f} vehicles")
    print(f"  Avg:   {TRAFFIC_THRESHOLD_LIGHT:.0f} - {TRAFFIC_THRESHOLD_HEAVY:.0f} vehicles")
    print(f"  Heavy: > {TRAFFIC_THRESHOLD_HEAVY:.0f} vehicles")
    
except FileNotFoundError as e:
    print(f"--- ERROR: FILE NOT FOUND ---")
    print(f"Missing file: {e.filename}")
    print("Please make sure 'traffic_model.joblib', 'model_columns.json', and 'Metro_Interstate_Traffic_Volume.csv' are in this folder.")
    exit()
except Exception as e:
    print(f"Error loading files: {e}")
    exit()

print("...Models and data loaded successfully. Starting server.")

# --- 3. DEFINE DROPDOWN OPTIONS (from model columns) ---
def extract_options(prefix, columns):
    """Helper to get dropdown options from one-hot-encoded column names."""
    options = []
    for col in columns:
        if col.startswith(prefix):
            options.append(col[len(prefix):])
    return options

HOLIDAY_OPTIONS = ["None"] + extract_options("holiday_", model_columns)
WEATHER_MAIN_OPTIONS = extract_options("weather_main_", model_columns)
WEATHER_DESC_OPTIONS = extract_options("weather_description_", model_columns)


# --- HELPER FUNCTIONS ---
def get_traffic_category(volume):
    """Classifies the numerical volume into a category."""
    if volume <= TRAFFIC_THRESHOLD_LIGHT:
        return "Light"
    elif volume <= TRAFFIC_THRESHOLD_HEAVY:
        return "Average"
    else:
        return "Heavy"

# --- 4. DEFINE THE PREDICTION LOGIC (THE "API") ---
@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests from the webpage, processes inputs,
    and returns a JSON prediction.
    """
    try:
        data = request.json
        
        # Start with a "row" of all zeros, matching the model's training
        data_row = {col: 0 for col in model_columns}

        # --- A. Process Date/Time ---
        try:
            # Standard HTML date input format
            date_obj = datetime.strptime(data['date'], '%Y-%m-%d')
        except ValueError:
            # Fallback for other potential formats like ISO (e.g., from JS new Date())
            date_obj = datetime.strptime(data['date'].split('T')[0], '%Y-%m-%d')
            
        data_row['day_of_week'] = date_obj.weekday()
        data_row['month'] = date_obj.month
        data_row['year'] = date_obj.year
        data_row['hour'] = int(data['hour'])

        # --- B. Process Optional Numeric Inputs ---
        
        # Handle Temperature (convert C to K)
        if data['temp_c'] == "":
            data_row['temp'] = DEFAULT_VALUES['temp'] # Use pre-calculated default (in Kelvin)
        else:
            data_row['temp'] = float(data['temp_c']) + 273.15 # Convert to Kelvin
            
        # Handle Rain, Snow, Clouds
        data_row['rain_1h'] = float(data['rain_1h']) if data['rain_1h'] else DEFAULT_VALUES['rain_1h']
        data_row['snow_1h'] = float(data['snow_1h']) if data['snow_1h'] else DEFAULT_VALUES['snow_1h']
        data_row['clouds_all'] = int(data['clouds_all']) if data['clouds_all'] else DEFAULT_VALUES['clouds_all']

        # --- C. Process Categorical (Dropdown) Inputs ---
        
        # Holiday
        holiday = data['holiday']
        if holiday != "None":
            col_name = f"holiday_{holiday}"
            if col_name in data_row:
                data_row[col_name] = 1
        
        # Weather Main
        weather_main = data['weather_main']
        col_name = f"weather_main_{weather_main}"
        if col_name in data_row:
            data_row[col_name] = 1

        # Weather Description
        weather_desc = data['weather_desc']
        col_name = f"weather_description_{weather_desc}"
        if col_name in data_row:
            data_row[col_name] = 1

        # --- D. Make Prediction ---
        
        # Convert the dictionary to a pandas DataFrame
        input_df = pd.DataFrame([data_row], columns=model_columns)
        
        # Get the prediction
        prediction = model.predict(input_df)
        predicted_traffic = int(prediction[0])
        
        # Get the category
        traffic_category = get_traffic_category(predicted_traffic)

        # --- E. Send Result Back ---
        return jsonify({
            "success": True,
            "prediction": predicted_traffic,
            "category": traffic_category
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# --- 5. DEFINE THE WEBPAGE (THE "FRONTEND") ---
@app.route("/")
def home():
    """
    Serves the main HTML webpage to the user's browser.
    """
    default_date = datetime.now().strftime('%Y-%m-%d')
    # Pass all dynamic data into the template
    return render_template_string(HTML_TEMPLATE, 
        holiday_options=HOLIDAY_OPTIONS,
        weather_main_options=WEATHER_MAIN_OPTIONS,
        weather_desc_options=WEATHER_DESC_OPTIONS,
        default_temp_c=f"{DEFAULT_TEMP_CELSIUS:.2f}",
        default_date=default_date
    )

# --- 6. DEFINE THE HTML, CSS, AND JAVASCRIPT FOR THE WEBPAGE ---

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Traffic Volume Predictor</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { 
            font-family: 'Inter', sans-serif;
            background-color: #0f172a; /* bg-slate-900 */
            color: #e2e8f0; /* text-slate-200 */
            overflow: hidden; /* Hide scrollbars */
        }
        
        /* Create the glowing background effect */
        body::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            background: 
                radial-gradient(circle at 10% 20%, rgba(22, 163, 74, 0.1), transparent 40%),
                radial-gradient(circle at 80% 30%, rgba(56, 189, 248, 0.15), transparent 40%),
                radial-gradient(circle at 20% 80%, rgba(219, 39, 119, 0.15), transparent 40%);
            filter: blur(100px);
            z-index: -1;
        }

        .loader {
            border-top-color: #06b6d4; /* cyan-500 */
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Dark mode date picker icon */
        input[type="date"]::-webkit-calendar-picker-indicator {
            cursor: pointer;
            filter: invert(1) opacity(0.5);
            transition: opacity 0.2s;
        }
        input[type="date"]::-webkit-calendar-picker-indicator:hover {
            opacity: 0.8;
        }
        
        /* Custom scrollbar for form */
        .form-column::-webkit-scrollbar {
            width: 6px;
        }
        .form-column::-webkit-scrollbar-track {
            background: transparent;
        }
        .form-column::-webkit-scrollbar-thumb {
            background: #475569; /* slate-600 */
            border-radius: 3px;
        }
    </style>
</head>
<body class="min-h-screen p-4 sm:p-8 flex items-center justify-center">

    <!-- Glassmorphism Main Card -->
    <main class="w-full max-w-6xl h-[90vh] bg-slate-800/60 backdrop-blur-2xl border border-slate-700 rounded-2xl shadow-2xl overflow-hidden">
        <div class="grid grid-cols-1 lg:grid-cols-5 h-full">
            
            <!-- ====== LEFT COLUMN: FORM ====== -->
            <div class="lg:col-span-3 p-8 sm:p-12 h-full overflow-y-auto form-column">
                <!-- Header -->
                <div class="flex items-center mb-10">
                    <span class="inline-block p-3 bg-cyan-900/50 rounded-lg border border-cyan-700/50">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor" class="w-8 h-8 text-cyan-400">
                            <path stroke-linecap="round" stroke-linejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25C3.504 21 3 20.496 3 19.875v-6.75zM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25c-.621 0-1.125-.504-1.125-1.125V8.625zM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25c-.621 0-1.125-.504-1.125-1.125V4.125z" />
                        </svg>                          
                    </span>
                    <div class="ml-4">
                        <h1 class="text-3xl font-bold text-white">Traffic Forecast</h1>
                        <p class="text-lg text-slate-400">Real-time ML Prediction</p>
                    </div>
                </div>

                <!-- Form -->
                <form id="prediction-form" class="space-y-10">

                    <!-- Section 1: Core Details -->
                    <div>
                        <h2 class="text-xl font-semibold text-white border-b border-slate-700 pb-2 mb-6">
                            Core Information
                        </h2>
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <!-- Date -->
                            <div>
                                <label for="date" class="block text-sm font-medium text-slate-400">Date</label>
                                <input type="date" id="date" value="{{default_date}}" required
                                       class="mt-1 block w-full rounded-md bg-slate-900/50 border-slate-700 text-white shadow-sm focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/50">
                            </div>
                            <!-- Hour -->
                            <div>
                                <label for="hour" class="block text-sm font-medium text-slate-400">Hour of Day (0-23)</label>
                                <input type="number" id="hour" min="0" max="23" value="17" required
                                       class="mt-1 block w-full rounded-md bg-slate-900/50 border-slate-700 text-white shadow-sm focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/50">
                            </div>
                        </div>
                    </div>

                    <!-- Section 2: Weather Details -->
                    <div>
                        <h2 class="text-xl font-semibold text-white border-b border-slate-700 pb-2 mb-6">
                            Weather (Optional)
                        </h2>
                        <p class="text-sm text-slate-500 -mt-4 mb-6">
                            Leave blank to use data averages (e.g., {{default_temp_c}}°C).
                        </p>
                        <div class="grid grid-cols-2 md:grid-cols-4 gap-6">
                            <!-- Temp -->
                            <div>
                                <label for="temp_c" class="block text-sm font-medium text-slate-400">Temp (°C)</label>
                                <input type="number" step="0.01" id="temp_c" 
                                       placeholder="{{default_temp_c}}"
                                       class="mt-1 block w-full rounded-md bg-slate-900/50 border-slate-700 text-white shadow-sm focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/50">
                            </div>
                            <!-- Rain -->
                            <div>
                                <label for="rain_1h" class="block text-sm font-medium text-slate-400">Rain (mm/hr)</label>
                                <input type="number" step="0.1" min="0" id="rain_1h" placeholder="0.0"
                                       class="mt-1 block w-full rounded-md bg-slate-900/50 border-slate-700 text-white shadow-sm focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/50">
                            </div>
                            <!-- Snow -->
                            <div>
                                <label for="snow_1h" class="block text-sm font-medium text-slate-400">Snow (mm/hr)</label>

                                <input type="number" step="0.1" min="0" id="snow_1h" placeholder="0.0"
                                       class="mt-1 block w-full rounded-md bg-slate-900/50 border-slate-700 text-white shadow-sm focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/50">
                            </div>
                            <!-- Clouds -->
                            <div>
                                <label for="clouds_all" class="block text-sm font-medium text-slate-400">Clouds (%)</label>
                                <input type="number" min="0" max="100" id="clouds_all" placeholder="40"
                                       class="mt-1 block w-full rounded-md bg-slate-900/50 border-slate-700 text-white shadow-sm focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/50">
                            </div>
                        </div>
                    </div>

                    <!-- Section 3: Categories -->
                    <div>
                        <h2 class="text-xl font-semibold text-white border-b border-slate-700 pb-2 mb-6">
                            Conditions
                        </h2>
                        <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                            <!-- Holiday -->
                            <div>
                                <label for="holiday" class="block text-sm font-medium text-slate-400">Holiday</label>
                                <select id="holiday" required
                                        class="mt-1 block w-full rounded-md bg-slate-900/50 border-slate-700 text-white shadow-sm focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/50">
                                    {% for option in holiday_options %}
                                        <option value="{{option}}">{{option}}</option>
                                    {% endfor %}
                                </select>
                            </div>
                            <!-- Weather Main -->
                            <div>
                                <label for="weather_main" class="block text-sm font-medium text-slate-400">Weather Type</label>
                                <select id="weather_main" required
                                        class="mt-1 block w-full rounded-md bg-slate-900/50 border-slate-700 text-white shadow-sm focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/50">
                                    {% for option in weather_main_options %}
                                        <option value="{{option}}" {% if option == 'Clouds' %}selected{% endif %}>
                                            {{option}}
                                        </option>
                                    {% endfor %}
                                </select>
                            </div>
                            <!-- Weather Desc -->
                            <div>
                                <label for="weather_desc" class="block text-sm font-medium text-slate-400">Weather Detail</label>
                                <select id="weather_desc" required
                                        class="mt-1 block w-full rounded-md bg-slate-900/50 border-slate-700 text-white shadow-sm focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/50">
                                    {% for option in weather_desc_options %}
                                        <option value="{{option}}" {% if option == 'broken clouds' %}selected{% endif %}>
                                            {{option}}
                                        </option>
                                    {% endfor %}
                                </select>
                            </div>
                        </div>
                    </div>
                </form>
            </div>

            <!-- ====== RIGHT COLUMN: CONTROLS & RESULT ====== -->
            <div class="lg:col-span-2 bg-black/10 p-8 sm:p-12 flex flex-col justify-center border-l border-slate-700/50">
                
                <!-- Submit Button -->
                <button type="submit" form="prediction-form" id="submit-button"
                        class="w-full bg-cyan-500 text-slate-900 font-bold py-4 px-6 rounded-lg shadow-lg
                               hover:bg-cyan-400 focus:outline-none focus:ring-4 focus:ring-cyan-500/50
                               transition duration-300 ease-in-out text-lg">
                    Forecast Traffic
                </button>
                
                <!-- Result Display -->
                <div id="result-container" class="mt-8">
                    <div id="result-message" class="text-center w-full min-h-[200px] flex flex-col items-center justify-center
                                                    bg-slate-900/50 backdrop-blur-sm border border-slate-700 p-6 rounded-lg shadow-inner">
                        <p class="text-slate-400">Your forecast will appear here.</p>
                    </div>
                </div>
            </div>

        </div>
    </main>

    <script>
        const form = document.getElementById('prediction-form');
        const submitButton = document.getElementById('submit-button');
        const resultMessage = document.getElementById('result-message');

        form.addEventListener('submit', async (e) => {
            e.preventDefault(); // Stop the form from submitting normally
            
            // Show loading state
            resultMessage.innerHTML = `
                <div class="loader w-10 h-10 rounded-full border-4 border-slate-700"></div>
                <p class="text-slate-400 mt-4">Forecasting...</p>
            `;
            submitButton.disabled = true;
            submitButton.classList.add('opacity-50', 'cursor-not-allowed');

            // 1. Collect all data from the form
            const formData = {
                date: document.getElementById('date').value,
                hour: document.getElementById('hour').value,
                temp_c: document.getElementById('temp_c').value,
                rain_1h: document.getElementById('rain_1h').value,
                snow_1h: document.getElementById('snow_1h').value,
                clouds_all: document.getElementById('clouds_all').value,
                holiday: document.getElementById('holiday').value,
                weather_main: document.getElementById('weather_main').value,
                weather_desc: document.getElementById('weather_desc').value,
            };

            try {
                // 2. Send data to the /predict endpoint on our server
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(formData),
                });

                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }

                const result = await response.json();

                // 3. Display the result
                if (result.success) {
                    let category = result.category;
                    let prediction = result.prediction;
                    
                    // --- Dark Mode Colors ---
                    let colorClass = 'text-cyan-400';
                    let badgeColor = 'bg-cyan-900/50 text-cyan-300 border border-cyan-800';

                    if (category === 'Light') {
                        colorClass = 'text-green-400';
                        badgeColor = 'bg-green-900/50 text-green-300 border border-green-800';
                    } else if (category === 'Average') {
                        colorClass = 'text-orange-400';
                        badgeColor = 'bg-orange-900/50 text-orange-300 border border-orange-800';
                    } else if (category === 'Heavy') {
                        colorClass = 'text-red-400';
                        badgeColor = 'bg-red-900/50 text-red-300 border border-red-800';
                    }

                    resultMessage.innerHTML = `
                        <div class="flex flex-col items-center">
                            <span class="text-lg font-medium ${badgeColor} px-4 py-1 rounded-full">
                                ${category} Traffic
                            </span>
                            <div class="mt-4 text-lg ${colorClass}">
                                Predicted Volume: 
                            </div>
                            <strong class="text-6xl font-bold ${colorClass}">${prediction.toLocaleString()}</strong> 
                            <span class="text-lg ${colorClass} -mt-1">vehicles/hr</span>
                        </div>
                    `;
                    
                } else {
                    throw new Error(result.error);
                }

            } catch (error) {
                console.error('Prediction Error:', error);
                resultMessage.innerHTML = `
                    <span class="text-red-400 font-semibold">Error: Could not get prediction.</span>
                    <p class="text-sm text-slate-400 mt-2">Please check your inputs or the server logs.</p>
                `;
            } finally {
                // Re-enable the button
                submitButton.disabled = false;
                submitButton.classList.remove('opacity-50', 'cursor-not-allowed');
            }
        });
    </script> 

</body>
</html>
"""

# --- 7. RUN THE FLASK APP ---
if __name__ == "__main__":
    # Runs the web server
    # debug=True means the server will auto-reload when you save changes
    app.run(debug=True)

