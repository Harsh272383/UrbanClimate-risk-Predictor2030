from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import logging
import json

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "climate-risk-predictor-secret")

# Function to load climate data
def load_climate_data():
    try:
        # Try different file name variations and locations
        file_paths = [
            'urban_climate_risk_predictor_2030 2.csv',
            'urban_climate_risk_predictor_2030_2.csv',  # Alternate name without space
            'urban_climate_risk_2030.csv',
            'attached_assets/urban_climate_risk_predictor_2030 2.csv'
        ]
        
        for path in file_paths:
            try:
                df = pd.read_csv(path)
                logging.info(f"Successfully loaded data from {path}")
                return df
            except FileNotFoundError:
                continue
        
        # If no file is found, create sample data
        logging.warning("No CSV file found. Creating a sample dataset.")
        
        # Create a minimal dataset with the required columns
        data = {
            'Date': pd.date_range(start='2029-01-01', periods=20, freq='W'),
            'City_Zone': np.random.choice(['Zone A', 'Zone B', 'Zone C', 'Zone D'], 20),
            'Avg_Temperature_C': np.random.uniform(20, 40, 20),
            'Rainfall_mm': np.random.uniform(0, 50, 20),
            'Humidity_%': np.random.uniform(30, 90, 20),
            'Air_Quality_Index': np.random.uniform(50, 200, 20),
            'Flood_Risk_Score': np.random.uniform(0, 1, 20),
            'Heatwave_Risk_Score': np.random.uniform(0, 1, 20),
            'Drought_Risk_Score': np.random.uniform(0, 1, 20)
        }
        
        # Determine risk level based on features
        risk_labels = []
        for i in range(20):
            temp = data['Avg_Temperature_C'][i]
            aqi = data['Air_Quality_Index'][i]
            humidity = data['Humidity_%'][i]
            
            if temp > 38 or aqi > 150 or humidity < 35:
                risk_labels.append('High')
            elif temp > 33 or aqi > 100 or humidity < 45:
                risk_labels.append('Medium')
            else:
                risk_labels.append('Low')
        
        data['Risk_Label'] = risk_labels
        
        df = pd.DataFrame(data)
        
        # Save the CSV file for future use
        df.to_csv('urban_climate_risk_predictor_2030.csv', index=False)
        logging.info("Created and saved sample dataset")
        
        return df
    except Exception as e:
        logging.error(f"Error loading climate data: {str(e)}")
        return None

# Function to create visualizations
def create_visualizations(df):
    try:
        # Ensure the directory exists
        os.makedirs('static/assets', exist_ok=True)
        
        # 1. Create risk chart by zone
        plt.figure(figsize=(10, 6))
        risk_by_zone = df.groupby('City_Zone')['Risk_Label'].value_counts().unstack().fillna(0)
        risk_by_zone.plot(kind='bar', stacked=True, colormap='viridis')
        plt.title('Climate Risk Distribution by City Zone (2030)')
        plt.xlabel('City Zone')
        plt.ylabel('Number of Weeks')
        plt.legend(title='Risk Level')
        plt.tight_layout()
        plt.savefig('static/assets/risk_chart.svg', format='svg')
        plt.close()
        
        # 2. Temperature vs Risk Level
        plt.figure(figsize=(10, 6))
        risk_order = {'Low': 0, 'Medium': 1, 'High': 2}
        df['Risk_Value'] = df['Risk_Label'].map(risk_order)
        
        # Group by temperature ranges and calculate mean risk value
        temp_bins = np.arange(20, 45, 5)
        df['Temp_Range'] = pd.cut(df['Avg_Temperature_C'], temp_bins)
        temp_risk = df.groupby('Temp_Range')['Risk_Value'].mean().reset_index()
        
        plt.bar(temp_risk['Temp_Range'].astype(str), temp_risk['Risk_Value'], color='#6b48ff')
        plt.title('Average Risk Level by Temperature Range (2030)')
        plt.xlabel('Temperature Range (°C)')
        plt.ylabel('Average Risk Level (0=Low, 1=Medium, 2=High)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('static/assets/green_space_impact.svg', format='svg')
        plt.close()
        
        # 3. Create risk decision boundary visualization
        plt.figure(figsize=(10, 8))
        
        # Create a grid of temperature and air quality values
        temps = np.linspace(20, 42, 100)
        aqi = np.linspace(50, 200, 100)
        temp_grid, aqi_grid = np.meshgrid(temps, aqi)
        
        # Create a risk score based on these two variables
        risk_score = (
            (temp_grid > 38) * 2 +  # High temperature → High risk
            (temp_grid > 33) * 1 +  # Medium temperature → Medium risk
            (aqi_grid > 150) * 2 +  # Poor air quality → High risk
            (aqi_grid > 100) * 1    # Medium air quality → Medium risk
        ) / 2
        
        # Plot the risk heatmap
        plt.contourf(temp_grid, aqi_grid, risk_score, levels=10, cmap='RdYlGn_r', alpha=0.7)
        plt.colorbar(label='Risk Level (0=Low, 1=Medium, 2=High)')
        
        # Add decision boundaries
        plt.contour(temp_grid, aqi_grid, risk_score, levels=[0.5, 1.5], colors='white', linewidths=2)
        
        # Add annotations
        plt.text(25, 70, "LOW RISK", fontsize=14, color='black')
        plt.text(36, 70, "MEDIUM RISK", fontsize=14, color='black')
        plt.text(39, 175, "HIGH RISK", fontsize=14, color='white')
        
        # Add real data points if available
        if 'Avg_Temperature_C' in df.columns and 'Air_Quality_Index' in df.columns and 'Risk_Label' in df.columns:
            risk_colors = {'Low': 'green', 'Medium': 'yellow', 'High': 'red'}
            for risk in risk_colors:
                subset = df[df['Risk_Label'] == risk]
                plt.scatter(subset['Avg_Temperature_C'], subset['Air_Quality_Index'], 
                            c=risk_colors[risk], edgecolor='black', s=100, label=f'{risk} Risk')
            plt.legend(loc='upper left')
        
        plt.title('Risk Level Based on Temperature and Air Quality', fontsize=16)
        plt.xlabel('Temperature (°C)', fontsize=14)
        plt.ylabel('Air Quality Index', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save the visualization
        plt.tight_layout()
        plt.savefig('static/assets/risk_decision_boundary.png')
        
        logging.info("Visualizations created successfully")
    except Exception as e:
        logging.error(f"Error creating visualizations: {str(e)}")

# Load and prepare the climate data
df = load_climate_data()

# Create visualizations if data was loaded
if df is not None:
    create_visualizations(df)

# --- 1. Machine learning model ---
# Define feature names for when we need a fallback model
feature_names = ['Avg_Temperature_C', 'Rainfall_mm', 'Humidity_%', 'Air_Quality_Index', 
                'Flood_Risk_Score', 'Heatwave_Risk_Score', 'Drought_Risk_Score', 'City_Zone_Encoded']

# Use the loaded climate data CSV file for model training
if df is not None:
    # Convert categorical features
    df['City_Zone_Encoded'] = pd.Categorical(df['City_Zone']).codes
    
    # Map risk labels to numerical values
    risk_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    df['risk_level'] = df['Risk_Label'].map(risk_mapping)
    
    # Select features for model training
    X = df[['Avg_Temperature_C', 'Rainfall_mm', 'Humidity_%', 'Air_Quality_Index', 
            'Flood_Risk_Score', 'Heatwave_Risk_Score', 'Drought_Risk_Score', 'City_Zone_Encoded']]
    y = df['risk_level']
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    logging.info(f"Model trained successfully using {len(df)} data points from the CSV file")
    
    # Get feature importance for visualization (using DataFrame columns)
    feature_importance = dict(zip(X.columns, model.feature_importances_))
else:
    logging.error("Failed to load climate data for model training")
    # Create a simple model with default data as fallback
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Create a simple dataset for training
    X_simple = np.array([[30, 10, 60, 100, 0.5, 0.5, 0.5, 0]])
    y_simple = np.array([1])  # Medium risk
    model.fit(X_simple, y_simple)
    
    # Create a basic feature importance dict using predefined feature names
    feature_importance = dict(zip(feature_names, [0.125] * 8))  # Equal importance

# Sort feature importance for visualization
sorted_feature_importance = {k: v for k, v in sorted(feature_importance.items(), key=lambda item: item[1], reverse=True)}

# --- 2. Flask Routes ---
@app.route('/')
def index():
    feature_importance_json = json.dumps(sorted_feature_importance)
    return render_template('index.html', feature_importance=feature_importance_json)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        temp = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        rainfall = float(request.form['precipitation'])
        city_zone = request.form.get('city_zone', 'Zone A')  # Default to Zone A if not provided
        air_quality = float(request.form['air_quality'])
        flood_risk = float(request.form.get('flood_risk', 0.5))  # Default to 0.5 if not provided
        heatwave_risk = float(request.form.get('heatwave_risk', 0.5))  # Default to 0.5 if not provided
        drought_risk = float(request.form.get('drought_risk', 0.5))  # Default to 0.5 if not provided

        # Convert city zone to encoded value
        city_zones = ['Zone A', 'Zone B', 'Zone C', 'Zone D']
        if city_zone not in city_zones:
            city_zone = 'Zone A'  # Default if invalid
        city_zone_encoded = city_zones.index(city_zone)
        
        # Create feature array for prediction
        features = np.array([[temp, rainfall, humidity, air_quality, 
                             flood_risk, heatwave_risk, drought_risk, city_zone_encoded]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        prediction_proba = model.predict_proba(features)[0]
        
        # Map prediction to risk level
        risk_labels = ['Low Risk', 'Medium Risk', 'High Risk']
        result = risk_labels[prediction]
        
        # Generate recommendations based on input values
        recommendations = []
        if prediction > 0:  # Medium or High risk
            if temp > 33:
                recommendations.append("Implement cooling strategies such as green roofs and increased vegetation")
            if humidity < 40:
                recommendations.append("Develop water conservation policies and drought management plans")
            if air_quality > 150:
                recommendations.append("Reduce emissions through improved public transportation and clean energy")
            if flood_risk > 0.7:
                recommendations.append("Enhance flood defenses and improve drainage systems")
            if heatwave_risk > 0.7:
                recommendations.append("Create more shaded areas and cooling centers for vulnerable populations")
            if drought_risk > 0.7:
                recommendations.append("Implement water recycling and rainwater harvesting systems")
        
        # If no specific recommendations were added, add generic ones
        if not recommendations:
            if prediction == 1:  # Medium risk
                recommendations.append("Monitor climate conditions regularly and develop preparedness plans")
                recommendations.append("Increase urban green spaces to mitigate temperature increases")
            elif prediction == 2:  # High risk
                recommendations.append("Urgent action needed: Develop comprehensive climate adaptation strategy")
                recommendations.append("Invest in climate-resilient infrastructure to protect vulnerable areas")
        
        # Create prediction results for template
        prediction_results = {
            'risk_level': result,
            'probability': {
                'Low Risk': round(prediction_proba[0] * 100, 2),
                'Medium Risk': round(prediction_proba[1] * 100, 2),
                'High Risk': round(prediction_proba[2] * 100, 2)
            },
            'recommendations': recommendations,
            'input_values': {
                'Temperature': f"{temp}°C",
                'Humidity': f"{humidity}%",
                'Rainfall': f"{rainfall}mm",
                'Air Quality Index': air_quality,
                'City Zone': city_zone
            }
        }
        
        return render_template('index.html', 
                             prediction_text=f"Predicted Climate Risk: {result}",
                             prediction_results=prediction_results,
                             feature_importance=json.dumps(sorted_feature_importance))
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        return render_template('index.html', 
                             prediction_text=f"Error: {str(e)}",
                             feature_importance=json.dumps(sorted_feature_importance))

@app.route('/api/prediction_data', methods=['GET'])
def prediction_data():
    if df is not None:
        # Use the actual data from the CSV
        # Group by city zone to get data for different zones
        zones = df['City_Zone'].unique().tolist()
        zone_data = {}
        
        for zone in zones:
            zone_df = df[df['City_Zone'] == zone]
            # Calculate average values for visualization
            zone_data[zone] = {
                'avg_temperature': round(zone_df['Avg_Temperature_C'].mean(), 1),
                'avg_rainfall': round(zone_df['Rainfall_mm'].mean(), 1),
                'risk_distribution': zone_df['Risk_Label'].value_counts().to_dict()
            }
        
        # Create a data structure for the visualization
        data = {
            'zones': zones,
            'risk_levels': [zone_data[z]['risk_distribution'] for z in zones],
            'temperatures': [zone_data[z]['avg_temperature'] for z in zones],
            'precipitation': [zone_data[z]['avg_rainfall'] for z in zones],
            'zone_data': zone_data
        }
    else:
        # Fallback if CSV data is not available
        logging.warning("CSV data not available for API, using example data")
        zones = ['Zone A', 'Zone B', 'Zone C', 'Zone D']
        data = {
            'zones': zones,
            'temperatures': [36.2, 32.1, 29.3, 31.5],
            'precipitation': [12.3, 18.7, 22.1, 15.6]
        }
    
    return jsonify(data)

if __name__ == '__main__':
    app.run(host='192.168.1.27', port=5000, debug=True)