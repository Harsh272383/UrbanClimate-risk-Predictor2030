import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def explain_risk_prediction():
    """
    This function explains how the model predicts risk levels and creates an example visualization
    """
    print("URBAN CLIMATE RISK PREDICTION LOGIC")
    print("===================================")
    
    # 1. Load or create example data
    try:
        df = pd.read_csv('urban_climate_risk_predictor_2030 2.csv')
        print(f"Loaded real data with {len(df)} entries")
    except:
        print("Creating example data")
        # Create synthetic dataset if real data not available
        df = pd.DataFrame({
            'Avg_Temperature_C': [25, 30, 33, 36, 39],
            'Rainfall_mm': [10, 5, 2, 1, 0.5],
            'Humidity_%': [70, 60, 50, 40, 30],
            'Air_Quality_Index': [80, 100, 150, 180, 200],
            'Flood_Risk_Score': [0.2, 0.3, 0.4, 0.6, 0.8],
            'Heatwave_Risk_Score': [0.1, 0.3, 0.5, 0.7, 0.9],
            'Drought_Risk_Score': [0.1, 0.2, 0.5, 0.7, 0.9],
            'City_Zone': ['Zone A', 'Zone B', 'Zone C', 'Zone D', 'Zone A'],
            'Risk_Label': ['Low', 'Low', 'Medium', 'High', 'High']
        })
    
    # 2. Explain the features used for prediction
    print("\n1. FEATURES USED FOR PREDICTION:")
    print("--------------------------------")
    print("Temperature: Higher temperatures increase risk")
    print("Rainfall: Lower rainfall often increases risk (drought conditions)")
    print("Humidity: Very low humidity can increase fire risk")
    print("Air Quality Index: Higher values (worse air quality) increase risk")
    print("Flood Risk Score: Higher values indicate higher flood risk")
    print("Heatwave Risk Score: Higher values indicate higher heatwave risk")
    print("Drought Risk Score: Higher values indicate higher drought risk")
    print("Urban Area Type: Urban areas typically have higher risk than rural areas")
    print("Green Space: More green space typically reduces risk")
    
    # 3. Explain how the model determines risk level
    print("\n2. HOW THE MODEL DETERMINES RISK LEVEL:")
    print("--------------------------------------")
    print("The model analyzes patterns in historical data to find relationships between features and risk levels.")
    print("It identifies thresholds and combinations of factors that lead to different risk categories.")
    print("For example:")
    print("- If temperature > 38°C AND humidity < 40% → Likely HIGH RISK")
    print("- If air quality index > 150 → Likely HIGH RISK")
    print("- If temperature 33-37°C → Likely MEDIUM RISK")
    print("- If temperature < 30°C AND good air quality AND adequate rainfall → Likely LOW RISK")
    
    # 4. Demonstrate with specific examples
    print("\n3. EXAMPLES OF RISK PREDICTION:")
    print("------------------------------")
    sample_data = [
        {'temp': 40, 'rain': 2, 'humidity': 35, 'aqi': 180, 'urban': 'Urban', 'green': 'Low', 'risk': 'HIGH'},
        {'temp': 34, 'rain': 8, 'humidity': 55, 'aqi': 120, 'urban': 'Urban', 'green': 'Medium', 'risk': 'MEDIUM'},
        {'temp': 28, 'rain': 15, 'humidity': 65, 'aqi': 75, 'urban': 'Rural', 'green': 'High', 'risk': 'LOW'}
    ]
    
    for i, sample in enumerate(sample_data):
        print(f"\nExample {i+1}:")
        print(f"Temperature: {sample['temp']}°C")
        print(f"Rainfall: {sample['rain']} mm")
        print(f"Humidity: {sample['humidity']}%")
        print(f"Air Quality Index: {sample['aqi']}")
        print(f"Area Type: {sample['urban']}")
        print(f"Green Space: {sample['green']}")
        print(f"Predicted Risk: {sample['risk']}")
        
        if i == 0:
            print("Explanation: High temperature combined with low humidity, low rainfall, poor air quality, ")
            print("            urban setting, and low green space create extreme climate vulnerability.")
        elif i == 1:
            print("Explanation: Moderately high temperature with average humidity and medium green space")
            print("            presents a moderate climate risk.")
        else:
            print("Explanation: Lower temperature with good humidity, rainfall, and air quality in a rural")
            print("            area with high green space results in a low climate risk.")
    
    # 5. Create and save a visualization showing the decision boundary
    try:
        # Create a visualization showing how temperature and air quality affect risk
        plt.figure(figsize=(10, 8))
        
        # Create a grid of temperature and air quality values
        temps = np.linspace(20, 42, 100)
        aqi = np.linspace(50, 200, 100)
        temp_grid, aqi_grid = np.meshgrid(temps, aqi)
        
        # Create a risk score based on these two variables
        # This is a simplified version of what the model might learn
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
        print("\nCreated risk decision boundary visualization!")
    except Exception as e:
        print(f"Could not create visualization: {str(e)}")
    
    # 6. Explain the machine learning behind it
    print("\n4. MACHINE LEARNING BEHIND THE PREDICTION:")
    print("----------------------------------------")
    print("The model uses Random Forest, an ensemble learning method that:")
    print("1. Creates multiple decision trees using random subsets of data and features")
    print("2. Each tree makes a prediction based on the feature values")
    print("3. The final prediction is determined by majority vote of all trees")
    print("4. This improves accuracy and reduces overfitting compared to a single decision tree")
    print("5. The model can calculate a probability for each risk level (Low, Medium, High)")
    
    # Show how to use the model with new data
    print("\n5. USING THE MODEL WITH NEW DATA:")
    print("---------------------------------")
    print("When you enter values in the web application:")
    print("1. Your input is scaled to match the training data distribution")
    print("2. Categorical inputs (like Urban/Rural) are encoded as numbers using one-hot encoding or label encoding")
    print("3. The model processes these features through its decision trees")
    print("4. Each tree votes for a risk level")
    print("5. The final prediction is the most common vote")
    print("6. The model also provides probability scores for each risk level")
    
    # Summary
    print("\nSUMMARY:")
    print("--------")
    print("The model learns patterns from historical climate data to identify combinations")
    print("of factors that lead to different risk levels. When given new input, it compares")
    print("those values to the patterns it learned to predict the most likely risk category.")

if __name__ == "__main__":
    explain_risk_prediction()