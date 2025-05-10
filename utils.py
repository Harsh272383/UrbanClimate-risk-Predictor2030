import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import logging

def load_climate_data():
    """
    Load the climate data from CSV file
    """
    try:
        # First try in the current directory
        df = pd.read_csv('urban_climate_risk_predictor_2030 2.csv')
        logging.info("Loaded CSV from current directory")
        return df
    except FileNotFoundError:
        try:
            # Try in the attached_assets directory
            df = pd.read_csv('attached_assets/urban_climate_risk_predictor_2030 2.csv')
            logging.info("Loaded CSV from attached_assets directory")
            return df
        except FileNotFoundError:
            logging.warning("CSV file not found in current or attached_assets directory")
            raise FileNotFoundError("Climate data CSV file not found")

def create_visualizations(df):
    """
    Create static visualizations for the dashboard
    """
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
    temp_risk = df.groupby('Temp_Range', dropna=False)['Risk_Value'].mean().reset_index()
    temp_risk['Temp_Range'] = temp_risk['Temp_Range'].cat.add_categories('No Data').fillna('No Data')
    
    plt.bar(temp_risk['Temp_Range'].astype(str), temp_risk['Risk_Value'], color='#6b48ff')
    plt.title('Average Risk Level by Temperature Range (2030)')
    plt.xlabel('Temperature Range (°C)')
    plt.ylabel('Average Risk Level (0=Low, 1=Medium, 2=High)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/assets/green_space_impact.svg', format='svg')
    plt.close()