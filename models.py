from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

class ClimateRiskModel:
    def __init__(self):
        self.model = None
        self.features = None
        
    def train(self, data):
        """
        Train the climate risk model using the provided data
        
        Parameters:
        -----------
        data : pandas.DataFrame
            The climate data with features and target variable
        """
        # Define features and target
        X = data[['Avg_Temperature_C', 'Rainfall_mm', 'Humidity_%', 'Air_Quality_Index', 
                 'Flood_Risk_Score', 'Heatwave_Risk_Score', 'Drought_Risk_Score']]
        
        # Convert risk labels to numeric values
        risk_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
        y = data['Risk_Label'].map(risk_mapping)
        
        # Train the model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        self.features = X.columns.tolist()
        
        # Return feature importance
        return dict(zip(self.features, self.model.feature_importances_))
    
    def predict(self, features):
        """
        Make a prediction using the trained model
        
        Parameters:
        -----------
        features : numpy.ndarray or pandas.DataFrame
            The input features for prediction
            
        Returns:
        --------
        tuple (prediction, probabilities)
            The predicted risk level and associated probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        risk_labels = ['Low Risk', 'Medium Risk', 'High Risk']
        result = risk_labels[prediction]
        
        # Create probability dictionary
        prob_dict = {
            'Low Risk': round(probabilities[0] * 100, 2),
            'Medium Risk': round(probabilities[1] * 100, 2),
            'High Risk': round(probabilities[2] * 100, 2)
        }
        
        return result, prob_dict
    
    def get_recommendations(self, features):
        """
        Generate recommendations based on input features
        
        Parameters:
        -----------
        features : dict
            Dictionary containing feature values
            
        Returns:
        --------
        list
            List of recommendations based on the features
        """
        recommendations = []
        
        if features['Avg_Temperature_C'] > 35:
            recommendations.append("Implement cooling strategies such as green roofs and increased vegetation")
            
        if features['Humidity_%'] < 40:
            recommendations.append("Develop water conservation policies and drought management plans")
        elif features['Humidity_%'] > 80:
            recommendations.append("Implement dehumidification measures to prevent mold and infrastructure damage")
            
        if features['Air_Quality_Index'] > 150:
            recommendations.append("Reduce emissions through improved public transportation and clean energy")
            
        if features['Flood_Risk_Score'] > 0.7:
            recommendations.append("Enhance flood defenses and improve drainage systems")
            
        if features['Heatwave_Risk_Score'] > 0.7:
            recommendations.append("Create more shaded areas and cooling centers for vulnerable populations")
            
        return recommendations