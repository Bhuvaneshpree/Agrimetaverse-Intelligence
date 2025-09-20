import pandas as pd
import numpy as np
from datetime import datetime

# Create enhanced dataset with external features
def create_enhanced_dataset():
    """
    Create a synthetic dataset with external features for agricultural price forecasting
    """
    
    # Years from 2014 to 2024
    years = list(range(2014, 2025))
    
    # Simulated external features (in real scenario, you'd collect real data)
    external_features = {
        'Year': years,
        
        # Weather Data
        'Rainfall_mm': [850, 920, 780, 1100, 650, 980, 1200, 750, 890, 1050, 820],  # Annual rainfall
        'Avg_Temperature_C': [25.2, 26.1, 24.8, 25.9, 26.5, 25.0, 24.6, 26.8, 25.7, 25.3, 26.0],  # Average temperature
        'Drought_Index': [0.2, 0.1, 0.4, 0.0, 0.6, 0.1, 0.0, 0.5, 0.2, 0.1, 0.3],  # 0-1 scale
        
        # Economic Indicators
        'GDP_Growth_%': [7.4, 8.0, 8.3, 6.8, 4.0, 6.5, -7.3, 8.9, 7.0, 6.1, 6.8],  # GDP growth rate
        'Inflation_%': [6.0, 5.9, 2.0, 3.4, 4.5, 6.2, 6.6, 5.1, 6.7, 5.0, 5.4],  # Inflation rate
        'USD_INR_Rate': [60.5, 64.1, 67.2, 68.4, 74.4, 74.9, 76.9, 79.4, 82.8, 83.1, 84.2],  # Exchange rate
        'Crude_Oil_Price': [105, 53, 44, 43, 69, 64, 43, 71, 95, 82, 75],  # USD per barrel
        
        # Agricultural Factors
        'Fertilizer_Price_Index': [100, 95, 85, 90, 120, 115, 110, 140, 160, 155, 150],  # Base year 2014=100
        'Fuel_Price_Rs/L': [55, 52, 48, 58, 72, 68, 65, 78, 95, 88, 85],  # Diesel price
        'Minimum_Support_Price_Index': [100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150],  # MSP index
        
        # Market Factors
        'Export_Volume_Index': [100, 110, 95, 120, 80, 105, 70, 125, 140, 130, 115],  # Export volume
        'Storage_Capacity_Index': [100, 102, 105, 108, 110, 112, 115, 118, 120, 122, 125],  # Storage facilities
        'Transport_Cost_Index': [100, 103, 98, 112, 125, 118, 115, 130, 145, 140, 135]  # Transportation costs
    }
    
    return pd.DataFrame(external_features)

# Create the enhanced dataset
enhanced_df = create_enhanced_dataset()

# Save to CSV
enhanced_df.to_csv('enhanced_features.csv', index=False)

print("Enhanced dataset created with the following features:")
print(enhanced_df.columns.tolist())
print("\nFirst 5 rows:")
print(enhanced_df.head())
print("\nDataset shape:", enhanced_df.shape)