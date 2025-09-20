import pandas as pd
import numpy as np

def smooth_extreme_price_changes():
    """
    Create a smoothed version of the dataset with extreme price changes moderated
    """
    # Load original data
    df = pd.read_csv("DatasetSIH1647.csv")
    df.set_index('Commodities', inplace=True)
    df = df.T
    df.index = pd.date_range(start='2014', periods=len(df), freq='YE')
    
    # Create a copy for smoothing
    df_smoothed = df.copy()
    
    print("🔧 SMOOTHING EXTREME PRICE CHANGES...")
    
    # Define maximum acceptable year-over-year change (30%)
    max_change = 0.30
    
    for commodity in df.columns:
        print(f"\nProcessing {commodity}:")
        
        for i in range(1, len(df)):
            prev_price = df_smoothed[commodity].iloc[i-1]
            current_price = df[commodity].iloc[i]
            
            # Calculate percentage change
            change = (current_price - prev_price) / prev_price
            
            if abs(change) > max_change:
                # Cap the change to max_change
                if change > 0:
                    smoothed_price = prev_price * (1 + max_change)
                    print(f"  📉 {df.index[i].year}: Reduced from ₹{current_price:.2f} to ₹{smoothed_price:.2f} ({change*100:.1f}% → {max_change*100:.1f}%)")
                else:
                    smoothed_price = prev_price * (1 - max_change)
                    print(f"  📈 {df.index[i].year}: Adjusted from ₹{current_price:.2f} to ₹{smoothed_price:.2f} ({change*100:.1f}% → {-max_change*100:.1f}%)")
                
                df_smoothed[commodity].iloc[i] = smoothed_price
            else:
                print(f"  ✅ {df.index[i].year}: No change needed (₹{current_price:.2f}, {change*100:.1f}%)")
    
    # Save smoothed dataset
    df_smoothed_export = df_smoothed.T
    df_smoothed_export.index.name = 'Commodities'
    df_smoothed_export.to_csv('DatasetSIH1647_smoothed.csv')
    
    print(f"\n✅ Smoothed dataset saved as 'DatasetSIH1647_smoothed.csv'")
    
    # Compare key statistics
    print(f"\n📊 COMPARISON SUMMARY:")
    print(f"{'Commodity':<25} {'Original Range':<20} {'Smoothed Range':<20} {'Improvement'}")
    print("-" * 80)
    
    for commodity in df.columns[:10]:  # Show first 10 commodities
        orig_range = df[commodity].max() - df[commodity].min()
        smooth_range = df_smoothed[commodity].max() - df_smoothed[commodity].min()
        improvement = ((orig_range - smooth_range) / orig_range) * 100
        
        print(f"{commodity:<25} ₹{orig_range:<18.2f} ₹{smooth_range:<18.2f} {improvement:>6.1f}%")
    
    return df_smoothed

# Alternative: Keep original realistic data but add explanation
def create_price_explanation():
    """
    Create explanations for price variations to help users understand the data
    """
    explanations = {
        2016: {
            'Gram Dal': "Drought in pulse-growing regions led to supply shortage",
            'general': "Below-normal monsoon affected crop yields"
        },
        2020: {
            'Potato': "COVID-19 lockdown disrupted supply chains",
            'Vanaspati (Packed)': "Pandemic supply chain disruptions",
            'general': "COVID-19 pandemic affected food supply chains"
        },
        2021: {
            'Mustard Oil (Packed)': "Post-COVID supply recovery + global commodity inflation",
            'Groundnut Oil (Packed)': "Global edible oil price surge",
            'general': "Post-pandemic economic recovery and global inflation"
        },
        2022: {
            'general': "Ukraine war affected global commodity prices"
        }
    }
    
    # Save explanations
    import json
    with open('price_explanations.json', 'w') as f:
        json.dump(explanations, f, indent=2)
    
    print("📝 Price explanations saved to 'price_explanations.json'")
    return explanations

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Keep original realistic data (recommended)")
    print("2. Create smoothed dataset")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        df_smoothed = smooth_extreme_price_changes()
        print("\n✅ Smoothed dataset created!")
    else:
        explanations = create_price_explanation()
        print("\n✅ Original data kept with explanations!")
        print("\nThe price variations in your dataset are actually realistic!")
        print("They reflect real market conditions during:")
        print("- 2016: Drought affecting pulses")
        print("- 2020-2021: COVID-19 pandemic")
        print("- 2022: Ukraine war effects")