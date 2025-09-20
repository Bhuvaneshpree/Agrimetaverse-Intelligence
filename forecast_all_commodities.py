import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

def generate_all_commodity_forecasts():
    """
    Generate 5-year forecasts (2025-2029) for all 22 commodities using SARIMAX model
    """
    print("🌾 GENERATING 5-YEAR FORECASTS FOR ALL 22 COMMODITIES")
    print("=" * 65)
    print("📅 Forecast Period: 2025-2029")
    print("🤖 AI Model: SARIMAX (1,1,1)")
    print("=" * 65)
    
    # Load the dataset
    df = pd.read_csv("DatasetSIH1647.csv")
    df.set_index('Commodities', inplace=True)
    df = df.T
    df.index = pd.date_range(start='2014', periods=len(df), freq='YE')
    df = df.ffill()
    
    # Forecast years
    forecast_years = pd.date_range(start='2025', periods=5, freq='YE')
    
    # Store all forecasts
    all_forecasts = {}
    forecast_summary = []
    
    print("\n🔮 GENERATING FORECASTS...")
    print("-" * 65)
    
    for i, commodity in enumerate(df.columns, 1):
        print(f"[{i:2d}/22] Processing {commodity}...")
        
        try:
            # Get commodity data
            data = df[commodity]
            
            # Train SARIMAX model
            model = SARIMAX(data, order=(1, 1, 1))
            fitted_model = model.fit(disp=False)
            
            # Generate forecast
            forecast = fitted_model.get_forecast(steps=5)
            forecasted_values = forecast.predicted_mean.values
            
            # Store forecast
            all_forecasts[commodity] = forecasted_values
            
            # Calculate growth metrics
            current_price = data.iloc[-1]  # 2024 price
            forecast_2029 = forecasted_values[-1]  # 2029 price
            total_growth = ((forecast_2029 / current_price) ** (1/5) - 1) * 100
            
            forecast_summary.append({
                'Commodity': commodity,
                '2024_Price': current_price,
                '2025_Forecast': forecasted_values[0],
                '2026_Forecast': forecasted_values[1],
                '2027_Forecast': forecasted_values[2],
                '2028_Forecast': forecasted_values[3],
                '2029_Forecast': forecasted_values[4],
                'Annual_Growth_%': total_growth,
                'Total_Growth_%': ((forecast_2029 / current_price) - 1) * 100
            })
            
        except Exception as e:
            print(f"    ❌ Error forecasting {commodity}: {str(e)}")
            continue
    
    print(f"\n✅ Successfully generated forecasts for {len(all_forecasts)} commodities!")
    
    # Create comprehensive forecast DataFrame
    forecast_df = pd.DataFrame(forecast_summary)
    
    # Display results
    display_forecast_results(forecast_df)
    
    # Create detailed forecast table
    create_detailed_forecast_table(all_forecasts, forecast_years)
    
    # Save results
    forecast_df.to_csv('all_commodities_forecast_2025-2029.csv', index=False)
    print(f"\n💾 Forecast results saved to 'all_commodities_forecast_2025-2029.csv'")
    
    return forecast_df, all_forecasts

def display_forecast_results(forecast_df):
    """
    Display formatted forecast results
    """
    print("\n📊 COMPREHENSIVE FORECAST SUMMARY (2025-2029)")
    print("=" * 85)
    
    # Sort by commodity type for better organization
    grains = ['Rice', 'Wheat', 'Atta (Wheat)']
    pulses = ['Gram Dal', 'Tur/Arhar Dal', 'Urad DaI', 'Moong DaI', 'Masoor Dal']
    oils = ['Groundnut Oil (Packed)', 'Mustard Oil (Packed)', 'Vanaspati (Packed)', 
            'Soya Oil (Packed)', 'Sunflower Oil (Packed)', 'Palm Oil (Packed)']
    vegetables = ['Potato', 'Onion', 'Tomato']
    others = ['Sugar', 'Gur', 'Milk', 'Tea Loose', 'Salt Pack (lodised)']
    
    categories = [
        ("🌾 GRAINS & CEREALS", grains),
        ("🫘 PULSES & DALS", pulses),
        ("🛢️ COOKING OILS", oils),
        ("🥬 VEGETABLES", vegetables),
        ("🍯 OTHER COMMODITIES", others)
    ]
    
    for category_name, commodities in categories:
        print(f"\n{category_name}")
        print("-" * 85)
        print(f"{'Commodity':<25} {'2024':<8} {'2025':<8} {'2026':<8} {'2027':<8} {'2028':<8} {'2029':<8} {'Growth%':<8}")
        print("-" * 85)
        
        category_data = forecast_df[forecast_df['Commodity'].isin(commodities)]
        
        for _, row in category_data.iterrows():
            growth_indicator = "📈" if row['Annual_Growth_%'] > 5 else "📊" if row['Annual_Growth_%'] > 2 else "➡️"
            print(f"{row['Commodity']:<25} "
                  f"₹{row['2024_Price']:<7.1f} "
                  f"₹{row['2025_Forecast']:<7.1f} "
                  f"₹{row['2026_Forecast']:<7.1f} "
                  f"₹{row['2027_Forecast']:<7.1f} "
                  f"₹{row['2028_Forecast']:<7.1f} "
                  f"₹{row['2029_Forecast']:<7.1f} "
                  f"{growth_indicator}{row['Annual_Growth_%']:<6.1f}%")

def create_detailed_forecast_table(all_forecasts, forecast_years):
    """
    Create a detailed forecast table
    """
    print("\n📋 DETAILED FORECAST TABLE")
    print("=" * 65)
    
    # Create DataFrame with forecasts
    forecast_table = pd.DataFrame(all_forecasts, index=forecast_years)
    
    # Display top performers
    print("\n🏆 TOP GROWTH COMMODITIES (2025-2029):")
    print("-" * 45)
    
    growth_rates = {}
    for commodity in all_forecasts.keys():
        if len(all_forecasts[commodity]) == 5:
            # Load current price for comparison
            df = pd.read_csv("DatasetSIH1647.csv")
            df.set_index('Commodities', inplace=True)
            current_price = df.loc[commodity, '2024']
            forecast_2029 = all_forecasts[commodity][-1]
            growth_rate = ((forecast_2029 / current_price) ** (1/5) - 1) * 100
            growth_rates[commodity] = growth_rate
    
    # Sort by growth rate
    sorted_growth = sorted(growth_rates.items(), key=lambda x: x[1], reverse=True)
    
    for i, (commodity, growth) in enumerate(sorted_growth[:10], 1):
        indicator = "🚀" if growth > 8 else "📈" if growth > 5 else "📊"
        print(f"{i:2d}. {commodity:<30} {indicator} {growth:>6.1f}% annually")
    
    # Display lowest growth
    print("\n📉 STABLE PRICE COMMODITIES (Lowest Growth):")
    print("-" * 45)
    
    for i, (commodity, growth) in enumerate(sorted_growth[-5:], 1):
        print(f"{i:2d}. {commodity:<30} ➡️ {growth:>6.1f}% annually")

def analyze_forecast_insights(forecast_df):
    """
    Provide insights from the forecasts
    """
    print("\n💡 KEY FORECAST INSIGHTS")
    print("=" * 45)
    
    # Average growth by category
    high_growth = forecast_df[forecast_df['Annual_Growth_%'] > 6]
    moderate_growth = forecast_df[(forecast_df['Annual_Growth_%'] >= 3) & (forecast_df['Annual_Growth_%'] <= 6)]
    low_growth = forecast_df[forecast_df['Annual_Growth_%'] < 3]
    
    print(f"\n📈 HIGH GROWTH (>6% annually): {len(high_growth)} commodities")
    for _, row in high_growth.iterrows():
        print(f"   • {row['Commodity']}: {row['Annual_Growth_%']:.1f}%")
    
    print(f"\n📊 MODERATE GROWTH (3-6% annually): {len(moderate_growth)} commodities")
    for _, row in moderate_growth.iterrows():
        print(f"   • {row['Commodity']}: {row['Annual_Growth_%']:.1f}%")
    
    print(f"\n➡️ STABLE PRICES (<3% annually): {len(low_growth)} commodities")
    for _, row in low_growth.iterrows():
        print(f"   • {row['Commodity']}: {row['Annual_Growth_%']:.1f}%")
    
    # Price range predictions for 2029
    print(f"\n🎯 PRICE RANGE PREDICTIONS FOR 2029:")
    print("-" * 40)
    
    affordable = forecast_df[forecast_df['2029_Forecast'] < 50]
    moderate = forecast_df[(forecast_df['2029_Forecast'] >= 50) & (forecast_df['2029_Forecast'] < 100)]
    premium = forecast_df[forecast_df['2029_Forecast'] >= 100]
    
    print(f"💚 AFFORDABLE (<₹50): {len(affordable)} commodities")
    print(f"💛 MODERATE (₹50-100): {len(moderate)} commodities")  
    print(f"💰 PREMIUM (>₹100): {len(premium)} commodities")

if __name__ == "__main__":
    try:
        forecast_df, all_forecasts = generate_all_commodity_forecasts()
        analyze_forecast_insights(forecast_df)
        
        print("\n" + "="*65)
        print("🎯 FORECAST GENERATION COMPLETE!")
        print("📊 All 22 commodities forecasted for 2025-2029")
        print("💾 Results saved to CSV file")
        print("🤖 Powered by Advanced AI (SARIMAX)")
        print("="*65)
        
    except Exception as e:
        print(f"❌ Error generating forecasts: {e}")
        print("Please ensure DatasetSIH1647.csv is in the current directory.")