import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def validate_commodity_prices():
    """
    Validate commodity prices for inconsistencies, outliers, and unrealistic values
    """
    # Load the dataset
    df = pd.read_csv("DatasetSIH1647.csv")
    df.set_index('Commodities', inplace=True)
    df = df.T
    df.index = pd.date_range(start='2014', periods=len(df), freq='YE')
    
    print("=== COMMODITY PRICE VALIDATION REPORT ===\n")
    
    issues_found = []
    
    # 1. Check for negative prices
    negative_prices = (df < 0).any()
    if negative_prices.any():
        print("❌ NEGATIVE PRICES FOUND:")
        for commodity in negative_prices[negative_prices].index:
            print(f"   - {commodity}")
        issues_found.append("Negative prices")
    else:
        print("✅ No negative prices found")
    
    # 2. Check for unrealistically low prices (< ₹5)
    very_low_prices = (df < 5).any()
    if very_low_prices.any():
        print("\n⚠️  VERY LOW PRICES (< ₹5) FOUND:")
        for commodity in very_low_prices[very_low_prices].index:
            min_price = df[commodity].min()
            print(f"   - {commodity}: ₹{min_price:.2f}")
        issues_found.append("Very low prices")
    else:
        print("\n✅ No unrealistically low prices found")
    
    # 3. Check for unrealistically high prices (potential data entry errors)
    print("\n📊 PRICE RANGE ANALYSIS:")
    for commodity in df.columns:
        min_price = df[commodity].min()
        max_price = df[commodity].max()
        price_range = max_price - min_price
        price_volatility = df[commodity].std()
        
        # Check for extreme volatility (more than 200% change)
        if (max_price / min_price) > 3:
            print(f"⚠️  {commodity}: High volatility - Min: ₹{min_price:.2f}, Max: ₹{max_price:.2f} ({(max_price/min_price-1)*100:.1f}% change)")
            issues_found.append(f"High volatility in {commodity}")
        else:
            print(f"✅ {commodity}: Min: ₹{min_price:.2f}, Max: ₹{max_price:.2f}")
    
    # 4. Check for sudden jumps (year-over-year changes > 50%)
    print("\n📈 YEAR-OVER-YEAR CHANGE ANALYSIS:")
    for commodity in df.columns:
        yoy_changes = df[commodity].pct_change() * 100
        extreme_changes = yoy_changes[abs(yoy_changes) > 50]
        
        if not extreme_changes.empty:
            print(f"⚠️  {commodity} - Extreme year-over-year changes:")
            for year, change in extreme_changes.items():
                print(f"     {year}: {change:.1f}%")
            issues_found.append(f"Extreme changes in {commodity}")
    
    # 5. Check for missing or zero values
    print("\n🔍 MISSING/ZERO VALUE CHECK:")
    for commodity in df.columns:
        zero_values = (df[commodity] == 0).sum()
        nan_values = df[commodity].isna().sum()
        
        if zero_values > 0 or nan_values > 0:
            print(f"⚠️  {commodity}: {zero_values} zero values, {nan_values} missing values")
            issues_found.append(f"Missing/zero values in {commodity}")
        else:
            print(f"✅ {commodity}: No missing or zero values")
    
    # 6. Market Reality Check
    print("\n🏪 MARKET REALITY CHECK:")
    
    # Check if basic commodities are reasonably priced
    market_checks = {
        'Rice': (20, 60, "Basic staple grain"),
        'Wheat': (15, 50, "Basic staple grain"), 
        'Potato': (5, 40, "Seasonal vegetable"),
        'Onion': (10, 50, "Seasonal vegetable"),
        'Tomato': (15, 60, "Seasonal vegetable"),
        'Milk': (30, 80, "Daily dairy product"),
        'Sugar': (25, 60, "Basic sweetener"),
        'Groundnut Oil (Packed)': (80, 250, "Cooking oil"),
        'Tea Loose': (150, 350, "Beverage")
    }
    
    for commodity, (min_expected, max_expected, category) in market_checks.items():
        if commodity in df.columns:
            actual_min = df[commodity].min()
            actual_max = df[commodity].max()
            
            if actual_min < min_expected or actual_max > max_expected:
                print(f"⚠️  {commodity} ({category}): Outside expected range ₹{min_expected}-{max_expected}")
                print(f"     Actual range: ₹{actual_min:.2f}-{actual_max:.2f}")
                issues_found.append(f"Unrealistic pricing for {commodity}")
            else:
                print(f"✅ {commodity}: Within expected range ₹{min_expected}-{max_expected}")
    
    # Summary
    print(f"\n=== VALIDATION SUMMARY ===")
    if issues_found:
        print(f"❌ {len(issues_found)} issues found:")
        for i, issue in enumerate(issues_found, 1):
            print(f"{i}. {issue}")
        
        print("\n🔧 RECOMMENDATIONS:")
        print("1. Review data sources for accuracy")
        print("2. Check for unit consistency (per kg, per liter, etc.)")
        print("3. Verify seasonal price variations are realistic")
        print("4. Consider regional price differences")
        print("5. Cross-check with market reports from that period")
    else:
        print("✅ All prices appear reasonable!")
    
    return df, issues_found

# Run validation
if __name__ == "__main__":
    df, issues = validate_commodity_prices()
    
    # Create visualization for price trends
    print("\n📊 Creating price trend visualization...")
    
    # Plot a few key commodities
    key_commodities = ['Rice', 'Wheat', 'Potato', 'Onion', 'Tomato', 'Milk']
    available_commodities = [c for c in key_commodities if c in df.columns]
    
    if available_commodities:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, commodity in enumerate(available_commodities[:6]):
            if i < len(axes):
                axes[i].plot(df.index, df[commodity], marker='o', linewidth=2)
                axes[i].set_title(f'{commodity} Price Trend')
                axes[i].set_ylabel('Price (₹)')
                axes[i].grid(True, alpha=0.3)
                axes[i].tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(available_commodities), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('price_trends_validation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Price trend chart saved as 'price_trends_validation.png'")