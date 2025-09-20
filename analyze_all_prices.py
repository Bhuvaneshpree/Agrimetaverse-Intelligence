import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_all_commodity_prices():
    """
    Comprehensive analysis of all commodity prices with detailed visualizations
    """
    # Load the dataset
    df = pd.read_csv("DatasetSIH1647.csv")
    df.set_index('Commodities', inplace=True)
    df = df.T
    df.index = pd.date_range(start='2014', periods=len(df), freq='YE')
    
    print("🌾 COMPREHENSIVE COMMODITY PRICE ANALYSIS (2014-2024)")
    print("=" * 70)
    
    # 1. Summary Statistics for all commodities
    print("\n📊 PRICE SUMMARY FOR ALL COMMODITIES:")
    print("-" * 70)
    print(f"{'Commodity':<25} {'Min (₹)':<10} {'Max (₹)':<10} {'Avg (₹)':<10} {'Growth %':<12}")
    print("-" * 70)
    
    for commodity in df.columns:
        min_price = df[commodity].min()
        max_price = df[commodity].max()
        avg_price = df[commodity].mean()
        growth_rate = ((df[commodity].iloc[-1] / df[commodity].iloc[0]) ** (1/10) - 1) * 100
        
        print(f"{commodity:<25} {min_price:<10.2f} {max_price:<10.2f} {avg_price:<10.2f} {growth_rate:<12.1f}")
    
    # 2. Price categories
    print("\n🏷️ COMMODITY CATEGORIES BY PRICE RANGE:")
    print("-" * 50)
    
    categories = {
        'Basic Staples (₹15-50)': [],
        'Processed Foods (₹50-100)': [],
        'Premium Items (₹100-200)': [],
        'Specialty Products (₹200+)': []
    }
    
    for commodity in df.columns:
        avg_price = df[commodity].mean()
        if avg_price <= 50:
            categories['Basic Staples (₹15-50)'].append(f"{commodity} (₹{avg_price:.1f})")
        elif avg_price <= 100:
            categories['Processed Foods (₹50-100)'].append(f"{commodity} (₹{avg_price:.1f})")
        elif avg_price <= 200:
            categories['Premium Items (₹100-200)'].append(f"{commodity} (₹{avg_price:.1f})")
        else:
            categories['Specialty Products (₹200+)'].append(f"{commodity} (₹{avg_price:.1f})")
    
    for category, items in categories.items():
        print(f"\n{category}:")
        for item in items:
            print(f"  • {item}")
    
    # 3. Volatility Analysis
    print("\n📈 PRICE VOLATILITY ANALYSIS:")
    print("-" * 50)
    
    volatility_data = []
    for commodity in df.columns:
        std_dev = df[commodity].std()
        cv = (std_dev / df[commodity].mean()) * 100  # Coefficient of variation
        volatility_data.append((commodity, std_dev, cv))
    
    # Sort by volatility
    volatility_data.sort(key=lambda x: x[2], reverse=True)
    
    print(f"{'Commodity':<25} {'Std Dev':<12} {'Volatility %':<12}")
    print("-" * 50)
    for commodity, std_dev, cv in volatility_data:
        if cv > 20:
            print(f"{commodity:<25} {std_dev:<12.2f} {cv:<12.1f} 🔴 High")
        elif cv > 10:
            print(f"{commodity:<25} {std_dev:<12.2f} {cv:<12.1f} 🟡 Medium")
        else:
            print(f"{commodity:<25} {std_dev:<12.2f} {cv:<12.1f} 🟢 Low")
    
    # 4. Year-wise price changes
    print("\n📅 YEAR-WISE AVERAGE PRICE CHANGES:")
    print("-" * 40)
    
    yearly_changes = df.pct_change().mean() * 100
    for year, change in yearly_changes.items():
        if pd.notna(change):
            if change > 10:
                print(f"{year}: +{change:.1f}% 📈 High inflation")
            elif change > 5:
                print(f"{year}: +{change:.1f}% 📊 Moderate increase")
            elif change > -5:
                print(f"{year}: {change:+.1f}% ➡️ Stable")
            else:
                print(f"{year}: {change:.1f}% 📉 Price decline")
    
    # 5. Create comprehensive visualizations
    create_comprehensive_charts(df)
    
    return df

def create_comprehensive_charts(df):
    """
    Create detailed charts for all commodities
    """
    print("\n📊 Creating comprehensive charts...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. All commodities price trends
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot all commodities
    for commodity in df.columns:
        ax.plot(df.index, df[commodity], label=commodity, linewidth=1.5, alpha=0.8)
    
    ax.set_title('All Commodity Price Trends (2014-2024)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Price (₹)', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('all_commodities_trends.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Price heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Normalize prices for better visualization
    df_normalized = df.div(df.mean(axis=1), axis=0)
    
    sns.heatmap(df_normalized.T, annot=False, cmap='RdYlGn_r', center=1, 
                cbar_kws={'label': 'Price Relative to Average'}, ax=ax)
    ax.set_title('Commodity Price Heatmap (Relative to Average)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Commodity', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('commodity_price_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Category-wise comparison
    categories = {
        'Grains': ['Rice', 'Wheat', 'Atta (Wheat)'],
        'Pulses': ['Gram Dal', 'Tur/Arhar Dal', 'Urad DaI', 'Moong DaI', 'Masoor Dal'],
        'Oils': ['Groundnut Oil (Packed)', 'Mustard Oil (Packed)', 'Vanaspati (Packed)', 
                'Soya Oil (Packed)', 'Sunflower Oil (Packed)', 'Palm Oil (Packed)'],
        'Vegetables': ['Potato', 'Onion', 'Tomato'],
        'Others': ['Sugar', 'Gur', 'Milk', 'Tea Loose', 'Salt Pack (lodised)']
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (category, commodities) in enumerate(categories.items()):
        available_commodities = [c for c in commodities if c in df.columns]
        
        if available_commodities and i < len(axes):
            for commodity in available_commodities:
                axes[i].plot(df.index, df[commodity], label=commodity, linewidth=2, marker='o')
            
            axes[i].set_title(f'{category} Price Trends', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Year')
            axes[i].set_ylabel('Price (₹)')
            axes[i].legend(fontsize=8)
            axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplot
    if len(categories) < len(axes):
        axes[-1].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('category_wise_trends.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Price distribution box plot
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Select top 15 commodities by average price for readability
    top_commodities = df.mean().sort_values(ascending=False).head(15)
    df_subset = df[top_commodities.index]
    
    box_data = [df_subset[col].values for col in df_subset.columns]
    
    bp = ax.boxplot(box_data)
    ax.set_xticklabels([col.replace(' (Packed)', '') for col in df_subset.columns])
    ax.set_title('Price Distribution - Top 15 Commodities', fontsize=14, fontweight='bold')
    ax.set_ylabel('Price (₹)')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('price_distribution_boxplot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Charts saved:")
    print("  • all_commodities_trends.png")
    print("  • commodity_price_heatmap.png") 
    print("  • category_wise_trends.png")
    print("  • price_distribution_boxplot.png")

if __name__ == "__main__":
    df = analyze_all_commodity_prices()
    print(f"\n🎯 ANALYSIS COMPLETE!")
    print(f"Total commodities analyzed: {len(df.columns)}")
    print(f"Time period: {df.index[0].year} - {df.index[-1].year}")
    print(f"All charts saved for detailed review!")