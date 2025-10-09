import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "DatasetSIH1647.csv")
df = pd.read_csv(file_path)

df.set_index('Commodities', inplace=True)
df = df.T
df.index = pd.date_range(start='2014', periods=len(df), freq='YE')

df = df.ffill()

commodities = df.columns.tolist()

st.title("🌾 Commodity Price Forecasting with Algorithm Comparison")
st.markdown("### Compare ARIMA vs SARIMAX Models")

selected_commodity = st.selectbox("Choose a Commodity", commodities)

# Model selection
col1, col2 = st.columns(2)
with col1:
    use_arima = st.checkbox("📊 ARIMA Model", value=True)
with col2:
    use_sarimax = st.checkbox("📈 SARIMAX Model", value=True)

if st.button("🚀 Generate Forecasts"):
    data = df[selected_commodity]
    
    st.write(f"### {selected_commodity} Price Analysis & Forecasts")
    
    # Store predictions and metrics
    predictions = {}
    metrics = {}
    
    # ARIMA Model
    if use_arima:
        with st.expander("📊 ARIMA Model Results"):
            try:
                # Find best ARIMA order
                best_aic = np.inf
                best_order = (1, 1, 1)
                for p in range(3):
                    for d in range(2):
                        for q in range(3):
                            try:
                                arima_temp = ARIMA(data, order=(p, d, q))
                                arima_fit = arima_temp.fit()
                                if arima_fit.aic < best_aic:
                                    best_aic = arima_fit.aic
                                    best_order = (p, d, q)
                            except:
                                continue
                
                # Train ARIMA with best order
                arima_model = ARIMA(data, order=best_order)
                arima_fit = arima_model.fit()
                arima_forecast = arima_fit.get_forecast(steps=5)
                arima_pred = arima_forecast.predicted_mean.values
                
                predictions['ARIMA'] = arima_pred
                
                # Calculate metrics
                validation_size = 3
                actual_val = data.iloc[-validation_size:].values
                pred_val = arima_pred[:validation_size]
                
                mse = mean_squared_error(actual_val, pred_val)
                mae = mean_absolute_error(actual_val, pred_val)
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs((actual_val - pred_val) / actual_val)) * 100
                
                metrics['ARIMA'] = {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
                
                st.success(f"✅ ARIMA{best_order} trained successfully!")
                st.write(f"**AIC:** {best_aic:.2f}")
                st.write(f"**RMSE:** {rmse:.4f}")
                st.write(f"**MAPE:** {mape:.2f}%")
                
            except Exception as e:
                st.error(f"❌ ARIMA failed: {e}")
    
    # SARIMAX Model
    if use_sarimax:
        with st.expander("📈 SARIMAX Model Results"):
            try:
                sarimax_model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
                sarimax_fit = sarimax_model.fit(disp=False)
                sarimax_forecast = sarimax_fit.get_forecast(steps=5)
                sarimax_pred = sarimax_forecast.predicted_mean.values
                
                predictions['SARIMAX'] = sarimax_pred
                
                # Calculate metrics
                validation_size = 3
                actual_val = data.iloc[-validation_size:].values
                pred_val = sarimax_pred[:validation_size]
                
                mse = mean_squared_error(actual_val, pred_val)
                mae = mean_absolute_error(actual_val, pred_val)
                rmse = np.sqrt(mse)
                mape = np.mean(np.abs((actual_val - pred_val) / actual_val)) * 100
                
                metrics['SARIMAX'] = {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
                
                st.success("✅ SARIMAX trained successfully!")
                st.write(f"**AIC:** {sarimax_fit.aic:.2f}")
                st.write(f"**RMSE:** {rmse:.4f}")
                st.write(f"**MAPE:** {mape:.2f}%")
                
            except Exception as e:
                st.error(f"❌ SARIMAX failed: {e}")
    
    # Model Comparison and Results
    if predictions:
        # 🏆 Find Best Algorithm
        st.subheader("🏆 Algorithm Performance Comparison")
        
        if len(metrics) > 1:
            best_model = min(metrics.keys(), key=lambda x: metrics[x]['MAPE'])
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"🥇 **Best Algorithm: {best_model}**")
                best_mape = metrics[best_model]['MAPE']
                st.metric("Best MAPE", f"{best_mape:.2f}%")
            
            with col2:
                st.info("📊 **Model Rankings:**")
                sorted_models = sorted(metrics.items(), key=lambda x: x[1]['MAPE'])
                for i, (model, metric) in enumerate(sorted_models):
                    medal = "🥇" if i == 0 else "🥈"
                    st.write(f"{medal} {model}: {metric['MAPE']:.2f}% MAPE")
        
        # Comparison Table
        if metrics:
            st.subheader("📊 Detailed Metrics Comparison")
            metrics_df = pd.DataFrame(metrics).T
            st.dataframe(metrics_df.round(4))
        
        # Forecasts Table
        forecast_years = pd.date_range(start='2025', periods=5, freq='YE')
        forecast_df = pd.DataFrame(index=forecast_years)
        
        for model_name, pred in predictions.items():
            forecast_df[f'{model_name}_Forecast'] = pred
        
        st.subheader("🔮 Price Forecasts (2025-2029)")
        st.dataframe(forecast_df.style.format("₹{:.2f}"))
        
        # Visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot historical data
        ax.plot(data.index, data.values, label='Historical Prices', linewidth=2, marker='o')
        
        # Plot forecasts
        colors = ['orange', 'red', 'green', 'purple']
        for i, (model_name, pred) in enumerate(predictions.items()):
            ax.plot(forecast_years, pred, label=f'{model_name} Forecast', 
                   linewidth=2, linestyle='--', color=colors[i % len(colors)])
        
        ax.set_title(f'{selected_commodity} Price Forecasting Comparison')
        ax.set_xlabel('Year')
        ax.set_ylabel('Price (₹)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Algorithm Recommendations
        st.subheader("💡 Why This Algorithm is Best")
        if len(metrics) > 1:
            if best_model == 'ARIMA':
                st.info("📊 **ARIMA** performed best because:")
                st.write("• Simple and effective for this commodity's price pattern")
                st.write("• Lower prediction error (MAPE)")
                st.write("• Good for univariate time series forecasting")
            elif best_model == 'SARIMAX':
                st.info("📈 **SARIMAX** performed best because:")
                st.write("• Handles seasonal patterns effectively")
                st.write("• Better suited for this commodity's price behavior")
                st.write("• Combines trends and seasonal components")
    else:
        st.warning("⚠️ Please select at least one model to generate forecasts.")