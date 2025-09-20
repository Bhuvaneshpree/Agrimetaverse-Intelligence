import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

# Install required packages first
try:
    from prophet import Prophet
except ImportError:
    st.error("Prophet not installed. Run: pip install prophet")
    st.stop()

try:
    import xgboost as xgb
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.preprocessing import StandardScaler
except ImportError:
    st.error("XGBoost and sklearn not installed. Run: pip install xgboost scikit-learn")
    st.stop()

class AdvancedCommodityForecaster:
    def __init__(self):
        self.models = {}
        self.predictions = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load commodity prices and external features"""
        try:
            # Load commodity prices
            df_prices = pd.read_csv("DatasetSIH1647.csv")
            df_prices.set_index('Commodities', inplace=True)
            df_prices = df_prices.T
            df_prices.index = pd.date_range(start='2014', periods=len(df_prices), freq='YE')
            df_prices = df_prices.ffill()
            
            # Load external features
            df_features = pd.read_csv("enhanced_features.csv")
            df_features['Year'] = pd.date_range(start='2014', periods=len(df_features), freq='YE')
            df_features.set_index('Year', inplace=True)
            
            return df_prices, df_features
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None, None
    
    def prepare_data_for_xgboost(self, commodity_data, features, lookback=3):
        """Prepare data for XGBoost with time series features"""
        X, y = [], []
        
        # Combine commodity prices with external features
        combined_data = pd.concat([commodity_data, features], axis=1)
        
        for i in range(lookback, len(combined_data)):
            # Create lagged features
            X_row = []
            
            # Add lagged commodity prices
            for lag in range(1, lookback + 1):
                X_row.append(commodity_data.iloc[i - lag])
            
            # Add current external features
            X_row.extend(features.iloc[i].values)
            
            X.append(X_row)
            y.append(commodity_data.iloc[i])
        
        return np.array(X), np.array(y)
    
    def train_sarimax(self, data):
        """Train SARIMAX model"""
        try:
            model = SARIMAX(data, order=(1, 1, 1))
            fitted_model = model.fit(disp=False)
            return fitted_model
        except Exception as e:
            st.warning(f"SARIMAX training failed: {e}")
            return None
    
    def train_prophet(self, data, features):
        """Train Prophet model with external regressors"""
        try:
            # Prepare data for Prophet
            prophet_data = pd.DataFrame({
                'ds': data.index,
                'y': data.values
            })
            
            # Add external regressors
            for col in features.columns:
                if col != 'Year':
                    prophet_data[col] = features[col].values
            
            # Initialize Prophet with external regressors
            model = Prophet(
                yearly_seasonality=True,
                daily_seasonality=False,
                weekly_seasonality=False
            )
            
            # Add external regressors
            for col in features.columns:
                if col != 'Year':
                    model.add_regressor(col)
            
            model.fit(prophet_data)
            return model, prophet_data
        except Exception as e:
            st.warning(f"Prophet training failed: {e}")
            return None, None
    
    def train_xgboost(self, data, features):
        """Train XGBoost model"""
        try:
            X, y = self.prepare_data_for_xgboost(data, features)
            
            if len(X) < 5:  # Need minimum data points
                return None
            
            # Split for training (use first 80% for training)
            split_idx = int(0.8 * len(X))
            X_train, y_train = X[:split_idx], y[:split_idx]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Train XGBoost
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)
            
            return model
        except Exception as e:
            st.warning(f"XGBoost training failed: {e}")
            return None
    
    def forecast_sarimax(self, model, steps=5):
        """Generate SARIMAX forecasts"""
        try:
            forecast = model.get_forecast(steps=steps)
            return forecast.predicted_mean.values
        except:
            return None
    
    def forecast_prophet(self, model, prophet_data, features, steps=5):
        """Generate Prophet forecasts"""
        try:
            # Create future dataframe
            future = model.make_future_dataframe(periods=steps, freq='YE')
            
            # Add future external regressors (simple trend extrapolation)
            for col in features.columns:
                if col != 'Year':
                    # Simple linear extrapolation for future values
                    last_values = features[col].tail(3).values
                    trend = (last_values[-1] - last_values[0]) / 2
                    
                    for i in range(len(future)):
                        if i < len(prophet_data):
                            future.loc[i, col] = prophet_data.iloc[i][col]
                        else:
                            future.loc[i, col] = last_values[-1] + trend * (i - len(prophet_data) + 1)
            
            forecast = model.predict(future)
            return forecast['yhat'].tail(steps).values
        except:
            return None
    
    def forecast_xgboost(self, model, data, features, steps=5):
        """Generate XGBoost forecasts"""
        try:
            predictions = []
            current_data = data.values.copy()
            
            for step in range(steps):
                # Prepare input for prediction
                X_input = []
                
                # Add lagged values
                for lag in range(1, 4):  # lookback=3
                    X_input.append(current_data[-(lag)])
                
                # Add extrapolated features
                last_year_features = features.iloc[-1].values
                X_input.extend(last_year_features)
                
                X_input = np.array(X_input).reshape(1, -1)
                X_input_scaled = self.scaler.transform(X_input)
                
                # Predict
                pred = model.predict(X_input_scaled)[0]
                predictions.append(pred)
                
                # Update current_data
                current_data = np.append(current_data, pred)
            
            return np.array(predictions)
        except:
            return None
    
    def ensemble_forecast(self, predictions_dict, weights=None):
        """Create ensemble forecast from multiple models"""
        valid_predictions = {k: v for k, v in predictions_dict.items() if v is not None}
        
        if not valid_predictions:
            return None
        
        if weights is None:
            # Equal weights
            weights = {k: 1/len(valid_predictions) for k in valid_predictions.keys()}
        
        # Weighted average
        ensemble = np.zeros(5)  # 5 years forecast
        
        for model_name, preds in valid_predictions.items():
            if len(preds) == 5:
                ensemble += weights.get(model_name, 0) * preds
        
        return ensemble
    
    def calculate_metrics(self, actual, predicted):
        """Calculate evaluation metrics"""
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape
        }

def main():
    st.set_page_config(page_title="Advanced Commodity Forecasting", layout="wide")
    st.title("🌾 Advanced Agricultural Commodity Price Forecasting")
    st.markdown("### AI-Powered Multi-Model Ensemble System")
    
    forecaster = AdvancedCommodityForecaster()
    
    # Load data
    df_prices, df_features = forecaster.load_data()
    
    if df_prices is None or df_features is None:
        st.error("Failed to load data. Please ensure CSV files are present.")
        return
    
    # Sidebar for model selection
    st.sidebar.header("Model Configuration")
    
    commodities = df_prices.columns.tolist()
    selected_commodity = st.sidebar.selectbox("Choose Commodity", commodities)
    
    # Model selection
    use_sarimax = st.sidebar.checkbox("SARIMAX Model", value=True)
    use_prophet = st.sidebar.checkbox("Prophet Model", value=True)
    use_xgboost = st.sidebar.checkbox("XGBoost Model", value=True)
    use_ensemble = st.sidebar.checkbox("Ensemble Forecast", value=True)
    
    if st.sidebar.button("🚀 Generate Forecasts"):
        
        data = df_prices[selected_commodity]
        
        # Display current data
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader(f"Historical {selected_commodity} Prices")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(data.index, data.values, marker='o', linewidth=2)
            ax.set_title(f"{selected_commodity} Price Trend (2014-2024)")
            ax.set_xlabel("Year")
            ax.set_ylabel("Price")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        with col2:
            st.subheader("Key Statistics")
            st.metric("Average Price", f"₹{data.mean():.2f}")
            st.metric("Price Volatility", f"{data.std():.2f}")
            st.metric("Growth Rate", f"{((data.iloc[-1]/data.iloc[0])**(1/10) - 1)*100:.1f}%")
        
        # Train models and generate forecasts
        predictions = {}
        
        with st.spinner("Training models and generating forecasts..."):
            
            if use_sarimax:
                with st.expander("📊 SARIMAX Model"):
                    sarimax_model = forecaster.train_sarimax(data)
                    if sarimax_model:
                        sarimax_pred = forecaster.forecast_sarimax(sarimax_model)
                        predictions['SARIMAX'] = sarimax_pred
                        st.success("✅ SARIMAX model trained successfully")
                    else:
                        st.error("❌ SARIMAX model failed")
            
            if use_prophet:
                with st.expander("🔮 Prophet Model"):
                    prophet_model, prophet_data = forecaster.train_prophet(data, df_features)
                    if prophet_model:
                        prophet_pred = forecaster.forecast_prophet(prophet_model, prophet_data, df_features)
                        predictions['Prophet'] = prophet_pred
                        st.success("✅ Prophet model trained successfully")
                    else:
                        st.error("❌ Prophet model failed")
            
            if use_xgboost:
                with st.expander("🚀 XGBoost Model"):
                    xgb_model = forecaster.train_xgboost(data, df_features)
                    if xgb_model:
                        xgb_pred = forecaster.forecast_xgboost(xgb_model, data, df_features)
                        predictions['XGBoost'] = xgb_pred
                        st.success("✅ XGBoost model trained successfully")
                    else:
                        st.error("❌ XGBoost model failed")
        
        # Generate ensemble forecast
        if use_ensemble and len(predictions) > 1:
            ensemble_pred = forecaster.ensemble_forecast(predictions)
            if ensemble_pred is not None:
                predictions['Ensemble'] = ensemble_pred
        
        # Display forecasts
        if predictions:
            st.subheader("🔮 Price Forecasts (2025-2029)")
            
            # Create forecast DataFrame
            forecast_years = pd.date_range(start='2025', periods=5, freq='YE')
            forecast_df = pd.DataFrame(index=forecast_years)
            
            for model_name, preds in predictions.items():
                if preds is not None and len(preds) == 5:
                    forecast_df[f'{model_name}_Forecast'] = preds
            
            st.dataframe(forecast_df)
            
            # Visualization
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Plot historical data
            ax.plot(data.index, data.values, 'o-', label='Historical Prices', linewidth=2, color='black')
            
            # Plot forecasts
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            for i, (model_name, preds) in enumerate(predictions.items()):
                if preds is not None and len(preds) == 5:
                    ax.plot(forecast_years, preds, 'o--', label=f'{model_name}', 
                           linewidth=2, color=colors[i % len(colors)])
            
            ax.set_title(f'{selected_commodity} Price Forecasting - Multiple Models Comparison')
            ax.set_xlabel('Year')
            ax.set_ylabel('Price (₹)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Feature importance (if XGBoost was used)
            if 'XGBoost' in predictions and xgb_model:
                st.subheader("📈 Feature Importance Analysis")
                
                feature_names = ['Price_lag1', 'Price_lag2', 'Price_lag3'] + [col for col in df_features.columns if col != 'Year']
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': xgb_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=importance_df.head(10), x='Importance', y='Feature', ax=ax)
                ax.set_title('Top 10 Most Important Features for Price Prediction')
                st.pyplot(fig)
        
        else:
            st.error("No valid predictions generated. Please check your data and model settings.")

if __name__ == "__main__":
    main()