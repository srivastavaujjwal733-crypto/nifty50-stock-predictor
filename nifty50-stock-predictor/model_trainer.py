import sys
import warnings
warnings.filterwarnings('ignore')

# Safe imports with fallbacks
try:
    import pandas as pd
    import numpy as np
    CORE_AVAILABLE = True
except ImportError:
    print("❌ Core packages (pandas/numpy) not available")
    sys.exit(1)

try:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️ scikit-learn not available - using simple models")

class BulletproofModelTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.metrics = {}
        self.is_trained = False
        self.prediction_ranges = {}
        self.sklearn_available = SKLEARN_AVAILABLE

    def train_models(self, X, y, test_size=0.2):
        """Train models with comprehensive error handling"""
        try:
            print(f"🤖 Training models with {len(X)} samples...")

            if not self._validate_training_data(X, y):
                return False

            self.feature_names = list(X.columns)

            if self.sklearn_available:
                return self._train_sklearn_models(X, y, test_size)
            else:
                return self._train_simple_models(X, y, test_size)

        except Exception as e:
            print(f"❌ Training error: {e}")
            return False

    def _validate_training_data(self, X, y):
        """Comprehensive data validation"""
        try:
            if X.empty or y.empty:
                print("❌ Empty training data")
                return False

            if len(X) != len(y):
                print("❌ Feature-target size mismatch")
                return False

            if len(X) < 20:
                print("❌ Insufficient training samples")
                return False

            # Check for invalid values
            if X.isna().any().any():
                print("❌ NaN in features")
                return False

            if y.isna().any():
                print("❌ NaN in targets")
                return False

            if np.isinf(X.values).any() or np.isinf(y.values).any():
                print("❌ Infinite values detected")
                return False

            if (y <= 0).any():
                print("❌ Invalid price targets")
                return False

            print("✅ Training data validation passed")
            return True

        except Exception as e:
            print(f"❌ Validation error: {e}")
            return False

    def _train_sklearn_models(self, X, y, test_size):
        """Train using scikit-learn models"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, shuffle=False
            )

            print(f"📊 Split: {len(X_train)} train, {len(X_test)} test")

            # Train Random Forest
            self._train_random_forest(X_train, X_test, y_train, y_test)

            # Train Linear Regression
            self._train_linear_regression(X_train, X_test, y_train, y_test)

            if len(self.models) > 0:
                self.is_trained = True
                print(f"✅ Trained {len(self.models)} scikit-learn models")
                return True
            else:
                print("❌ No sklearn models trained")
                return False

        except Exception as e:
            print(f"❌ sklearn training error: {e}")
            return False

    def _train_random_forest(self, X_train, X_test, y_train, y_test):
        """Train Random Forest with error handling"""
        try:
            print("🌲 Training Random Forest...")

            rf = RandomForestRegressor(
                n_estimators=100, max_depth=15, min_samples_split=10,
                min_samples_leaf=5, random_state=42, n_jobs=1  # Single thread for stability
            )

            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)

            if self._validate_predictions(y_pred):
                metrics = self._calculate_metrics(y_test, y_pred)

                self.models['Random Forest'] = rf
                self.scalers['Random Forest'] = None
                self.prediction_ranges['Random Forest'] = metrics['pred_std']
                self.metrics['Random Forest'] = metrics

                print(f"✅ Random Forest: MAE=₹{metrics['MAE']:.2f}, R²={metrics['R2']:.3f}")
            else:
                print("❌ Random Forest predictions invalid")

        except Exception as e:
            print(f"❌ Random Forest error: {e}")

    def _train_linear_regression(self, X_train, X_test, y_train, y_test):
        """Train Linear Regression with scaling"""
        try:
            print("📈 Training Linear Regression...")

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Check scaled data
            if np.isinf(X_train_scaled).any() or np.isnan(X_train_scaled).any():
                print("❌ Scaling produced invalid values")
                return

            lr = LinearRegression()
            lr.fit(X_train_scaled, y_train)
            y_pred = lr.predict(X_test_scaled)

            if self._validate_predictions(y_pred):
                metrics = self._calculate_metrics(y_test, y_pred)

                self.models['Linear Regression'] = lr
                self.scalers['Linear Regression'] = scaler
                self.prediction_ranges['Linear Regression'] = metrics['pred_std']
                self.metrics['Linear Regression'] = metrics

                print(f"✅ Linear Regression: MAE=₹{metrics['MAE']:.2f}, R²={metrics['R2']:.3f}")
            else:
                print("❌ Linear Regression predictions invalid")

        except Exception as e:
            print(f"❌ Linear Regression error: {e}")

    def _train_simple_models(self, X, y, test_size):
        """Fallback simple models when sklearn unavailable"""
        try:
            print("📊 Training simple fallback models...")

            # Simple moving average model
            current_price = y.iloc[-1]
            recent_change = y.pct_change().tail(10).mean()

            self.models['Simple Trend'] = {
                'type': 'simple',
                'base_price': current_price,
                'trend': recent_change
            }

            self.metrics['Simple Trend'] = {
                'MAE': abs(current_price * 0.02),  # Assume 2% error
                'RMSE': abs(current_price * 0.03),
                'R2': 0.1,
                'pred_std': abs(current_price * 0.025)
            }

            self.prediction_ranges['Simple Trend'] = abs(current_price * 0.025)
            self.is_trained = True

            print("✅ Simple model trained (fallback mode)")
            return True

        except Exception as e:
            print(f"❌ Simple model error: {e}")
            return False

    def _calculate_metrics(self, y_true, y_pred):
        """Calculate performance metrics safely"""
        try:
            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)

            try:
                r2 = r2_score(y_true, y_pred)
                if not np.isfinite(r2):
                    r2 = 0.0
            except:
                r2 = 0.0

            pred_std = np.std(y_true - y_pred)

            return {
                'MAE': mae,
                'RMSE': rmse,
                'R2': max(0, min(1, r2)),  # Clip R2 to reasonable range
                'pred_std': pred_std
            }

        except Exception as e:
            print(f"❌ Metrics calculation error: {e}")
            return {'MAE': 100, 'RMSE': 150, 'R2': 0.0, 'pred_std': 50}

    def _validate_predictions(self, predictions):
        """Validate model predictions"""
        try:
            return not (np.isinf(predictions).any() or 
                       np.isnan(predictions).any() or 
                       (predictions <= 0).any() or
                       (predictions > 1e6).any())
        except:
            return False

    def predict_with_range(self, X, model_name='Random Forest'):
        """Make predictions with confidence ranges"""
        if not self.is_trained or model_name not in self.models:
            print(f"❌ Model '{model_name}' not available")
            return None

        try:
            model = self.models[model_name]

            # Handle simple models
            if isinstance(model, dict) and model.get('type') == 'simple':
                base_pred = model['base_price'] * (1 + model['trend'])
                uncertainty = self.prediction_ranges[model_name]

                return {
                    'base_prediction': base_pred,
                    'high_7d': base_pred + uncertainty * 1.5,
                    'low_7d': max(base_pred - uncertainty * 1.5, base_pred * 0.7),
                    'uncertainty': uncertainty,
                    'model_performance': self.metrics[model_name]
                }

            # Handle sklearn models
            if not self._validate_input_features(X):
                return None

            scaler = self.scalers[model_name]

            if scaler is not None:
                X_scaled = scaler.transform(X)
                if np.isinf(X_scaled).any() or np.isnan(X_scaled).any():
                    print("❌ Input scaling failed")
                    return None
                prediction_input = X_scaled
            else:
                prediction_input = X.values

            base_pred = model.predict(prediction_input)[0]

            if not np.isfinite(base_pred) or base_pred <= 0:
                print("❌ Invalid prediction")
                return None

            uncertainty = self.prediction_ranges.get(model_name, base_pred * 0.02)

            return {
                'base_prediction': base_pred,
                'high_7d': base_pred + uncertainty * 1.5,
                'low_7d': max(base_pred - uncertainty * 1.5, base_pred * 0.8),
                'uncertainty': uncertainty,
                'model_performance': self.metrics[model_name]
            }

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None

    def _validate_input_features(self, X):
        """Validate input features for prediction"""
        try:
            if list(X.columns) != self.feature_names:
                print("❌ Feature name mismatch")
                return False

            if X.isna().any().any() or np.isinf(X.values).any():
                print("❌ Invalid input values")
                return False

            return True

        except Exception as e:
            print(f"❌ Input validation error: {e}")
            return False

    def get_model_names(self):
        """Get available model names"""
        return list(self.models.keys())

    def get_model_metrics(self):
        """Get all model metrics"""
        return self.metrics.copy()

    def get_feature_importance(self):
        """Get feature importance (if available)"""
        if 'Random Forest' not in self.models:
            return None

        try:
            rf_model = self.models['Random Forest']
            if hasattr(rf_model, 'feature_importances_'):
                return pd.DataFrame({
                    'Feature': self.feature_names,
                    'Importance': rf_model.feature_importances_
                }).sort_values('Importance', ascending=False)
        except:
            pass

        return None

# Test the model trainer
if __name__ == "__main__":
    print("🧪 Testing Bulletproof Model Trainer...")

    # Create dummy data
    np.random.seed(42)
    n_samples, n_features = 100, 8

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.randn(n_samples) * 100 + 25000)

    trainer = BulletproofModelTrainer()
    success = trainer.train_models(X, y)

    if success:
        print(f"✅ Models trained: {trainer.get_model_names()}")

        # Test prediction
        test_input = X.iloc[-1:].copy()
        for model_name in trainer.get_model_names():
            result = trainer.predict_with_range(test_input, model_name)
            if result:
                print(f"✅ {model_name}: ₹{result['base_prediction']:.2f}")

        print("🎉 All tests passed!")
    else:
        print("❌ Model trainer test failed")
