import sys
import warnings
warnings.filterwarnings('ignore')

# Safe imports
try:
    import pandas as pd
    import numpy as np
    CORE_AVAILABLE = True
except ImportError:
    print("❌ Core packages not available")
    sys.exit(1)

from datetime import datetime

# Import our modules with error handling
try:
    from data_fetcher import BulletproofDataFetcher
    from model_trainer import BulletproofModelTrainer
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"❌ Module import error: {e}")
    MODULES_AVAILABLE = False

class BulletproofPredictionSystem:
    def __init__(self):
        if not MODULES_AVAILABLE:
            print("❌ Required modules not available")
            return

        self.data_fetcher = BulletproofDataFetcher()
        self.model_trainer = BulletproofModelTrainer()
        self.predictions = {}
        self.market_data = None
        self.is_initialized = False
        self.initialization_status = "Not started"

    def initialize_system(self, data_period_days=365):
        """Initialize with comprehensive error handling"""
        try:
            print("🚀 Initializing NIFTY 50 Prediction System...")
            self.initialization_status = "Starting..."

            # Step 1: Data fetching
            print("📊 Step 1/3: Fetching market data...")
            self.initialization_status = "Fetching data..."

            self.market_data = self.data_fetcher.fetch_data(data_period_days)

            if self.market_data is None:
                self.initialization_status = "Failed - No data"
                print("❌ Data fetching failed")
                return False

            print(f"✅ Data loaded: {len(self.market_data)} days")

            # Step 2: Training data preparation
            print("🔧 Step 2/3: Preparing training data...")
            self.initialization_status = "Preparing training data..."

            X, y, feature_names = self.data_fetcher.prepare_training_data()

            if X is None or len(X) < 20:
                self.initialization_status = "Failed - Insufficient training data"
                print("❌ Training data preparation failed")
                return False

            print(f"✅ Training data ready: {X.shape}")

            # Step 3: Model training
            print("🤖 Step 3/3: Training prediction models...")
            self.initialization_status = "Training models..."

            training_success = self.model_trainer.train_models(X, y)

            if not training_success:
                self.initialization_status = "Failed - Model training"
                print("❌ Model training failed")
                return False

            # Success!
            self.is_initialized = True
            self.initialization_status = "Ready"

            models = self.model_trainer.get_model_names()
            print(f"✅ System ready! Trained models: {', '.join(models)}")

            return True

        except Exception as e:
            self.initialization_status = f"Error: {str(e)[:30]}..."
            print(f"❌ System initialization error: {e}")
            return False

    def generate_predictions(self):
        """Generate comprehensive predictions"""
        if not self.is_initialized:
            print("❌ System not initialized")
            return None

        try:
            print("🔮 Generating 7-day predictions...")

            # Get latest features and market data
            latest_features = self.data_fetcher.get_latest_features()
            market_summary = self.data_fetcher.get_market_summary()

            if latest_features is None or market_summary is None:
                print("❌ Cannot get latest market data")
                return None

            current_price = market_summary['current_price']
            predictions = {}

            # Generate predictions for each model
            for model_name in self.model_trainer.get_model_names():
                print(f"🔮 Generating {model_name} prediction...")

                prediction_result = self.model_trainer.predict_with_range(latest_features, model_name)

                if prediction_result is not None:
                    predictions[model_name] = self._process_prediction(
                        prediction_result, current_price, model_name
                    )

                    base_pred = prediction_result['base_prediction']
                    print(f"✅ {model_name}: ₹{base_pred:.2f}")
                else:
                    print(f"❌ {model_name} prediction failed")

            if len(predictions) > 0:
                self.predictions = predictions
                print(f"✅ Generated predictions for {len(predictions)} models")
                return predictions
            else:
                print("❌ No predictions generated")
                return None

        except Exception as e:
            print(f"❌ Prediction generation error: {e}")
            return None

    def _process_prediction(self, prediction_result, current_price, model_name):
        """Process raw prediction into detailed analysis"""
        try:
            base_pred = prediction_result['base_prediction']
            high_pred = prediction_result['high_7d']
            low_pred = prediction_result['low_7d']

            # Calculate changes and percentages
            base_change = base_pred - current_price
            base_change_pct = (base_change / current_price) * 100

            upside_potential = high_pred - current_price
            upside_pct = (upside_potential / current_price) * 100

            downside_risk = current_price - low_pred
            downside_pct = (downside_risk / current_price) * 100

            # Analysis
            trend = "Bullish" if base_change > 0 else "Bearish"
            trend_strength = self._calculate_trend_strength(base_change, prediction_result['uncertainty'])
            volatility_assessment = self._assess_volatility(prediction_result['uncertainty'], current_price)

            return {
                # Basic info
                'current_price': current_price,
                'base_prediction': base_pred,
                'highest_7d': high_pred,
                'lowest_7d': low_pred,

                # Changes
                'base_change': base_change,
                'base_change_pct': base_change_pct,
                'upside_potential': upside_potential,
                'upside_pct': upside_pct,
                'downside_risk': downside_risk,
                'downside_pct': downside_pct,

                # Analysis
                'trend': trend,
                'trend_strength': trend_strength,
                'prediction_range': high_pred - low_pred,
                'uncertainty': prediction_result['uncertainty'],
                'model_metrics': prediction_result['model_performance'],
                'risk_reward_ratio': upside_potential / max(downside_risk, 1),
                'volatility_assessment': volatility_assessment
            }

        except Exception as e:
            print(f"❌ Prediction processing error: {e}")
            return None

    def _calculate_trend_strength(self, price_change, uncertainty):
        """Calculate trend strength"""
        try:
            if uncertainty == 0:
                return "Neutral"

            strength_ratio = abs(price_change) / uncertainty

            if strength_ratio > 1.5:
                return "Strong"
            elif strength_ratio > 1.0:
                return "Moderate"
            elif strength_ratio > 0.5:
                return "Weak"
            else:
                return "Very Weak"
        except:
            return "Unknown"

    def _assess_volatility(self, uncertainty, current_price):
        """Assess volatility level"""
        try:
            volatility_pct = (uncertainty / current_price) * 100

            if volatility_pct > 5:
                return "High"
            elif volatility_pct > 3:
                return "Moderate"
            elif volatility_pct > 1:
                return "Low"
            else:
                return "Very Low"
        except:
            return "Unknown"

    def get_prediction_summary(self):
        """Get formatted summary"""
        if not self.predictions:
            return "No predictions available"

        try:
            lines = []
            lines.append("=" * 60)
            lines.append("🎯 NIFTY 50 - 7 DAY PREDICTION SUMMARY")
            lines.append("=" * 60)
            lines.append("")

            current_price = list(self.predictions.values())[0]['current_price']
            lines.append(f"💰 Current Price: ₹{current_price:.2f}")
            lines.append(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            lines.append("")

            for model_name, pred_data in self.predictions.items():
                lines.append(f"🤖 {model_name.upper()}")
                lines.append(f"   Base: ₹{pred_data['base_prediction']:.2f} ({pred_data['base_change_pct']:+.2f}%)")
                lines.append(f"   📈 High (7d): ₹{pred_data['highest_7d']:.2f} (+{pred_data['upside_pct']:.2f}%)")
                lines.append(f"   📉 Low (7d): ₹{pred_data['lowest_7d']:.2f} ({-pred_data['downside_pct']:.2f}%)")
                lines.append(f"   🎯 Trend: {pred_data['trend']} ({pred_data['trend_strength']})")
                lines.append(f"   ⚡ Volatility: {pred_data['volatility_assessment']}")
                lines.append("")

            lines.append("=" * 60)

            return "\n".join(lines)

        except Exception as e:
            return f"Error creating summary: {e}"

    def get_market_overview(self):
        """Get market overview"""
        try:
            return self.data_fetcher.get_market_summary()
        except Exception as e:
            print(f"❌ Market overview error: {e}")
            return None

    def get_historical_data(self, days=60):
        """Get historical data"""
        try:
            return self.data_fetcher.get_historical_data(days)
        except Exception as e:
            print(f"❌ Historical data error: {e}")
            return None

    def get_model_performance(self):
        """Get model performance metrics"""
        try:
            return self.model_trainer.get_model_metrics()
        except Exception as e:
            print(f"❌ Model performance error: {e}")
            return {}

    def get_feature_importance(self):
        """Get feature importance"""
        try:
            return self.model_trainer.get_feature_importance()
        except Exception as e:
            print(f"❌ Feature importance error: {e}")
            return None

    def get_system_status(self):
        """Get system status"""
        try:
            return {
                'is_initialized': self.is_initialized,
                'status': self.initialization_status,
                'has_predictions': len(self.predictions) > 0,
                'models_available': self.model_trainer.get_model_names() if self.is_initialized else [],
                'data_points': len(self.market_data) if self.market_data is not None else 0,
                'modules_ok': MODULES_AVAILABLE
            }
        except Exception as e:
            return {'error': str(e)}

# Test function
def run_system_test():
    """Test the complete system"""
    print("🧪 TESTING BULLETPROOF PREDICTION SYSTEM")
    print("=" * 70)

    if not MODULES_AVAILABLE:
        print("❌ Required modules not available")
        return

    system = BulletproofPredictionSystem()

    print("\n🔄 Testing system initialization...")
    success = system.initialize_system(100)  # Smaller dataset for testing

    if success:
        print("\n🔮 Testing prediction generation...")
        predictions = system.generate_predictions()

        if predictions:
            print("\n📊 RESULTS:")
            print(system.get_prediction_summary())
            print("\n🎉 SYSTEM TEST PASSED!")
        else:
            print("\n❌ Prediction generation failed")
    else:
        print("\n❌ System initialization failed")

if __name__ == "__main__":
    run_system_test()
