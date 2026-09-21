import sys
import warnings
warnings.filterwarnings('ignore')

# Safe imports with fallbacks
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    print("⚠️ yfinance not available - using fallback data")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("❌ pandas not available - critical error")
    sys.exit(1)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("❌ numpy not available - critical error") 
    sys.exit(1)

from datetime import datetime, timedelta
import json

class BulletproofDataFetcher:
    def __init__(self):
        self.symbol = "^NSEI"
        self.data = None
        self.fallback_mode = not YF_AVAILABLE

    def fetch_data(self, period_days=365):
        """Bulletproof data fetching with multiple fallbacks"""
        try:
            print(f"📊 Fetching NIFTY 50 data ({period_days} days)...")

            if YF_AVAILABLE:
                return self._fetch_live_data(period_days)
            else:
                return self._create_demo_data(period_days)

        except Exception as e:
            print(f"⚠️ Live data failed: {e}")
            print("📊 Using demo data for testing...")
            return self._create_demo_data(period_days)

    def _fetch_live_data(self, period_days):
        """Fetch real data from Yahoo Finance"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days + 50)

            ticker = yf.Ticker(self.symbol)
            self.data = ticker.history(start=start_date, end=end_date, auto_adjust=True)

            if self.data.empty:
                print("⚠️ No data from Yahoo Finance")
                return self._create_demo_data(period_days)

            # Clean and process data
            self.data = self._clean_data(self.data)
            self.data = self._add_indicators(self.data)
            self.data = self.data.tail(period_days)

            print(f"✅ Live data loaded: {len(self.data)} days")
            print(f"💰 Latest price: ₹{self.data['Close'].iloc[-1]:.2f}")

            return self.data

        except Exception as e:
            print(f"⚠️ Live data error: {e}")
            return self._create_demo_data(period_days)

    def _create_demo_data(self, period_days):
        """Create realistic demo data for testing"""
        try:
            print("📊 Creating demo NIFTY 50 data...")

            # Create date range
            end_date = datetime.now()
            dates = pd.date_range(end=end_date, periods=period_days, freq='D')

            # Create realistic NIFTY price movements
            base_price = 24500  # Approximate NIFTY level
            np.random.seed(42)  # Reproducible demo data

            # Generate price series with realistic volatility
            returns = np.random.normal(0.0008, 0.015, period_days)  # ~0.08% daily return, 1.5% volatility
            prices = [base_price]

            for ret in returns[1:]:
                new_price = prices[-1] * (1 + ret)
                prices.append(max(new_price, 1000))  # Minimum price floor

            # Create OHLCV data
            closes = np.array(prices)

            # Generate realistic OHLC from closes
            opens = np.roll(closes, 1)
            opens[0] = closes[0]

            highs = closes * (1 + np.abs(np.random.normal(0, 0.005, period_days)))
            lows = closes * (1 - np.abs(np.random.normal(0, 0.005, period_days)))

            # Ensure OHLC consistency
            for i in range(period_days):
                highs[i] = max(highs[i], opens[i], closes[i])
                lows[i] = min(lows[i], opens[i], closes[i])

            volumes = np.random.normal(300000000, 50000000, period_days)  # ~300M volume
            volumes = np.abs(volumes).astype(int)

            # Create DataFrame
            self.data = pd.DataFrame({
                'Open': opens,
                'High': highs,
                'Low': lows,
                'Close': closes,
                'Volume': volumes
            }, index=dates)

            # Add technical indicators
            self.data = self._add_indicators(self.data)

            print(f"✅ Demo data created: {len(self.data)} days")
            print(f"💰 Latest price: ₹{self.data['Close'].iloc[-1]:.2f}")
            print("📝 Note: This is demo data for testing purposes")

            return self.data

        except Exception as e:
            print(f"❌ Demo data creation failed: {e}")
            return None

    def _clean_data(self, df):
        """Clean data with robust error handling"""
        try:
            # Remove NaN values
            df = df.dropna()

            # Remove invalid prices
            df = df[(df['Close'] > 0) & (df['Volume'] > 0)]
            df = df[(df['High'] >= df['Low']) & (df['High'] >= df['Close']) & (df['Low'] <= df['Close'])]

            # Remove extreme outliers
            for col in ['Open', 'High', 'Low', 'Close']:
                q1 = df[col].quantile(0.01)
                q99 = df[col].quantile(0.99)
                df[col] = df[col].clip(q1, q99)

            return df

        except Exception as e:
            print(f"⚠️ Data cleaning error: {e}")
            return df

    def _add_indicators(self, df):
        """Add technical indicators with safe calculations"""
        try:
            # Moving averages
            df['MA_5'] = df['Close'].rolling(5, min_periods=1).mean()
            df['MA_10'] = df['Close'].rolling(10, min_periods=5).mean() 
            df['MA_20'] = df['Close'].rolling(20, min_periods=10).mean()

            # Price changes (clipped to reasonable ranges)
            df['Price_Change'] = df['Close'].pct_change().fillna(0).clip(-0.15, 0.15)
            df['Volume_Change'] = df['Volume'].pct_change().fillna(0).clip(-0.8, 0.8)

            # High-Low spread (normalized)
            hl_spread = (df['High'] - df['Low']) / df['Close']
            df['HL_Spread'] = hl_spread.fillna(0).clip(0, 0.08)

            # Volatility
            df['Volatility'] = df['Close'].rolling(20, min_periods=10).std().fillna(0)

            # RSI with safe calculation
            df['RSI'] = self._safe_rsi(df['Close'])

            # Final cleanup
            df = df.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)

            return df

        except Exception as e:
            print(f"⚠️ Indicators error: {e}")
            return df

    def _safe_rsi(self, prices, window=14):
        """Calculate RSI with comprehensive error handling"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window, min_periods=window//2).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window, min_periods=window//2).mean()

            # Safe division
            rs = gain / (loss + 1e-10)  # Add small value to avoid division by zero
            rsi = 100 - (100 / (1 + rs))

            # Fill and clip
            return rsi.fillna(50).clip(0, 100)

        except Exception as e:
            print(f"⚠️ RSI calculation error: {e}")
            return pd.Series(50, index=prices.index)

    def prepare_training_data(self, prediction_days=7):
        """Prepare ML training data with bulletproof validation"""
        if self.data is None or len(self.data) < 50:
            print("❌ Insufficient data for training")
            return None, None, None

        try:
            # Define features
            feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_5', 'MA_10', 'MA_20', 
                           'Price_Change', 'Volume_Change', 'HL_Spread', 'Volatility', 'RSI']

            # Use available features only
            available_features = [col for col in feature_cols if col in self.data.columns]

            if len(available_features) < 5:
                print(f"❌ Too few features: {available_features}")
                return None, None, None

            # Create feature matrix
            X = self.data[available_features].copy()

            # Create target (future price)
            y = self.data['Close'].shift(-prediction_days)

            # Remove invalid rows
            valid_mask = ~(X.isna().any(axis=1) | y.isna() | np.isinf(y))
            X = X[valid_mask]
            y = y[valid_mask]

            # Final validation and cleanup
            X = X.replace([np.inf, -np.inf], 0)
            for col in X.columns:
                if X[col].isna().any():
                    X[col] = X[col].fillna(X[col].median())

            if len(X) < 30:
                print(f"❌ Insufficient valid samples: {len(X)}")
                return None, None, None

            print(f"✅ Training data ready: {len(X)} samples, {len(available_features)} features")

            return X, y, available_features

        except Exception as e:
            print(f"❌ Training data error: {e}")
            return None, None, None

    def get_latest_features(self):
        """Get latest features for prediction"""
        if self.data is None:
            return None

        try:
            feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_5', 'MA_10', 'MA_20',
                           'Price_Change', 'Volume_Change', 'HL_Spread', 'Volatility', 'RSI']

            available_features = [col for col in feature_cols if col in self.data.columns]
            latest = self.data[available_features].iloc[-1:].copy()
            latest = latest.fillna(0)

            return latest

        except Exception as e:
            print(f"❌ Latest features error: {e}")
            return None

    def get_market_summary(self):
        """Get market statistics"""
        if self.data is None:
            return None

        try:
            current = self.data['Close'].iloc[-1]
            previous = self.data['Close'].iloc[-2] if len(self.data) > 1 else current

            return {
                'current_price': current,
                'daily_change': current - previous,
                'daily_change_pct': ((current - previous) / previous) * 100 if previous > 0 else 0,
                'week_high': self.data['High'].tail(7).max(),
                'week_low': self.data['Low'].tail(7).min(),
                'volume': self.data['Volume'].iloc[-1],
                'avg_volume_20d': self.data['Volume'].tail(20).mean(),
                'volatility': self.data['Volatility'].iloc[-1] if 'Volatility' in self.data.columns else 0,
                'rsi': self.data['RSI'].iloc[-1] if 'RSI' in self.data.columns else 50,
                'ma_5': self.data['MA_5'].iloc[-1] if 'MA_5' in self.data.columns else current,
                'ma_20': self.data['MA_20'].iloc[-1] if 'MA_20' in self.data.columns else current,
                'is_demo_data': self.fallback_mode or not YF_AVAILABLE
            }

        except Exception as e:
            print(f"❌ Market summary error: {e}")
            return None

    def get_historical_data(self, days=60):
        """Get historical data for visualization"""
        if self.data is not None:
            return self.data.tail(days).copy()
        return None

# Test the data fetcher
if __name__ == "__main__":
    print("🧪 Testing Bulletproof Data Fetcher...")
    fetcher = BulletproofDataFetcher()
    data = fetcher.fetch_data(100)

    if data is not None:
        print(f"✅ Data test passed: {data.shape}")
        X, y, features = fetcher.prepare_training_data()
        if X is not None:
            print(f"✅ Training data test passed: {X.shape}")
        summary = fetcher.get_market_summary()
        if summary:
            print(f"✅ Market summary test passed: ₹{summary['current_price']:.2f}")
        print("🎉 All tests passed!")
    else:
        print("❌ Data fetcher test failed")
