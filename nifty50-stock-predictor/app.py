import sys
import warnings
warnings.filterwarnings('ignore')

# Safe imports with user-friendly error messages
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    print("❌ Streamlit not available. Please install: pip install streamlit")
    sys.exit(1)

try:
    import pandas as pd
    import numpy as np
    CORE_AVAILABLE = True
except ImportError:
    st.error("❌ Core packages (pandas/numpy) not available. Please install requirements.")
    st.stop()

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("⚠️ Plotly not available - charts will be limited")

from datetime import datetime
import time
import traceback

# Import prediction system
try:
    from predictor import BulletproofPredictionSystem
    SYSTEM_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Prediction System Import Error: {e}")
    st.error("Please ensure all files are in the same directory")
    SYSTEM_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="NIFTY 50 Predictor - Fixed",
    page_icon="📈",
    layout="wide"
)

# Session state initialization
def init_session_state():
    """Initialize session state with defaults"""
    defaults = {
        'prediction_system': None,
        'system_ready': False,
        'predictions_ready': False,
        'current_predictions': None,
        'system_log': [],
        'demo_mode': False
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def add_log(message):
    """Add timestamped message to system log"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.system_log.append(f"[{timestamp}] {message}")
    if len(st.session_state.system_log) > 20:
        st.session_state.system_log = st.session_state.system_log[-20:]

def main():
    """Main application function"""
    if not SYSTEM_AVAILABLE:
        show_error_screen()
        return

    init_session_state()

    # Header
    st.title("🏛️ NIFTY 50 Stock Price Predictor")
    st.markdown("**Fixed Version - Python 3.12 Compatible**")

    # Layout
    create_sidebar()

    if not st.session_state.system_ready:
        show_welcome_screen()
    else:
        show_main_dashboard()

def show_error_screen():
    """Show error screen when system unavailable"""
    st.error("🚨 System Error - Required Components Missing")

    st.markdown("""
    **To fix this issue:**

    1. **Run the emergency launcher:**
       - Double-click `EMERGENCY_LAUNCHER.py`
       - OR run: `python EMERGENCY_LAUNCHER.py`

    2. **Manual installation:**
       ```
       pip install --upgrade pip setuptools wheel
       pip install --only-binary=all streamlit pandas numpy yfinance scikit-learn plotly
       ```

    3. **Check file integrity:**
       - Ensure all Python files are in the same folder
       - Re-extract the ZIP if files are missing

    4. **System compatibility:**
       - Python 3.8+ required
       - Windows 10+ recommended
       - Internet connection for live data
    """)

def create_sidebar():
    """Create sidebar with controls"""
    with st.sidebar:
        st.header("🎯 Control Panel")

        # System initialization
        if st.button("🚀 Initialize System", type="primary", use_container_width=True):
            initialize_system()

        # Prediction generation
        if st.session_state.system_ready:
            if st.button("🔮 Generate Predictions", type="secondary", use_container_width=True):
                generate_predictions()

        # Utilities
        if st.button("🗑️ Clear Log", use_container_width=True):
            st.session_state.system_log = []
            st.rerun()

        # Status display
        st.markdown("---")
        st.markdown("### 📊 System Status")

        if st.session_state.system_ready:
            st.success("✅ System Ready")
            if st.session_state.prediction_system:
                status = st.session_state.prediction_system.get_system_status()
                st.metric("Data Points", status.get('data_points', 0))
                st.metric("Models Available", len(status.get('models_available', [])))
        else:
            st.warning("⏳ Not Initialized")

        if st.session_state.predictions_ready:
            st.success("✅ Predictions Ready")

        # Demo mode indicator
        if st.session_state.demo_mode:
            st.info("📊 Demo Mode Active")

        # System log
        if st.session_state.system_log:
            st.markdown("### 📋 Activity Log")
            with st.expander("Recent Activity", expanded=False):
                for log_entry in st.session_state.system_log[-8:]:
                    st.text(log_entry)

def initialize_system():
    """Initialize prediction system with comprehensive error handling"""
    try:
        add_log("🚀 Starting system initialization...")

        progress_bar = st.progress(0)
        status_text = st.empty()

        with st.spinner("🔄 Initializing system..."):
            # Create system
            status_text.text("Creating prediction system...")
            add_log("📦 Creating system components...")
            progress_bar.progress(20)

            system = BulletproofPredictionSystem()

            # Check if system creation was successful
            if not hasattr(system, 'data_fetcher'):
                add_log("❌ System creation failed")
                status_text.error("❌ System creation failed")
                return

            # Initialize system
            status_text.text("Fetching data and training models...")
            add_log("📊 Fetching NIFTY 50 data...")
            progress_bar.progress(40)

            # Use smaller dataset for faster initialization
            init_success = system.initialize_system(200)  # 200 days
            progress_bar.progress(90)

            if init_success:
                st.session_state.prediction_system = system
                st.session_state.system_ready = True

                add_log("✅ System initialized successfully!")
                status_text.success("✅ System ready!")

                # Get system status
                sys_status = system.get_system_status()
                add_log(f"📊 Loaded {sys_status['data_points']} data points")
                add_log(f"🤖 Trained {len(sys_status['models_available'])} models")

                # Check for demo mode
                market_data = system.get_market_overview()
                if market_data and market_data.get('is_demo_data', False):
                    st.session_state.demo_mode = True
                    add_log("📊 Running in demo mode (no live data)")

                progress_bar.progress(100)
                time.sleep(2)
                st.rerun()

            else:
                add_log("❌ System initialization failed")
                status_text.error("❌ Initialization failed - check log for details")

                # Show troubleshooting tips
                st.error("""
                **Initialization Failed. Try:**
                1. Check internet connection
                2. Run as Administrator
                3. Use EMERGENCY_LAUNCHER.py
                4. Manual setup via command line
                """)

    except Exception as e:
        error_msg = str(e)
        add_log(f"❌ Initialization error: {error_msg}")
        st.error(f"❌ Error: {error_msg}")

        # Show technical details in expander
        with st.expander("🔍 Technical Details"):
            st.code(traceback.format_exc())

def generate_predictions():
    """Generate predictions with error handling"""
    try:
        add_log("🔮 Generating predictions...")

        with st.spinner("🔮 Generating 7-day predictions..."):
            predictions = st.session_state.prediction_system.generate_predictions()

            if predictions:
                st.session_state.current_predictions = predictions
                st.session_state.predictions_ready = True

                add_log(f"✅ Generated {len(predictions)} model predictions")
                st.success("✅ Predictions generated!")

                time.sleep(1)
                st.rerun()
            else:
                add_log("❌ Prediction generation failed")
                st.error("❌ Prediction generation failed")

    except Exception as e:
        add_log(f"❌ Prediction error: {str(e)}")
        st.error(f"❌ Prediction Error: {str(e)}")

def show_welcome_screen():
    """Welcome screen with instructions"""
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 👋 Welcome to NIFTY 50 Predictor!")
        st.markdown("**Fixed Version - All Python 3.12 Issues Resolved**")

        st.markdown("""
        **🎯 This System Provides:**
        - Real-time NIFTY 50 stock data analysis
        - **7-Day Highest Prediction** - Maximum expected price
        - **7-Day Lowest Prediction** - Minimum expected price  
        - Advanced ML predictions with confidence analysis
        - Professional risk assessment and trading signals
        - Interactive charts and comprehensive market insights

        **🔧 Python 3.12 Fixes Applied:**
        - ✅ Resolved setuptools/pip compatibility issues
        - ✅ Fixed `pkgutil.ImpImporter` error
        - ✅ Bypassed corrupted package installations
        - ✅ Enhanced error handling and recovery
        - ✅ Pre-compiled wheel installations only
        - ✅ Fallback modes for missing packages
        """)

    with col2:
        st.markdown("### 🚀 Getting Started")

        st.markdown("""
        **Quick Steps:**
        1. 🚀 Click **"Initialize System"**
        2. ⏳ Wait 2-3 minutes for setup
        3. ✅ Look for "System Ready" status
        4. 🔮 Click **"Generate Predictions"**
        5. 📊 Explore detailed analysis

        **If Issues Occur:**
        - Check Activity Log in sidebar
        - Try EMERGENCY_LAUNCHER.py
        - Run as Administrator
        - Check internet connection
        """)

        # System diagnostics
        with st.expander("🔍 System Diagnostics"):
            st.write("Python Version:", sys.version[:10])
            st.write("Streamlit Available:", "✅" if STREAMLIT_AVAILABLE else "❌")
            st.write("Core Packages:", "✅" if CORE_AVAILABLE else "❌")  
            st.write("Plotly Charts:", "✅" if PLOTLY_AVAILABLE else "❌")
            st.write("System Modules:", "✅" if SYSTEM_AVAILABLE else "❌")

def show_main_dashboard():
    """Main dashboard with all features"""
    # Market overview
    show_market_overview()

    # Historical chart
    show_historical_chart()

    # Predictions section
    if st.session_state.predictions_ready and st.session_state.current_predictions:
        show_predictions_section()
        show_analysis_section()
        show_risk_assessment()
    else:
        st.info("🔮 Click 'Generate Predictions' to see comprehensive 7-day forecasts")

    # Model performance
    show_model_performance()

def show_market_overview():
    """Display market overview with error handling"""
    st.markdown("### 📊 Current Market Overview")

    try:
        market_data = st.session_state.prediction_system.get_market_overview()

        if market_data:
            # Display demo mode warning
            if market_data.get('is_demo_data', False):
                st.warning("📊 **Demo Mode**: Using simulated data for demonstration")

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric(
                    "💰 Current Price",
                    f"₹{market_data['current_price']:.2f}",
                    delta=f"₹{market_data['daily_change']:.2f}"
                )

            with col2:
                st.metric("📊 Daily Change", f"{market_data['daily_change_pct']:.2f}%")

            with col3:
                st.metric("📈 Week High", f"₹{market_data['week_high']:.2f}")

            with col4:
                st.metric("📉 Week Low", f"₹{market_data['week_low']:.2f}")

            with col5:
                volume_ratio = market_data['volume'] / market_data['avg_volume_20d']
                st.metric(
                    "📊 Volume",
                    f"{market_data['volume']/1000000:.1f}M",
                    delta=f"{volume_ratio:.1f}x avg"
                )

    except Exception as e:
        st.error(f"❌ Market overview error: {str(e)}")

def show_historical_chart():
    """Show historical price chart"""
    st.markdown("### 📊 Historical Price Analysis")

    if not PLOTLY_AVAILABLE:
        st.warning("⚠️ Charts unavailable - plotly not installed")
        return

    try:
        data = st.session_state.prediction_system.get_historical_data(60)

        if data is not None and len(data) > 0:
            fig = go.Figure()

            # Candlestick chart
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'], 
                low=data['Low'],
                close=data['Close'],
                name='NIFTY 50'
            ))

            # Moving averages if available
            if 'MA_5' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index, y=data['MA_5'],
                    mode='lines', name='5-Day MA',
                    line=dict(color='orange', width=2)
                ))

            if 'MA_20' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index, y=data['MA_20'],
                    mode='lines', name='20-Day MA',
                    line=dict(color='blue', width=2)
                ))

            fig.update_layout(
                title="NIFTY 50 - Last 60 Days",
                xaxis_title="Date",
                yaxis_title="Price (₹)",
                height=500,
                xaxis_rangeslider_visible=False
            )

            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Chart error: {str(e)}")

def show_predictions_section():
    """Display prediction results"""
    st.markdown("### 🔮 7-Day Predictions with High/Low Analysis")

    try:
        predictions = st.session_state.current_predictions

        # Create prediction table
        pred_data = []
        for model, pred in predictions.items():
            pred_data.append({
                'Model': model,
                'Current': f"₹{pred['current_price']:.2f}",
                'Base Prediction': f"₹{pred['base_prediction']:.2f}",
                '📈 Highest (7d)': f"₹{pred['highest_7d']:.2f}",
                '📉 Lowest (7d)': f"₹{pred['lowest_7d']:.2f}",
                'Change %': f"{pred['base_change_pct']:+.2f}%",
                'Trend': f"{pred['trend']} ({pred['trend_strength']})"
            })

        df = pd.DataFrame(pred_data)
        st.dataframe(df, use_container_width=True)

        # Create chart if plotly available
        if PLOTLY_AVAILABLE:
            create_prediction_chart(predictions)

    except Exception as e:
        st.error(f"❌ Predictions display error: {str(e)}")

def create_prediction_chart(predictions):
    """Create prediction visualization"""
    try:
        fig = go.Figure()

        for model in predictions.keys():
            pred = predictions[model]

            scenarios = ['Current', 'Low (7d)', 'Base', 'High (7d)']
            values = [
                pred['current_price'],
                pred['lowest_7d'],
                pred['base_prediction'], 
                pred['highest_7d']
            ]
            colors = ['lightgray', 'red', 'lightblue', 'green']

            fig.add_trace(go.Bar(
                name=model,
                x=[f'{model}\n{s}' for s in scenarios],
                y=values,
                marker_color=colors,
                text=[f"₹{v:.0f}" for v in values],
                textposition='auto'
            ))

        fig.update_layout(
            title="7-Day Prediction Scenarios",
            xaxis_title="Scenarios",
            yaxis_title="Price (₹)",
            height=500,
            barmode='group',
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Chart creation error: {str(e)}")

def show_analysis_section():
    """Show detailed analysis"""
    st.markdown("### 📊 Risk-Reward Analysis")

    try:
        predictions = st.session_state.current_predictions

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📈 Upside Analysis")
            upside_data = []
            for model, pred in predictions.items():
                upside_data.append({
                    'Model': model,
                    'Upside Potential': f"₹{pred['upside_potential']:.2f}",
                    'Upside %': f"{pred['upside_pct']:.2f}%"
                })
            st.dataframe(pd.DataFrame(upside_data), use_container_width=True)

        with col2:
            st.markdown("#### 📉 Downside Analysis")
            downside_data = []
            for model, pred in predictions.items():
                downside_data.append({
                    'Model': model,
                    'Downside Risk': f"₹{pred['downside_risk']:.2f}",
                    'Risk %': f"{pred['downside_pct']:.2f}%"
                })
            st.dataframe(pd.DataFrame(downside_data), use_container_width=True)

    except Exception as e:
        st.error(f"❌ Analysis error: {str(e)}")

def show_risk_assessment():
    """Show risk assessment"""
    st.markdown("### ⚠️ Risk Assessment & Trading Signals")

    try:
        predictions = st.session_state.current_predictions

        # Calculate consensus
        bullish_count = sum(1 for p in predictions.values() if p['trend'] == 'Bullish')
        total_models = len(predictions)
        consensus = bullish_count / total_models if total_models > 0 else 0

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 🎯 Consensus Signal")
            if consensus >= 0.75:
                signal = "🟢 Strong Buy"
            elif consensus >= 0.5:
                signal = "🔵 Buy"
            elif consensus >= 0.25:
                signal = "🟡 Hold"
            else:
                signal = "🔴 Sell"

            st.markdown(f"**{signal}**")
            st.progress(consensus)
            st.caption(f"{bullish_count}/{total_models} models bullish")

        with col2:
            st.markdown("#### 📊 Model Confidence")
            avg_uncertainty = np.mean([p['uncertainty'] for p in predictions.values()])
            current_price = list(predictions.values())[0]['current_price']
            confidence = max(0, min(100, (1 - avg_uncertainty/current_price) * 100))

            st.metric("Confidence Level", f"{confidence:.1f}%")

        with col3:
            st.markdown("#### 🎲 Risk Level")
            avg_vol = np.mean([p['uncertainty']/p['current_price']*100 for p in predictions.values()])

            if avg_vol > 5:
                risk = "🔴 High"
            elif avg_vol > 3:
                risk = "🟡 Moderate"
            else:
                risk = "🟢 Low"

            st.markdown(f"**{risk}**")
            st.metric("Expected Volatility", f"{avg_vol:.2f}%")

    except Exception as e:
        st.error(f"❌ Risk assessment error: {str(e)}")

def show_model_performance():
    """Show model performance"""
    st.markdown("### 🎯 Model Performance Analysis")

    try:
        performance = st.session_state.prediction_system.get_model_performance()

        if performance:
            perf_data = []
            for model, metrics in performance.items():
                perf_data.append({
                    'Model': model,
                    'MAE (₹)': f"{metrics['MAE']:.2f}",
                    'RMSE (₹)': f"{metrics['RMSE']:.2f}",
                    'R² Score': f"{metrics['R2']:.4f}",
                    'Status': '✅ Trained'
                })

            st.dataframe(pd.DataFrame(perf_data), use_container_width=True)

            # Feature importance if available
            if PLOTLY_AVAILABLE:
                importance = st.session_state.prediction_system.get_feature_importance()
                if importance is not None and len(importance) > 0:
                    st.markdown("#### 🔍 Feature Importance")

                    top_features = importance.head(6)
                    fig = px.bar(
                        top_features, x='Importance', y='Feature',
                        orientation='h', title="Key Prediction Features"
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Performance display error: {str(e)}")

# Application entry point
if __name__ == "__main__":
    main()
