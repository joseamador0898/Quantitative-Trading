"""
ADR Arbitrage AI System - Web Application
=========================================

A comprehensive web interface for the ADR arbitrage system using Streamlit.
Provides real-time monitoring, backtesting, and optimization capabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import asyncio
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="ADR Arbitrage AI System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .success-metric {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .warning-metric {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
    }
    .danger-metric {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

def create_sample_data():
    """Create sample data for demonstration"""
    dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    sample_results = {
        'backtest_results': {
            'total_return': 0.28,
            'metrics': {
                'sharpe_ratio': 2.35,
                'max_drawdown': -0.08,
                'calmar_ratio': 2.94,
                'volatility': 0.12,
                'annualized_return': 0.28
            },
            'portfolio_values': pd.DataFrame({
                'date': dates[:500],
                'portfolio_value': 1000000 * np.cumprod(1 + np.random.normal(0.001, 0.02, 500)),
                'daily_pnl': np.random.normal(1000, 5000, 500),
                'num_positions': np.random.randint(0, 10, 500)
            }),
            'trades': pd.DataFrame({
                'date': np.random.choice(dates[:500], 50),
                'symbol': np.random.choice(['BABA', 'JD', 'NTES', 'VALE', 'ITUB'], 50),
                'action': np.random.choice(['enter', 'exit'], 50),
                'pnl': np.random.normal(500, 1000, 50),
                'signal': np.random.uniform(-1, 1, 50),
                'confidence': np.random.uniform(0.5, 1.0, 50)
            })
        },
        'model_performance': {
            'BABA': {
                'ensemble': {'test_r2': 0.72, 'test_mse': 0.0045},
                'lstm': {'test_r2': 0.68, 'test_mse': 0.0052},
                'dql': {'epsilon': 0.05}
            },
            'JD': {
                'ensemble': {'test_r2': 0.69, 'test_mse': 0.0048},
                'lstm': {'test_r2': 0.65, 'test_mse': 0.0055},
                'dql': {'epsilon': 0.03}
            },
            'VALE': {
                'ensemble': {'test_r2': 0.74, 'test_mse': 0.0041},
                'lstm': {'test_r2': 0.71, 'test_mse': 0.0046},
                'dql': {'epsilon': 0.07}
            }
        },
        'optimization_results': {
            'best_score': 2.35,
            'best_params': {
                'confidence_threshold': 0.75,
                'spread_threshold': 0.025,
                'lookback_window': 180,
                'stop_loss': 0.045
            }
        }
    }
    return sample_results

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🚀 ADR Arbitrage AI System</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced Statistical Arbitrage using Machine Learning & Deep Learning")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Dashboard", "Data Analysis", "Model Performance", "Backtesting", "Live Trading", "Research Papers"]
    )
    
    # Load sample data if not loaded
    if not st.session_state.data_loaded:
        st.session_state.results = create_sample_data()
        st.session_state.data_loaded = True
        st.session_state.models_trained = True
    
    if page == "Dashboard":
        show_dashboard()
    elif page == "Data Analysis":
        show_data_analysis()
    elif page == "Model Performance":
        show_model_performance()
    elif page == "Backtesting":
        show_backtesting()
    elif page == "Live Trading":
        show_live_trading()
    elif page == "Research Papers":
        show_research_papers()

def show_dashboard():
    """Show main dashboard"""
    st.header("📊 System Overview")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    if st.session_state.results:
        metrics = st.session_state.results['backtest_results']['metrics']
        
        with col1:
            sharpe = metrics['sharpe_ratio']
            color = "success" if sharpe >= 2.0 else "warning" if sharpe >= 1.5 else "danger"
            st.markdown(f"""
            <div class="metric-card {color}-metric">
                <h3>Sharpe Ratio</h3>
                <h2>{sharpe:.2f}</h2>
                <p>Target: ≥2.0</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            total_return = st.session_state.results['backtest_results']['total_return']
            st.markdown(f"""
            <div class="metric-card success-metric">
                <h3>Total Return</h3>
                <h2>{total_return:.1%}</h2>
                <p>Annualized</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            max_dd = metrics['max_drawdown']
            color = "success" if max_dd >= -0.1 else "warning" if max_dd >= -0.15 else "danger"
            st.markdown(f"""
            <div class="metric-card {color}-metric">
                <h3>Max Drawdown</h3>
                <h2>{max_dd:.1%}</h2>
                <p>Risk Metric</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            calmar = metrics['calmar_ratio']
            st.markdown(f"""
            <div class="metric-card success-metric">
                <h3>Calmar Ratio</h3>
                <h2>{calmar:.2f}</h2>
                <p>Risk-Adj Return</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Portfolio Performance Chart
    st.subheader("📈 Portfolio Performance")
    
    if st.session_state.results:
        portfolio_data = st.session_state.results['backtest_results']['portfolio_values']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Portfolio Value', 'Daily P&L', 'Number of Positions', 'Cumulative Returns'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Portfolio value
        fig.add_trace(
            go.Scatter(x=portfolio_data['date'], y=portfolio_data['portfolio_value'],
                      name='Portfolio Value', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # Daily P&L
        fig.add_trace(
            go.Scatter(x=portfolio_data['date'], y=portfolio_data['daily_pnl'],
                      name='Daily P&L', line=dict(color='green', width=1)),
            row=1, col=2
        )
        
        # Number of positions
        fig.add_trace(
            go.Scatter(x=portfolio_data['date'], y=portfolio_data['num_positions'],
                      name='Active Positions', line=dict(color='orange', width=2)),
            row=2, col=1
        )
        
        # Cumulative returns
        returns = portfolio_data['daily_pnl'] / portfolio_data['portfolio_value'].shift(1)
        cum_returns = (1 + returns).cumprod() - 1
        fig.add_trace(
            go.Scatter(x=portfolio_data['date'], y=cum_returns * 100,
                      name='Cumulative Returns (%)', line=dict(color='purple', width=2)),
            row=2, col=2
        )
        
        fig.update_layout(height=600, title="Portfolio Performance Dashboard")
        st.plotly_chart(fig, use_container_width=True)
    
    # ADR Pairs Status
    st.subheader("🌐 ADR Pairs Status")
    
    adr_pairs = {
        'BABA': {'market': 'HK', 'status': 'Active', 'last_signal': 0.75, 'confidence': 0.82},
        'JD': {'market': 'HK', 'status': 'Active', 'last_signal': -0.45, 'confidence': 0.78},
        'NTES': {'market': 'HK', 'status': 'Monitoring', 'last_signal': 0.15, 'confidence': 0.55},
        'VALE': {'market': 'BR', 'status': 'Active', 'last_signal': 0.62, 'confidence': 0.88},
        'ITUB': {'market': 'BR', 'status': 'Active', 'last_signal': -0.38, 'confidence': 0.71},
        'AMX': {'market': 'MX', 'status': 'Monitoring', 'last_signal': 0.22, 'confidence': 0.49},
    }
    
    pairs_df = pd.DataFrame.from_dict(adr_pairs, orient='index').reset_index()
    pairs_df.columns = ['Symbol', 'Market', 'Status', 'Last Signal', 'Confidence']
    
    # Color code the dataframe
    def highlight_status(val):
        if val == 'Active':
            return 'background-color: #d4edda'
        elif val == 'Monitoring':
            return 'background-color: #fff3cd'
        else:
            return 'background-color: #f8d7da'
    
    styled_df = pairs_df.style.applymap(highlight_status, subset=['Status'])
    st.dataframe(styled_df, use_container_width=True)

def show_data_analysis():
    """Show data analysis page"""
    st.header("📊 Data Analysis")
    
    # Sample data for visualization
    st.subheader("ADR vs Underlying Price Comparison")
    
    # Create sample spread data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    adr_price = 100 + np.cumsum(np.random.normal(0, 1, len(dates)))
    underlying_price = adr_price / 8 + np.random.normal(0, 0.5, len(dates))
    spread = (adr_price / 8 - underlying_price) / underlying_price
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('ADR vs Underlying Prices', 'Arbitrage Spread', 'Spread Z-Score'),
        vertical_spacing=0.08
    )
    
    # Prices
    fig.add_trace(
        go.Scatter(x=dates, y=adr_price, name='ADR Price', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=dates, y=underlying_price * 8, name='Underlying Price (Adjusted)', line=dict(color='red')),
        row=1, col=1
    )
    
    # Spread
    fig.add_trace(
        go.Scatter(x=dates, y=spread * 100, name='Spread (%)', line=dict(color='green')),
        row=2, col=1
    )
    fig.add_hline(y=2, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=-2, line_dash="dash", line_color="red", row=2, col=1)
    
    # Z-Score
    spread_mean = np.mean(spread)
    spread_std = np.std(spread)
    z_score = (spread - spread_mean) / spread_std
    
    fig.add_trace(
        go.Scatter(x=dates, y=z_score, name='Z-Score', line=dict(color='purple')),
        row=3, col=1
    )
    fig.add_hline(y=2, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=-2, line_dash="dash", line_color="red", row=3, col=1)
    
    fig.update_layout(height=800, title="BABA ADR Analysis")
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.subheader("📈 Feature Importance Analysis")
    
    features = ['Arbitrage Spread', 'RSI', 'MACD', 'Volatility', 'Rolling Correlation', 
               'Volume Ratio', 'Momentum', 'Bollinger Bands', 'Regime Indicator']
    importance = [0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04, 0.02]
    
    fig = px.bar(x=importance, y=features, orientation='h', 
                title="Feature Importance for ADR Arbitrage Prediction")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def show_model_performance():
    """Show model performance analysis"""
    st.header("🤖 Model Performance")
    
    if st.session_state.results:
        model_perf = st.session_state.results['model_performance']
        
        # Model comparison table
        st.subheader("Model Performance Comparison")
        
        comparison_data = []
        for symbol, models in model_perf.items():
            for model_name, metrics in models.items():
                if 'test_r2' in metrics:
                    comparison_data.append({
                        'Symbol': symbol,
                        'Model': model_name.title(),
                        'R² Score': metrics['test_r2'],
                        'MSE': metrics['test_mse'],
                        'Performance': 'Excellent' if metrics['test_r2'] > 0.7 else 'Good' if metrics['test_r2'] > 0.6 else 'Fair'
                    })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Model performance visualization
        fig = px.bar(comparison_df, x='Symbol', y='R² Score', color='Model',
                    title="Model R² Scores by ADR Pair")
        st.plotly_chart(fig, use_container_width=True)
    
    # Training progress simulation
    st.subheader("🏋️ Model Training Progress")
    
    epochs = list(range(1, 51))
    ensemble_loss = [0.1 * np.exp(-x/20) + 0.01 for x in epochs]
    lstm_loss = [0.12 * np.exp(-x/25) + 0.015 for x in epochs]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=ensemble_loss, name='Ensemble Model', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=epochs, y=lstm_loss, name='LSTM Model', line=dict(color='red')))
    fig.update_layout(title="Training Loss Over Time", xaxis_title="Epoch", yaxis_title="Loss")
    st.plotly_chart(fig, use_container_width=True)

def show_backtesting():
    """Show backtesting results"""
    st.header("📉 Backtesting Results")
    
    if st.session_state.results:
        # Performance metrics
        metrics = st.session_state.results['backtest_results']['metrics']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Performance Metrics")
            metrics_df = pd.DataFrame([
                ["Total Return", f"{st.session_state.results['backtest_results']['total_return']:.2%}"],
                ["Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}"],
                ["Max Drawdown", f"{metrics['max_drawdown']:.2%}"],
                ["Calmar Ratio", f"{metrics['calmar_ratio']:.2f}"],
                ["Volatility", f"{metrics['volatility']:.2%}"],
                ["Annualized Return", f"{metrics['annualized_return']:.2%}"]
            ], columns=["Metric", "Value"])
            st.dataframe(metrics_df, use_container_width=True)
        
        with col2:
            st.subheader("Risk Metrics")
            # Risk gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = metrics['sharpe_ratio'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Sharpe Ratio"},
                gauge = {
                    'axis': {'range': [None, 4]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 1], 'color': "lightgray"},
                        {'range': [1, 2], 'color': "yellow"},
                        {'range': [2, 4], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 2.0
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Trade analysis
        st.subheader("📋 Trade Analysis")
        trades_df = st.session_state.results['backtest_results']['trades']
        
        # Trade statistics
        winning_trades = trades_df[trades_df['pnl'] > 0] if 'pnl' in trades_df.columns else pd.DataFrame()
        losing_trades = trades_df[trades_df['pnl'] < 0] if 'pnl' in trades_df.columns else pd.DataFrame()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Trades", len(trades_df))
        with col2:
            st.metric("Winning Trades", len(winning_trades))
        with col3:
            st.metric("Win Rate", f"{len(winning_trades)/len(trades_df)*100:.1f}%" if len(trades_df) > 0 else "0%")
        with col4:
            avg_pnl = trades_df['pnl'].mean() if 'pnl' in trades_df.columns else 0
            st.metric("Avg P&L", f"${avg_pnl:.0f}")

def show_live_trading():
    """Show live trading interface"""
    st.header("🔴 Live Trading Dashboard")
    
    st.warning("⚠️ This is a demo interface. Live trading requires proper API connections and risk management.")
    
    # Control panel
    col1, col2, col3 = st.columns(3)
    
    with col1:
        trading_mode = st.selectbox("Trading Mode", ["Paper Trading", "Live Trading"])
        
    with col2:
        auto_trading = st.checkbox("Auto Trading Enabled", value=False)
        
    with col3:
        max_position = st.slider("Max Position Size", 0.01, 0.20, 0.10, 0.01)
    
    # Real-time signals (simulated)
    st.subheader("🚨 Real-Time Signals")
    
    current_signals = pd.DataFrame({
        'Symbol': ['BABA', 'JD', 'VALE', 'ITUB', 'AMX'],
        'Current Spread': ['2.3%', '-1.8%', '1.9%', '-2.1%', '0.8%'],
        'Signal Strength': [0.85, -0.72, 0.68, -0.81, 0.25],
        'Confidence': [0.89, 0.76, 0.83, 0.91, 0.52],
        'Action': ['LONG ADR', 'SHORT ADR', 'LONG ADR', 'SHORT ADR', 'HOLD'],
        'Last Updated': ['00:15:32', '00:15:28', '00:15:35', '00:15:29', '00:15:31']
    })
    
    st.dataframe(current_signals, use_container_width=True)
    
    # Position monitoring
    st.subheader("📊 Current Positions")
    
    positions = pd.DataFrame({
        'Symbol': ['BABA', 'VALE'],
        'Side': ['LONG ADR', 'SHORT ADR'],
        'Entry Price': ['$85.20', '$15.40'],
        'Current Price': ['$87.10', '$15.15'],
        'P&L': ['+$1,420', '+$780'],
        'Duration': ['2 days', '4 hours']
    })
    
    st.dataframe(positions, use_container_width=True)

def show_research_papers():
    """Show research papers and documentation"""
    st.header("📚 Research Papers & Documentation")
    
    st.markdown("""
    ## Latest Research Papers on Statistical Arbitrage & ADR Trading
    
    This section contains the latest research papers that informed the development of this AI system.
    """)
    
    papers = [
        {
            "title": "Advanced Statistical Arbitrage with Reinforcement Learning",
            "authors": "Boming Ning, Kiseop Lee",
            "year": "2024",
            "journal": "arXiv:2403.12180",
            "key_insights": [
                "Model-free reinforcement learning framework for statistical arbitrage",
                "Empirical mean reversion time metric for spread construction",
                "State space design capturing recent price movements"
            ],
            "implementation": "Deep Q-Learning agent implemented in our system"
        },
        {
            "title": "On statistical arbitrage under a conditional factor model",
            "authors": "Trent Spears, Stefan Zohren, Stephen Roberts",
            "year": "2023", 
            "journal": "Oxford-Man Institute",
            "key_insights": [
                "State space framework for factor modeling",
                "Kalman Filter for latent factor estimation",
                "29-year empirical study showing consistent performance"
            ],
            "implementation": "Kalman Filter model for factor-based arbitrage"
        },
        {
            "title": "Harnessing Deep Q-Learning for Enhanced Statistical Arbitrage",
            "authors": "Soumyadip Sarkar",
            "year": "2023",
            "journal": "arXiv:2311.10718",
            "key_insights": [
                "Deep Q-Learning for high-frequency arbitrage",
                "Adaptive learning in dynamic market regimes",
                "15% improvement over SOTA models"
            ],
            "implementation": "Enhanced DQN architecture with regime adaptation"
        }
    ]
    
    for i, paper in enumerate(papers):
        with st.expander(f"📄 {paper['title']} ({paper['year']})"):
            st.write(f"**Authors:** {paper['authors']}")
            st.write(f"**Journal:** {paper['journal']}")
            st.write("**Key Insights:**")
            for insight in paper['key_insights']:
                st.write(f"• {insight}")
            st.write(f"**Implementation in our system:** {paper['implementation']}")
            
            # Add download button (simulated)
            st.download_button(
                label="📥 Download PDF",
                data=f"# {paper['title']}\n\nThis is a simulated PDF content for {paper['title']}",
                file_name=f"{paper['title'].replace(' ', '_')}.pdf",
                mime="application/pdf"
            )
    
    # System architecture
    st.subheader("🏗️ System Architecture")
    
    st.markdown("""
    graph TD
        A[Data Collection] --> B[Feature Engineering]
        B --> C[Model Training]
        C --> D[Signal Generation]
        D --> E[Risk Management]
        E --> F[Portfolio Execution]
        F --> G[Performance Monitoring]
        G --> H[Optimization]
        H --> C
        
        C --> I[Ensemble Models]
        C --> J[LSTM Networks]
        C --> K[Deep Q-Learning]
        
        I --> D
        J --> D
        K --> D
    ```
    """)
    
    # Technical specifications
    st.subheader("⚙️ Technical Specifications")
    
    specs = {
        "Data Sources": "Tiingo API, Real-time market data",
        "Models": "Ensemble (RF+GB), LSTM, Deep Q-Learning",
        "Features": "50+ technical and fundamental indicators",
        "Backtesting": "Walk-forward analysis with realistic costs",
        "Risk Management": "Kelly criterion, VaR, drawdown limits",
        "Optimization": "Bayesian optimization for hyperparameters",
        "Execution": "Multi-market, multi-asset arbitrage",
        "Performance": "Target Sharpe Ratio ≥ 2.0"
    }
    
    for key, value in specs.items():
        st.write(f"**{key}:** {value}")

if __name__ == "__main__":
    main()