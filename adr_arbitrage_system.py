"""
Advanced AI Statistical Arbitrage System for ADR Trading
========================================================

A comprehensive machine learning system for detecting and exploiting 
arbitrage opportunities between ADRs and their underlying stocks across
HK/China/Mexico/Brazil markets.

Based on latest research papers:
1. "Advanced Statistical Arbitrage with Reinforcement Learning" (2024)
2. "On statistical arbitrage under a conditional factor model" (2023)
3. "Harnessing Deep Q-Learning for Enhanced Statistical Arbitrage" (2023)

Author: AI System
Date: 2024
"""

import numpy as np
import pandas as pd
import requests
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import asyncio
import aiohttp
import json
import pickle
import sqlite3
from pathlib import Path

# Machine Learning Libraries
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_regression

# Time Series and Signal Processing
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
import ta
from scipy import signal
from scipy.optimize import minimize
from pykalman import KalmanFilter as PyKalmanFilter

# Portfolio and Risk Management
import cvxpy as cp
import yfinance as yf
import quantlib as ql

# Visualization and Analysis
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration class for the ADR arbitrage system"""
    # API Configuration
    TIINGO_API_KEY: str = "5c30f6c2e27d1f902ace1d777b29691a610df388"
    
    # ADR Pairs Configuration
    ADR_PAIRS = {
        # HK/China ADRs
        'BABA': {'underlying': '9988.HK', 'ratio': 8, 'market': 'HK'},
        'JD': {'underlying': '9618.HK', 'ratio': 2, 'market': 'HK'},
        'NTES': {'underlying': '9999.HK', 'ratio': 25, 'market': 'HK'},
        'LI': {'underlying': '2015.HK', 'ratio': 2, 'market': 'HK'},
        'XPEV': {'underlying': '9868.HK', 'ratio': 2, 'market': 'HK'},
        'NIO': {'underlying': '9866.HK', 'ratio': 1, 'market': 'HK'},
        'BIDU': {'underlying': '9888.HK', 'ratio': 10, 'market': 'HK'},
        
        # Mexico ADRs  
        'AMX': {'underlying': 'AMXL.MX', 'ratio': 20, 'market': 'MX'},
        'TV': {'underlying': 'TLEVISA.MX', 'ratio': 5, 'market': 'MX'},
        'KOF': {'underlying': 'KOFUBL.MX', 'ratio': 1, 'market': 'MX'},
        
        # Brazil ADRs
        'VALE': {'underlying': 'VALE3.SA', 'ratio': 1, 'market': 'BR'},
        'ITUB': {'underlying': 'ITUB4.SA', 'ratio': 1, 'market': 'BR'},
        'BBD': {'underlying': 'BBDC4.SA', 'ratio': 1, 'market': 'BR'},
        'PBR': {'underlying': 'PETR4.SA', 'ratio': 2, 'market': 'BR'},
    }
    
    # Model Parameters
    LOOKBACK_WINDOW: int = 252  # 1 year
    PREDICTION_HORIZON: int = 5  # 5 days ahead
    CONFIDENCE_THRESHOLD: float = 0.8
    MAX_POSITION_SIZE: float = 0.1  # 10% max position
    
    # Risk Management
    MAX_DRAWDOWN: float = 0.15  # 15% max drawdown
    STOP_LOSS: float = 0.05  # 5% stop loss
    TARGET_SHARPE: float = 2.0  # Target Sharpe ratio
    
    # Trading Parameters
    TRANSACTION_COST: float = 0.001  # 10 bps
    SPREAD_THRESHOLD: float = 0.02  # 2% minimum spread
    
    # Data Storage
    DATA_PATH: str = "data/"
    MODEL_PATH: str = "models/"
    RESULTS_PATH: str = "results/"

class DataProvider:
    """Data provider for fetching ADR and underlying stock data"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tiingo.com/tiingo"
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Authorization': f'Token {api_key}'
        })
    
    async def fetch_price_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical price data for a symbol"""
        try:
            url = f"{self.base_url}/daily/{symbol}/prices"
            params = {
                'startDate': start_date,
                'endDate': end_date,
                'format': 'json'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers={
                    'Authorization': f'Token {self.api_key}'
                }) as response:
                    if response.status == 200:
                        data = await response.json()
                        df = pd.DataFrame(data)
                        if not df.empty:
                            df['date'] = pd.to_datetime(df['date'])
                            df.set_index('date', inplace=True)
                            df['symbol'] = symbol
                            return df
                    else:
                        logger.error(f"Error fetching data for {symbol}: {response.status}")
                        return pd.DataFrame()
        except Exception as e:
            logger.error(f"Exception fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def fetch_multiple_symbols(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols concurrently"""
        tasks = []
        for symbol in symbols:
            task = self.fetch_price_data(symbol, start_date, end_date)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data_dict = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, pd.DataFrame) and not result.empty:
                data_dict[symbol] = result
            else:
                logger.warning(f"No data received for {symbol}")
        
        return data_dict

class FeatureEngineering:
    """Advanced feature engineering for ADR arbitrage"""
    
    @staticmethod
    def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Moving averages
        for window in [5, 10, 20, 50, 200]:
            df[f'sma_{window}'] = df['close'].rolling(window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
        
        # Technical indicators using TA library
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        df['macd'] = ta.trend.MACD(df['close']).macd()
        df['macd_signal'] = ta.trend.MACD(df['close']).macd_signal()
        df['bb_high'] = ta.volatility.BollingerBands(df['close']).bollinger_hband()
        df['bb_low'] = ta.volatility.BollingerBands(df['close']).bollinger_lband()
        df['bb_mid'] = ta.volatility.BollingerBands(df['close']).bollinger_mavg()
        
        # Volume indicators
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        
        # Momentum indicators
        df['momentum'] = df['close'] / df['close'].shift(10) - 1
        df['roc'] = ta.momentum.ROCIndicator(df['close']).roc()
        
        return df
    
    @staticmethod
    def calculate_adr_features(adr_data: pd.DataFrame, underlying_data: pd.DataFrame, ratio: float) -> pd.DataFrame:
        """Calculate ADR-specific arbitrage features"""
        # Align dates
        common_dates = adr_data.index.intersection(underlying_data.index)
        adr_aligned = adr_data.loc[common_dates].copy()
        underlying_aligned = underlying_data.loc[common_dates].copy()
        
        # Calculate implied underlying price from ADR
        adr_aligned['implied_underlying'] = adr_aligned['close'] / ratio
        
        # Calculate arbitrage spread
        adr_aligned['arbitrage_spread'] = (adr_aligned['implied_underlying'] - underlying_aligned['close']) / underlying_aligned['close']
        adr_aligned['arbitrage_spread_abs'] = np.abs(adr_aligned['arbitrage_spread'])
        
        # Calculate correlation features
        adr_aligned['rolling_corr'] = adr_aligned['returns'].rolling(30).corr(underlying_aligned['returns'])
        
        # Calculate beta (hedge ratio)
        window = 60
        for i in range(window, len(adr_aligned)):
            y = adr_aligned['returns'].iloc[i-window:i].values
            x = underlying_aligned['returns'].iloc[i-window:i].values
            if len(x) > 0 and len(y) > 0:
                beta = np.cov(x, y)[0, 1] / np.var(x) if np.var(x) > 0 else 1.0
                adr_aligned.iloc[i, adr_aligned.columns.get_loc('beta') if 'beta' in adr_aligned.columns else len(adr_aligned.columns)] = beta
        
        if 'beta' not in adr_aligned.columns:
            adr_aligned['beta'] = 1.0
        
        # Calculate z-score of spread
        adr_aligned['spread_zscore'] = (adr_aligned['arbitrage_spread'] - adr_aligned['arbitrage_spread'].rolling(60).mean()) / adr_aligned['arbitrage_spread'].rolling(60).std()
        
        # Add underlying data features
        for col in ['close', 'volume', 'returns']:
            if col in underlying_aligned.columns:
                adr_aligned[f'underlying_{col}'] = underlying_aligned[col]
        
        return adr_aligned
    
    @staticmethod
    def calculate_regime_features(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate regime detection features"""
        # Volatility regime
        df['vol_regime'] = (df['volatility'] > df['volatility'].rolling(252).quantile(0.7)).astype(int)
        
        # Trend regime  
        df['trend_regime'] = np.where(
            df['close'] > df['sma_50'], 1,  # Uptrend
            np.where(df['close'] < df['sma_50'], -1, 0)  # Downtrend
        )
        
        # Market stress regime (based on VIX-like calculation)
        df['stress_regime'] = (df['volatility'] > df['volatility'].rolling(252).quantile(0.9)).astype(int)
        
        return df

class DeepQLearningAgent:
    """Deep Q-Learning agent for ADR arbitrage trading"""
    
    def __init__(self, state_size: int, action_size: int = 3, learning_rate: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size  # 0: Hold, 1: Long ADR/Short Underlying, 2: Short ADR/Long Underlying
        self.learning_rate = learning_rate
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory = []
        self.max_memory = 10000
        
        self.q_network = self._build_model()
        self.target_network = self._build_model()
        self.update_target_network()
    
    def _build_model(self):
        """Build the DQN model"""
        model = models.Sequential([
            layers.Dense(128, activation='relu', input_shape=(self.state_size,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        if len(self.memory) >= self.max_memory:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self, batch_size=32):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        
        states = np.array([self.memory[i][0] for i in batch])
        actions = np.array([self.memory[i][1] for i in batch])
        rewards = np.array([self.memory[i][2] for i in batch])
        next_states = np.array([self.memory[i][3] for i in batch])
        dones = np.array([self.memory[i][4] for i in batch])
        
        current_q_values = self.q_network.predict(states, verbose=0)
        next_q_values = self.target_network.predict(next_states, verbose=0)
        
        for i in range(batch_size):
            if dones[i]:
                current_q_values[i][actions[i]] = rewards[i]
            else:
                current_q_values[i][actions[i]] = rewards[i] + 0.95 * np.max(next_q_values[i])
        
        self.q_network.fit(states, current_q_values, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Update target network weights"""
        self.target_network.set_weights(self.q_network.get_weights())

class LSTMPredictor:
    """LSTM-based price prediction model"""
    
    def __init__(self, input_shape: Tuple[int, int], prediction_horizon: int = 5):
        self.input_shape = input_shape
        self.prediction_horizon = prediction_horizon
        self.model = self._build_model()
        self.scaler = StandardScaler()
    
    def _build_model(self):
        """Build LSTM model"""
        model = models.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=self.input_shape),
            layers.Dropout(0.3),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.3),
            layers.Dense(16, activation='relu'),
            layers.Dense(self.prediction_horizon)
        ])
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    def prepare_data(self, data: pd.DataFrame, features: List[str], target: str, sequence_length: int = 60):
        """Prepare data for LSTM training"""
        # Scale features
        feature_data = self.scaler.fit_transform(data[features].values)
        target_data = data[target].values
        
        X, y = [], []
        for i in range(sequence_length, len(feature_data) - self.prediction_horizon + 1):
            X.append(feature_data[i-sequence_length:i])
            y.append(target_data[i:i+self.prediction_horizon])
        
        return np.array(X), np.array(y)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100):
        """Train the LSTM model"""
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(patience=5, factor=0.5)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        return history
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)

class KalmanFilterModel:
    """Kalman Filter for state space modeling of ADR spreads"""
    
    def __init__(self, n_factors: int = 4):
        self.n_factors = n_factors
        self.kf = None
        self.factor_exposures = None
    
    def fit(self, returns: np.ndarray, factor_exposures: np.ndarray):
        """Fit the Kalman Filter model"""
        n_assets, n_periods = returns.shape
        
        # Initialize Kalman Filter
        self.kf = PyKalmanFilter(
            transition_matrices=np.eye(self.n_factors),
            observation_matrices=factor_exposures,
            initial_state_mean=np.zeros(self.n_factors),
            initial_state_covariance=np.eye(self.n_factors),
            transition_covariance=0.01 * np.eye(self.n_factors),
            observation_covariance=0.1 * np.eye(n_assets)
        )
        
        # Fit the model
        state_means, state_covariances = self.kf.em(returns.T)
        self.factor_exposures = factor_exposures
        
        return state_means, state_covariances
    
    def predict_spread(self, current_state, horizon=5):
        """Predict future spread evolution"""
        predictions = []
        state = current_state
        
        for _ in range(horizon):
            # Predict next state
            next_state = self.kf.transition_matrices @ state
            predictions.append(next_state)
            state = next_state
        
        return np.array(predictions)

class EnsembleModel:
    """Ensemble model combining multiple prediction approaches"""
    
    def __init__(self):
        self.models = {
            'rf': RandomForestRegressor(n_estimators=200, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=200, random_state=42),
        }
        self.weights = None
        self.scaler = RobustScaler()
    
    def fit(self, X_train, y_train, X_val, y_val):
        """Fit ensemble models and calculate optimal weights"""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train individual models
        predictions = {}
        for name, model in self.models.items():
            model.fit(X_train_scaled, y_train)
            predictions[name] = model.predict(X_val_scaled)
        
        # Calculate optimal weights using validation performance
        pred_matrix = np.column_stack(list(predictions.values()))
        
        def objective(weights):
            ensemble_pred = pred_matrix @ weights
            return mean_squared_error(y_val, ensemble_pred)
        
        # Optimize weights
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(len(self.models))]
        
        result = minimize(
            objective,
            x0=np.ones(len(self.models)) / len(self.models),
            bounds=bounds,
            constraints=constraints
        )
        
        self.weights = result.x
        
    def predict(self, X):
        """Make ensemble predictions"""
        X_scaled = self.scaler.transform(X)
        predictions = []
        
        for name, model in self.models.items():
            pred = model.predict(X_scaled)
            predictions.append(pred)
        
        pred_matrix = np.column_stack(predictions)
        return pred_matrix @ self.weights

class RiskManager:
    """Advanced risk management system"""
    
    def __init__(self, config: Config):
        self.config = config
        self.positions = {}
        self.portfolio_value = 1000000  # $1M starting capital
        
    def calculate_position_size(self, signal_strength: float, volatility: float, correlation: float = 0.5) -> float:
        """Calculate optimal position size using Kelly criterion"""
        # Adjust signal strength based on correlation (higher correlation = lower edge)
        adjusted_signal = signal_strength * (1 - correlation)
        
        # Kelly fraction
        if volatility > 0:
            kelly_fraction = adjusted_signal / (volatility ** 2)
            kelly_fraction = np.clip(kelly_fraction, 0, self.config.MAX_POSITION_SIZE)
        else:
            kelly_fraction = 0
        
        # Scale by portfolio value
        position_size = kelly_fraction * self.portfolio_value
        return position_size
    
    def check_risk_limits(self, positions: Dict) -> Dict:
        """Check and enforce risk limits"""
        total_exposure = sum(abs(pos['size']) for pos in positions.values())
        
        # Portfolio concentration limits
        if total_exposure > self.config.MAX_POSITION_SIZE * self.portfolio_value:
            # Scale down positions proportionally
            scale_factor = (self.config.MAX_POSITION_SIZE * self.portfolio_value) / total_exposure
            for symbol in positions:
                positions[symbol]['size'] *= scale_factor
        
        return positions
    
    def calculate_portfolio_metrics(self, returns: pd.Series) -> Dict:
        """Calculate portfolio performance metrics"""
        if len(returns) == 0:
            return {}
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + returns.mean()) ** 252 - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown calculation
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Additional metrics
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'skewness': skewness,
            'kurtosis': kurtosis
        }

class BacktestEngine:
    """Comprehensive backtesting engine"""
    
    def __init__(self, config: Config):
        self.config = config
        self.risk_manager = RiskManager(config)
        
    def run_backtest(self, data: Dict[str, pd.DataFrame], signals: Dict[str, pd.DataFrame], 
                    start_date: str, end_date: str) -> Dict:
        """Run comprehensive backtest"""
        
        # Initialize tracking variables
        portfolio_values = []
        positions = {}
        trades = []
        
        # Get date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        date_range = [d for d in date_range if d in data[list(data.keys())[0]].index]
        
        initial_capital = 1000000
        current_capital = initial_capital
        
        for date in date_range:
            daily_pnl = 0
            
            # Process each ADR pair
            for adr_symbol in self.config.ADR_PAIRS:
                if adr_symbol not in data or date not in data[adr_symbol].index:
                    continue
                
                underlying_symbol = self.config.ADR_PAIRS[adr_symbol]['underlying']
                if underlying_symbol not in data or date not in data[underlying_symbol].index:
                    continue
                
                # Get signals
                if adr_symbol in signals and date in signals[adr_symbol].index:
                    signal = signals[adr_symbol].loc[date, 'signal']
                    confidence = signals[adr_symbol].loc[date, 'confidence']
                    
                    # Get current prices
                    adr_price = data[adr_symbol].loc[date, 'close']
                    underlying_price = data[underlying_symbol].loc[date, 'close']
                    
                    # Calculate position size
                    volatility = data[adr_symbol].loc[date, 'volatility'] if 'volatility' in data[adr_symbol].columns else 0.02
                    position_size = self.risk_manager.calculate_position_size(
                        signal * confidence, volatility
                    )
                    
                    # Check if we should enter/exit positions
                    position_key = f"{adr_symbol}_{underlying_symbol}"
                    
                    if abs(signal) > 0.5 and confidence > self.config.CONFIDENCE_THRESHOLD:
                        # Enter position
                        if position_key not in positions:
                            positions[position_key] = {
                                'adr_shares': 0,
                                'underlying_shares': 0,
                                'entry_date': date,
                                'entry_adr_price': adr_price,
                                'entry_underlying_price': underlying_price
                            }
                        
                        # Update position based on signal
                        ratio = self.config.ADR_PAIRS[adr_symbol]['ratio']
                        
                        if signal > 0:  # Long ADR, Short Underlying
                            adr_shares = position_size / adr_price
                            underlying_shares = -adr_shares * ratio
                        else:  # Short ADR, Long Underlying  
                            adr_shares = -position_size / adr_price
                            underlying_shares = -adr_shares * ratio
                        
                        positions[position_key]['adr_shares'] = adr_shares
                        positions[position_key]['underlying_shares'] = underlying_shares
                        
                        # Record trade
                        trades.append({
                            'date': date,
                            'symbol': position_key,
                            'signal': signal,
                            'confidence': confidence,
                            'adr_price': adr_price,
                            'underlying_price': underlying_price,
                            'position_size': position_size,
                            'action': 'enter'
                        })
                    
                    # Calculate P&L for existing positions
                    if position_key in positions:
                        pos = positions[position_key]
                        if pos['adr_shares'] != 0:
                            # Calculate current P&L
                            adr_pnl = pos['adr_shares'] * (adr_price - pos['entry_adr_price'])
                            underlying_pnl = pos['underlying_shares'] * (underlying_price - pos['entry_underlying_price'])
                            total_pnl = adr_pnl + underlying_pnl
                            
                            daily_pnl += total_pnl
                            
                            # Check for exit conditions
                            total_return = total_pnl / (abs(pos['adr_shares'] * pos['entry_adr_price']) + 
                                                       abs(pos['underlying_shares'] * pos['entry_underlying_price']))
                            
                            if (total_return < -self.config.STOP_LOSS or 
                                abs(signal) < 0.2 or
                                confidence < 0.5):
                                # Exit position
                                trades.append({
                                    'date': date,
                                    'symbol': position_key,
                                    'pnl': total_pnl,
                                    'return': total_return,
                                    'action': 'exit'
                                })
                                del positions[position_key]
            
            # Apply transaction costs
            transaction_cost = len([t for t in trades if t.get('date') == date]) * self.config.TRANSACTION_COST * current_capital
            daily_pnl -= transaction_cost
            
            # Update capital
            current_capital += daily_pnl
            portfolio_values.append({
                'date': date,
                'portfolio_value': current_capital,
                'daily_pnl': daily_pnl,
                'num_positions': len(positions)
            })
        
        # Calculate performance metrics
        portfolio_df = pd.DataFrame(portfolio_values)
        portfolio_df.set_index('date', inplace=True)
        
        returns = portfolio_df['daily_pnl'] / portfolio_df['portfolio_value'].shift(1)
        returns = returns.dropna()
        
        metrics = self.risk_manager.calculate_portfolio_metrics(returns)
        
        return {
            'portfolio_values': portfolio_df,
            'trades': pd.DataFrame(trades),
            'metrics': metrics,
            'final_capital': current_capital,
            'total_return': (current_capital - initial_capital) / initial_capital
        }

class OptimizationEngine:
    """Advanced optimization engine for strategy parameters"""
    
    def __init__(self, config: Config):
        self.config = config
        
    def optimize_hyperparameters(self, data: Dict, objective_func, param_space: Dict) -> Dict:
        """Optimize hyperparameters using Bayesian optimization"""
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer
            from skopt.utils import use_named_args
            
            # Convert param space to skopt format
            dimensions = []
            param_names = []
            
            for name, (min_val, max_val, param_type) in param_space.items():
                param_names.append(name)
                if param_type == 'real':
                    dimensions.append(Real(min_val, max_val, name=name))
                elif param_type == 'int':
                    dimensions.append(Integer(min_val, max_val, name=name))
            
            @use_named_args(dimensions)
            def objective(**params):
                return -objective_func(data, params)  # Negative for minimization
            
            # Run optimization
            result = gp_minimize(objective, dimensions, n_calls=50, random_state=42)
            
            # Return best parameters
            best_params = dict(zip(param_names, result.x))
            return {
                'best_params': best_params,
                'best_score': -result.fun,
                'optimization_history': result
            }
            
        except ImportError:
            logger.warning("scikit-optimize not available, using grid search")
            return self._grid_search_optimization(data, objective_func, param_space)
    
    def _grid_search_optimization(self, data: Dict, objective_func, param_space: Dict) -> Dict:
        """Fallback grid search optimization"""
        from itertools import product
        
        # Generate parameter combinations
        param_values = []
        param_names = list(param_space.keys())
        
        for name, (min_val, max_val, param_type) in param_space.items():
            if param_type == 'real':
                values = np.linspace(min_val, max_val, 5)
            elif param_type == 'int':
                values = np.arange(min_val, max_val + 1, max(1, (max_val - min_val) // 4))
            param_values.append(values)
        
        best_score = -np.inf
        best_params = None
        
        for combination in product(*param_values):
            params = dict(zip(param_names, combination))
            score = objective_func(data, params)
            
            if score > best_score:
                best_score = score
                best_params = params
        
        return {
            'best_params': best_params,
            'best_score': best_score
        }

# Example usage and main function will be in the next file...