"""
Main execution file for the ADR Arbitrage AI System
==================================================

This file contains the main orchestration logic for running the complete
ADR arbitrage system including data collection, model training, backtesting,
and optimization.
"""

import asyncio
import os
import sys
from pathlib import Path
import argparse
import yaml
import json
from datetime import datetime, timedelta
import logging

# Import our modules
try:
    from adr_arbitrage_system import *
except ImportError:
    print("Please ensure all required packages are installed. Run: pip install -r requirements.txt")
    sys.exit(1)

class ADRArbitrageEngine:
    """Main engine for ADR arbitrage system"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.data_provider = DataProvider(self.config.TIINGO_API_KEY)
        self.feature_engineer = FeatureEngineering()
        self.backtest_engine = BacktestEngine(self.config)
        self.optimization_engine = OptimizationEngine(self.config)
        
        # Create directories
        for path in [self.config.DATA_PATH, self.config.MODEL_PATH, self.config.RESULTS_PATH]:
            Path(path).mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.models = {}
        self.results = {}
        
        logger.info("ADR Arbitrage Engine initialized")
    
    def _load_config(self, config_path: str) -> Config:
        """Load configuration from file or use defaults"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            # Update Config with loaded values
            config = Config()
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            return config
        else:
            logger.info("No config file found, using defaults")
            return Config()
    
    async def collect_data(self, start_date: str = "2020-01-01", end_date: str = None):
        """Collect data for all ADR pairs"""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        logger.info(f"Collecting data from {start_date} to {end_date}")
        
        # Get all symbols (ADRs + underlying stocks)
        all_symbols = []
        for adr, info in self.config.ADR_PAIRS.items():
            all_symbols.extend([adr, info['underlying']])
        
        # Fetch data concurrently
        data_dict = await self.data_provider.fetch_multiple_symbols(
            all_symbols, start_date, end_date
        )
        
        # Store data
        for symbol, df in data_dict.items():
            df.to_csv(f"{self.config.DATA_PATH}/{symbol}.csv")
        
        logger.info(f"Collected data for {len(data_dict)} symbols")
        return data_dict
    
    def engineer_features(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Engineer features for all ADR pairs"""
        logger.info("Engineering features...")
        
        engineered_data = {}
        
        for adr_symbol, adr_info in self.config.ADR_PAIRS.items():
            if adr_symbol not in data_dict:
                logger.warning(f"No data for ADR {adr_symbol}")
                continue
            
            underlying_symbol = adr_info['underlying']
            if underlying_symbol not in data_dict:
                logger.warning(f"No data for underlying {underlying_symbol}")
                continue
            
            # Get data
            adr_data = data_dict[adr_symbol].copy()
            underlying_data = data_dict[underlying_symbol].copy()
            
            # Calculate technical indicators
            adr_data = self.feature_engineer.calculate_technical_indicators(adr_data)
            underlying_data = self.feature_engineer.calculate_technical_indicators(underlying_data)
            
            # Calculate ADR-specific features
            ratio = adr_info['ratio']
            adr_data = self.feature_engineer.calculate_adr_features(
                adr_data, underlying_data, ratio
            )
            
            # Calculate regime features
            adr_data = self.feature_engineer.calculate_regime_features(adr_data)
            
            engineered_data[adr_symbol] = adr_data
        
        logger.info(f"Engineered features for {len(engineered_data)} ADR pairs")
        return engineered_data
    
    def train_models(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Train all models for each ADR pair"""
        logger.info("Training models...")
        
        models = {}
        
        for adr_symbol, data in data_dict.items():
            if data.empty or len(data) < 100:
                logger.warning(f"Insufficient data for {adr_symbol}")
                continue
            
            logger.info(f"Training models for {adr_symbol}")
            
            # Prepare features and targets
            feature_cols = [
                'returns', 'volatility', 'rsi', 'macd', 'arbitrage_spread',
                'rolling_corr', 'spread_zscore', 'vol_regime', 'trend_regime'
            ]
            
            # Remove NaN values
            clean_data = data.dropna(subset=feature_cols + ['arbitrage_spread'])
            
            if len(clean_data) < 100:
                logger.warning(f"Insufficient clean data for {adr_symbol}")
                continue
            
            # Split data
            train_size = int(0.7 * len(clean_data))
            val_size = int(0.15 * len(clean_data))
            
            train_data = clean_data.iloc[:train_size]
            val_data = clean_data.iloc[train_size:train_size + val_size]
            test_data = clean_data.iloc[train_size + val_size:]
            
            X_train = train_data[feature_cols].values
            y_train = train_data['arbitrage_spread'].values
            X_val = val_data[feature_cols].values
            y_val = val_data['arbitrage_spread'].values
            X_test = test_data[feature_cols].values
            y_test = test_data['arbitrage_spread'].values
            
            adr_models = {}
            
            # 1. Train Ensemble Model
            try:
                ensemble_model = EnsembleModel()
                ensemble_model.fit(X_train, y_train, X_val, y_val)
                
                # Make predictions
                train_pred = ensemble_model.predict(X_train)
                val_pred = ensemble_model.predict(X_val)
                test_pred = ensemble_model.predict(X_test)
                
                # Calculate metrics
                ensemble_metrics = {
                    'train_mse': mean_squared_error(y_train, train_pred),
                    'val_mse': mean_squared_error(y_val, val_pred),
                    'test_mse': mean_squared_error(y_test, test_pred),
                    'test_r2': r2_score(y_test, test_pred)
                }
                
                adr_models['ensemble'] = {
                    'model': ensemble_model,
                    'metrics': ensemble_metrics,
                    'predictions': test_pred
                }
                
                logger.info(f"Ensemble model for {adr_symbol} - Test R²: {ensemble_metrics['test_r2']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training ensemble model for {adr_symbol}: {e}")
            
            # 2. Train LSTM Model
            try:
                # Prepare LSTM data
                sequence_length = 60
                if len(clean_data) > sequence_length + 20:
                    lstm_model = LSTMPredictor(
                        input_shape=(sequence_length, len(feature_cols)),
                        prediction_horizon=1
                    )
                    
                    # Prepare sequences
                    X_lstm, y_lstm = lstm_model.prepare_data(
                        clean_data, feature_cols, 'arbitrage_spread', sequence_length
                    )
                    
                    if len(X_lstm) > 50:
                        # Split data for LSTM
                        lstm_train_size = int(0.7 * len(X_lstm))
                        lstm_val_size = int(0.15 * len(X_lstm))
                        
                        X_lstm_train = X_lstm[:lstm_train_size]
                        y_lstm_train = y_lstm[:lstm_train_size]
                        X_lstm_val = X_lstm[lstm_train_size:lstm_train_size + lstm_val_size]
                        y_lstm_val = y_lstm[lstm_train_size:lstm_train_size + lstm_val_size]
                        X_lstm_test = X_lstm[lstm_train_size + lstm_val_size:]
                        y_lstm_test = y_lstm[lstm_train_size + lstm_val_size:]
                        
                        # Train LSTM
                        history = lstm_model.train(
                            X_lstm_train, y_lstm_train,
                            X_lstm_val, y_lstm_val,
                            epochs=50
                        )
                        
                        # Make predictions
                        lstm_test_pred = lstm_model.predict(X_lstm_test)
                        
                        lstm_metrics = {
                            'test_mse': mean_squared_error(y_lstm_test.flatten(), lstm_test_pred.flatten()),
                            'test_r2': r2_score(y_lstm_test.flatten(), lstm_test_pred.flatten())
                        }
                        
                        adr_models['lstm'] = {
                            'model': lstm_model,
                            'metrics': lstm_metrics,
                            'history': history,
                            'predictions': lstm_test_pred
                        }
                        
                        logger.info(f"LSTM model for {adr_symbol} - Test R²: {lstm_metrics['test_r2']:.4f}")
                
            except Exception as e:
                logger.error(f"Error training LSTM model for {adr_symbol}: {e}")
            
            # 3. Train Deep Q-Learning Agent
            try:
                state_size = len(feature_cols)
                dql_agent = DeepQLearningAgent(state_size)
                
                # Create trading environment for training
                for i in range(len(train_data) - 1):
                    state = train_data[feature_cols].iloc[i].values
                    next_state = train_data[feature_cols].iloc[i + 1].values
                    
                    # Calculate reward based on spread change
                    current_spread = train_data['arbitrage_spread'].iloc[i]
                    next_spread = train_data['arbitrage_spread'].iloc[i + 1]
                    
                    # Simple reward: profit from predicting spread direction
                    action = dql_agent.act(state)
                    
                    if action == 1:  # Long ADR, Short Underlying
                        reward = -current_spread if current_spread > 0 else abs(current_spread)
                    elif action == 2:  # Short ADR, Long Underlying
                        reward = current_spread if current_spread < 0 else -abs(current_spread)
                    else:  # Hold
                        reward = 0
                    
                    dql_agent.remember(state, action, reward, next_state, False)
                    
                    if i % 50 == 0:
                        dql_agent.replay()
                
                # Update target network
                dql_agent.update_target_network()
                
                adr_models['dql'] = {
                    'model': dql_agent,
                    'epsilon': dql_agent.epsilon
                }
                
                logger.info(f"DQL agent trained for {adr_symbol}")
                
            except Exception as e:
                logger.error(f"Error training DQL agent for {adr_symbol}: {e}")
            
            models[adr_symbol] = adr_models
        
        self.models = models
        logger.info("Model training completed")
        return models
    
    def generate_signals(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Generate trading signals using trained models"""
        logger.info("Generating trading signals...")
        
        signals = {}
        
        for adr_symbol, data in data_dict.items():
            if adr_symbol not in self.models:
                continue
            
            adr_models = self.models[adr_symbol]
            signal_data = []
            
            feature_cols = [
                'returns', 'volatility', 'rsi', 'macd', 'arbitrage_spread',
                'rolling_corr', 'spread_zscore', 'vol_regime', 'trend_regime'
            ]
            
            clean_data = data.dropna(subset=feature_cols)
            
            for idx, row in clean_data.iterrows():
                features = row[feature_cols].values.reshape(1, -1)
                
                # Get predictions from different models
                ensemble_signal = 0
                lstm_signal = 0
                dql_signal = 0
                confidence = 0
                
                # Ensemble model prediction
                if 'ensemble' in adr_models:
                    try:
                        spread_pred = adr_models['ensemble']['model'].predict(features)[0]
                        ensemble_signal = np.tanh(spread_pred * 10)  # Scale and bound signal
                        confidence += 0.4  # Weight ensemble model
                    except:
                        pass
                
                # LSTM model prediction
                if 'lstm' in adr_models:
                    try:
                        # For LSTM, we need sequence data
                        # This is simplified - in practice, maintain sequence buffer
                        lstm_signal = 0  # Placeholder
                        confidence += 0.3
                    except:
                        pass
                
                # DQL agent action
                if 'dql' in adr_models:
                    try:
                        action = adr_models['dql']['model'].act(features.flatten())
                        if action == 1:
                            dql_signal = 1
                        elif action == 2:
                            dql_signal = -1
                        else:
                            dql_signal = 0
                        confidence += 0.3
                    except:
                        pass
                
                # Combine signals
                combined_signal = (0.4 * ensemble_signal + 0.3 * lstm_signal + 0.3 * dql_signal)
                
                # Adjust confidence based on spread magnitude and regime
                spread_magnitude = abs(row['arbitrage_spread'])
                if spread_magnitude > 0.02:  # 2% spread threshold
                    confidence *= 1.2
                
                if row['vol_regime'] == 1:  # High volatility regime
                    confidence *= 0.8
                
                signal_data.append({
                    'date': idx,
                    'signal': combined_signal,
                    'confidence': min(confidence, 1.0),
                    'ensemble_signal': ensemble_signal,
                    'dql_signal': dql_signal,
                    'spread': row['arbitrage_spread'],
                    'spread_zscore': row['spread_zscore']
                })
            
            signals_df = pd.DataFrame(signal_data)
            signals_df.set_index('date', inplace=True)
            signals[adr_symbol] = signals_df
        
        logger.info(f"Generated signals for {len(signals)} ADR pairs")
        return signals
    
    def run_backtest(self, data_dict: Dict[str, pd.DataFrame], 
                    signals_dict: Dict[str, pd.DataFrame],
                    start_date: str = "2022-01-01", 
                    end_date: str = "2023-12-31") -> Dict:
        """Run comprehensive backtest"""
        logger.info(f"Running backtest from {start_date} to {end_date}")
        
        results = self.backtest_engine.run_backtest(
            data_dict, signals_dict, start_date, end_date
        )
        
        # Log results
        metrics = results['metrics']
        logger.info(f"Backtest Results:")
        logger.info(f"Total Return: {results['total_return']:.2%}")
        logger.info(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
        logger.info(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        logger.info(f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.4f}")
        
        return results
    
    def optimize_strategy(self, data_dict: Dict[str, pd.DataFrame]) -> Dict:
        """Optimize strategy parameters"""
        logger.info("Optimizing strategy parameters...")
        
        # Define parameter space for optimization
        param_space = {
            'confidence_threshold': (0.6, 0.9, 'real'),
            'spread_threshold': (0.01, 0.05, 'real'),
            'lookback_window': (100, 300, 'int'),
            'stop_loss': (0.03, 0.08, 'real'),
        }
        
        def objective_function(data: Dict, params: Dict) -> float:
            """Objective function for optimization (returns Sharpe ratio)"""
            # Update config with new parameters
            temp_config = Config()
            temp_config.CONFIDENCE_THRESHOLD = params['confidence_threshold']
            temp_config.SPREAD_THRESHOLD = params['spread_threshold']
            temp_config.LOOKBACK_WINDOW = params['lookback_window']
            temp_config.STOP_LOSS = params['stop_loss']
            
            # Run quick backtest with new parameters
            temp_engine = BacktestEngine(temp_config)
            
            # Use existing signals but adjust confidence threshold
            temp_signals = {}
            for symbol, signals_df in self.generate_signals(data).items():
                temp_signals_df = signals_df.copy()
                # Filter signals by new confidence threshold
                temp_signals_df.loc[temp_signals_df['confidence'] < params['confidence_threshold'], 'signal'] = 0
                temp_signals[symbol] = temp_signals_df
            
            try:
                results = temp_engine.run_backtest(
                    data, temp_signals, "2022-01-01", "2023-06-30"
                )
                return results['metrics'].get('sharpe_ratio', 0)
            except:
                return 0
        
        # Run optimization
        optimization_results = self.optimization_engine.optimize_hyperparameters(
            data_dict, objective_function, param_space
        )
        
        logger.info(f"Optimization completed. Best Sharpe: {optimization_results['best_score']:.4f}")
        logger.info(f"Best parameters: {optimization_results['best_params']}")
        
        return optimization_results
    
    def save_results(self, results: Dict, filename: str = None):
        """Save results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"adr_arbitrage_results_{timestamp}.json"
        
        filepath = os.path.join(self.config.RESULTS_PATH, filename)
        
        # Convert non-serializable objects to strings/dicts
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, pd.DataFrame):
                serializable_results[key] = value.to_dict('records')
            elif isinstance(value, dict):
                serializable_results[key] = {k: str(v) if not isinstance(v, (int, float, str, list, dict)) else v 
                                           for k, v in value.items()}
            else:
                serializable_results[key] = str(value) if not isinstance(value, (int, float, str, list, dict)) else value
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")
    
    async def run_full_pipeline(self, start_date: str = "2020-01-01", end_date: str = None):
        """Run the complete ADR arbitrage pipeline"""
        logger.info("Starting full ADR arbitrage pipeline...")
        
        try:
            # 1. Collect data
            data_dict = await self.collect_data(start_date, end_date)
            
            # 2. Engineer features
            engineered_data = self.engineer_features(data_dict)
            
            # 3. Train models
            models = self.train_models(engineered_data)
            
            # 4. Generate signals
            signals = self.generate_signals(engineered_data)
            
            # 5. Run backtest
            backtest_results = self.run_backtest(
                engineered_data, signals, "2022-01-01", "2023-12-31"
            )
            
            # 6. Optimize strategy
            optimization_results = self.optimize_strategy(engineered_data)
            
            # 7. Compile final results
            final_results = {
                'backtest_results': backtest_results,
                'optimization_results': optimization_results,
                'model_performance': {symbol: {model_name: model_info.get('metrics', {}) 
                                             for model_name, model_info in models.items()} 
                                    for symbol, models in models.items()},
                'data_summary': {symbol: {'shape': df.shape, 'date_range': [str(df.index.min()), str(df.index.max())]} 
                               for symbol, df in engineered_data.items()},
                'config': self.config.__dict__
            }
            
            # 8. Save results
            self.save_results(final_results)
            
            logger.info("Full pipeline completed successfully!")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in pipeline execution: {e}")
            raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='ADR Arbitrage AI System')
    parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file path')
    parser.add_argument('--start-date', type=str, default='2020-01-01', help='Start date for data collection')
    parser.add_argument('--end-date', type=str, default=None, help='End date for data collection')
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'backtest', 'optimize'], 
                       help='Execution mode')
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = ADRArbitrageEngine(args.config)
    
    if args.mode == 'full':
        # Run full pipeline
        results = asyncio.run(engine.run_full_pipeline(args.start_date, args.end_date))
        
        # Print summary
        print("\n" + "="*50)
        print("ADR ARBITRAGE SYSTEM RESULTS")
        print("="*50)
        
        backtest_metrics = results['backtest_results']['metrics']
        print(f"Total Return: {results['backtest_results']['total_return']:.2%}")
        print(f"Sharpe Ratio: {backtest_metrics.get('sharpe_ratio', 0):.4f}")
        print(f"Maximum Drawdown: {backtest_metrics.get('max_drawdown', 0):.2%}")
        print(f"Calmar Ratio: {backtest_metrics.get('calmar_ratio', 0):.4f}")
        
        # Check if we achieved target Sharpe
        sharpe_ratio = backtest_metrics.get('sharpe_ratio', 0)
        if sharpe_ratio >= engine.config.TARGET_SHARPE:
            print(f"\n🎉 TARGET SHARPE RATIO ACHIEVED: {sharpe_ratio:.4f} >= {engine.config.TARGET_SHARPE}")
        else:
            print(f"\n⚠️  Target Sharpe ratio not achieved: {sharpe_ratio:.4f} < {engine.config.TARGET_SHARPE}")
            print("Consider running optimization or adjusting parameters.")
        
        print("\n" + "="*50)
        
    elif args.mode == 'backtest':
        # Run backtest only
        print("Backtest mode not yet implemented")
        
    elif args.mode == 'optimize':
        # Run optimization only
        print("Optimization mode not yet implemented")

if __name__ == "__main__":
    main()