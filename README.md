This is a repository with code used to prepare my master thesis called "Volatility Regime Switching in Algorithmic Investment Strategies on S&P500Index". 
In folder R there are modules written in R programming language. There were used to prepare S&P500 Index daily returns forecasts. To run code, please
insert the two files in the same working directory and run code from module called arima_garch_predictions.R.

The rest of the repository includes Python code and Jupyter Notebooks. In folder backtesting, there are notebook employed to perform backtesting of the 
strategy on the historical data. The notebook all_strategies_results.ipynb was prepared to generate plots, tables, etc. based on .csv files with predictions 
from constructed strategies. The notebook final_strategy_3_regimes.ipynb was prepared to backtest the strategies that do not use XGBoost to predict
volatility regime. The notebook ml_pred_strategy_2_regimes_hyperparams_tuning.ipynb was prepared to perform backtesting of the strategies that used XGBoost
to predict volatility regimes.
