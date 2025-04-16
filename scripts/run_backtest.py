# Placeholder for backtesting script
from aerthos_quant.utils.loader import fetch_carbon_data
from aerthos_quant.strategies.signal_generator import generate_signals, generate_ml_signals
from aerthos_quant.visualization.plot_signals import plot_with_mplfinance
from aerthos_quant.backtest.core import backtest, run_batch_backtest

df = fetch_carbon_data()
df = generate_signals(df, indicators=True)
df = generate_ml_signals(df, indicators=False)
signals = ['signal_rsi', 'signal_macd', 'signal_bb', 'signal_vol', 'signal_ma', 'signal_ml']
for s in signals:
    plot_with_mplfinance(df, signal_type=s, save_path=f'./aerthos_quant/predictions/pics/{s}.png')

run_batch_backtest(df, signals, backtest_func=backtest)
df.to_csv('./aerthos_quant/predictions/carbon_data_with_signals.csv', index=False)

