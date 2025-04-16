import mplfinance as mpf
import numpy as np
import matplotlib.pyplot as plt
def plot_with_mplfinance(df, signal_type='signal_rsi', save_path=None):
    df = df.copy()
    df = df[df['Date'] > '2021-01-01']
    df.set_index('Date', inplace=True)
    df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Price': 'close'}, inplace=True)

    y_arrow_buy = np.where(df[f"buy_{signal_type}"], df['low'] - 2, np.nan)
    y_arrow_sell = np.where(df[f"sell_{signal_type}"], df['high'] + 2, np.nan)

    apds = [
        mpf.make_addplot(y_arrow_buy, type='scatter', markersize=80, marker='^', color='green'),
        mpf.make_addplot(y_arrow_sell, type='scatter', markersize=80, marker='v', color='red')
    ]

    mpf.plot(
        df,
        type='candle',
        style='charles',
        addplot=apds,
        title=f"Signal Chart for {signal_type.upper()}",
        ylabel='Price',
        figsize=(14, 6),
        warn_too_much_data=len(df) + 1000,
        savefig=save_path
    )
