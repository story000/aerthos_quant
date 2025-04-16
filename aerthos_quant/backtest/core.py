import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def backtest(df, signal_type='signal_rsi', initial_capital=100000):
    df = df.copy()
    df = df[df['Date'] >= '2021-01-01']
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    buy_signals = sorted([(date, 'BUY') for date in df[df[f"buy_{signal_type}"]].index.tolist()])
    sell_signals = sorted([(date, 'SELL') for date in df[df[f"sell_{signal_type}"]].index.tolist()])
    all_signals = sorted(buy_signals + sell_signals)

    total_profit, position, capital = 0.0, 0, initial_capital
    status, buy_price = [], None
    df[f'buy_{signal_type}'] = df[f'buy_{signal_type}'].shift(1)
    df[f'sell_{signal_type}'] = df[f'sell_{signal_type}'].shift(1)
    for date, signal in all_signals:
        price = df.at[date, 'Price']
        if signal == 'BUY' and position == 0:
            position = capital // price
            capital -= position * price
            buy_price = price
            status.append((date, 'BUY', position, capital))
        elif signal == 'SELL' and position > 0:
            capital += position * price
            total_profit += (price - buy_price) * position
            position = 0
            status.append((date, 'SELL', 0, capital))

    status_df = pd.DataFrame(status, columns=['Date', 'Action', 'Position', 'Current Liquidity'])
    status_df.set_index('Date', inplace=True)
    new_df = pd.concat([df['Price'], status_df.reindex(df.index)], axis=1)
    new_df.fillna(method='ffill', inplace=True)
    new_df.fillna({'Position': 0, 'Current Liquidity': initial_capital}, inplace=True)
    new_df['Capital'] = new_df['Current Liquidity'] + new_df['Position'] * new_df['Price']

    returns = new_df['Capital'].pct_change().dropna()
    max_drawdown = (new_df['Capital'].cummax() - new_df['Capital']) / new_df['Capital'].cummax()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)

    # Plot
    capital_normalized = new_df['Capital'] / initial_capital
    price_normalized = new_df['Price'] / new_df['Price'].iloc[0]

    plt.figure(figsize=(14, 6))
    plt.plot(new_df.index, capital_normalized, label='Capital', color='blue')
    plt.plot(new_df.index, price_normalized, label='Price', color='red')

    info_text = f"Max Drawdown: {max_drawdown.max():.2f}\nSharpe Ratio: {sharpe_ratio:.2f}"
    x_pos = new_df.index[int(len(new_df) * 0.15)]
    y_pos = capital_normalized.max() * 0.95  # ✅ 注意这里用了 normalized 的 y 坐标

    plt.text(x_pos, y_pos, info_text, fontsize=10, color='black',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
            verticalalignment='top')
    plt.title('Capital Over Time')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Capital')
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"aerthos_quant/predictions/capital_chart/{signal_type}.png")
    plt.close()
    return total_profit, new_df


def run_batch_backtest(df, signal_types, backtest_func):
    results = []
    for signal_type in signal_types:
        profit, _ = backtest_func(df.copy(), signal_type)
        results.append((signal_type, profit))
    return pd.DataFrame(results, columns=['Signal', 'Profit'])
