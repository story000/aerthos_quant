import ta 
import pandas as pd
import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from aerthos_quant.api.supabase import fetch_table

def generate_signals(symbols=None):
    """
    生成交易信号的函数。
    """
    signals = {
        "AAPL": "BUY",
        "GOOGL": "SELL",
        "AMZN": "HOLD"
    }
    return signals[symbols] if symbols else signals

def Alpha_MA_Crossover(df=None):
    """
    获取碳价格数据，并计算20日和50日移动平均线，生成交易信号。
    
    参数:
        df: 包含碳价格数据的DataFrame
        
    返回:
        buy_signals: 买入信号的日期列表
        sell_signals: 卖出信号的日期列表
        df: 包含价格数据的DataFrame
    """
    # 获取价格数据
    if df is None:
        df = fetch_table('carbon') 
        df = pd.DataFrame(sorted(df, key=lambda x: x['Date']))
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

    # 计算20日和50日移动平均线
    df['20_Day_MA'] = df['Price'].rolling(window=20).mean()
    df['50_Day_MA'] = df['Price'].rolling(window=50).mean()

    # 生成信号
    df['Signal'] = 0
    df['Signal'][20:] = np.where(df['20_Day_MA'][20:] > df['50_Day_MA'][20:], 1, 0)  # 1表示买入信号，0表示卖出信号
    df['Crossover'] = df['Signal'].diff()  # 计算交叉点

    buy_signals = df[df['Crossover'] == 1].index.tolist()  # 买入信号
    sell_signals = df[df['Crossover'] == -1].index.tolist()  # 卖出信号

    return buy_signals, sell_signals, df

def Signal_Generator(df=None):
    """
    生成买入信号的函数。
    """
    if df is None:
        df = fetch_table('carbon')
        df = pd.DataFrame(sorted(df, key=lambda x: x['Date']))
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=False)

    # RSI (默认14日)
    df['rsi'] = ta.momentum.RSIIndicator(close=df['Price']).rsi()

    # MACD
    macd = ta.trend.MACD(close=df['Price'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()  

    # 布林带
    bb = ta.volatility.BollingerBands(close=df['Price'])
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_upper'] = bb.bollinger_hband()

    # 均线（趋势判断）
    df['ma5'] = df['Price'].rolling(window=5).mean()
    df['ma10'] = df['Price'].rolling(window=10).mean()

    # 成交量均值（判断是否放量）
    df['vol_ma5'] = df['Vol'].rolling(window=5).mean()
    # ========== 买入信号 ==========
    df['buy_signal_rsi'] = (df['rsi'] < 30) & (df['macd_diff'] < 0)
    df['buy_signal_macd'] = (df['macd_diff'] > 0) & (df['macd_diff'].shift(1) < 0)
    df['buy_signal_vol'] = (df['Vol'] > 1.6 * df['vol_ma5']) & (df['Vol'] > 1.6 * df['vol_ma5'].shift(1))
    df['buy_signal_bb'] = (df['Price'] > df['bb_upper']) & (df['Price'].shift(1) < df['bb_upper'].shift(1))
    df['buy_signal_ma'] = (df['ma5'] > df['ma10']) & (df['ma5'].shift(1) < df['ma10'].shift(1))
    
    df['sell_signal_rsi'] = (df['rsi'] > 70) & (df['macd_diff'] > 0)
    df['sell_signal_macd'] = (df['macd_diff'] < 0) & (df['macd_diff'].shift(1) > 0)
    df['sell_signal_vol'] = (df['Vol'] < 0.6 * df['vol_ma5']) & (df['Vol'] < 0.6 * df['vol_ma5'].shift(1))
    df['sell_signal_bb'] = (df['Price'] < df['bb_lower']) & (df['Price'].shift(1) > df['bb_lower'].shift(1))
    df['sell_signal_ma'] = (df['ma5'] < df['ma10']) & (df['ma5'].shift(1) > df['ma10'].shift(1))
    
    
    return df
    
        
def backtest(df, signal_type='signal_rsi', initial_capital=100000):
    """
    回测Alpha_MA_Crossover的买入和卖出信号，计算收益。
    
    参数:
        buy_signals: 买入信号的日期列表
        sell_signals: 卖出信号的日期列表
        df: 包含价格数据的DataFrame
        initial_capital: 初始资金
    返回:
        总收益
    """
    
    total_profit = 0.0
    position = 0  # 当前持仓状态，0表示未持仓，1表示持仓
    current_capital = initial_capital  # 当前资金
    current_position = 0  # 当前持仓数量
    df = df[df['Date'] >= '2021-01-01'] 

    buy_signals = sorted([(date, 'BUY') for date in df[df["buy_"+signal_type]]['Date'].tolist()])
    sell_signals = sorted([(date, 'SELL') for date in df[df["sell_"+signal_type]]['Date'].tolist()])
    all_signals = sorted(buy_signals + sell_signals)
    
    df['Date'] = pd.to_datetime(df['Date'])  # 转换成 datetime 类型（如果还没）
    df.set_index('Date', inplace=True)
    
    
    
    status = []
    for date, signal in all_signals:
        if signal == 'BUY' and position == 0:  # 如果当前未持仓，则买入
            print(df.index[:5])         # 看看是不是 DatetimeIndex
            print(type(date))  
            buy_price = df.loc[date,'Price']
            position = current_capital // buy_price 
            current_capital -= position * buy_price
            print(f"买入日期: {date}, 买入价格: {buy_price}, 持仓数量: {position}, 当前资金: {current_capital}")
            status.append((date, 'BUY', buy_price, position, current_capital))
        elif signal == 'SELL' and position > 0:  # 如果当前持仓，则卖出
            sell_price = df.loc[date,'Price']
            profit = sell_price - buy_price  # 计算收益
            total_profit += profit * position
            current_capital += position * sell_price
            position = 0  # 更新持仓状态为未持仓
            print(f"卖出日期: {date}, 卖出价格: {sell_price}, 收益: {profit}, 当前资金: {current_capital}")
            status.append((date, 'SELL', sell_price, position, current_capital))
    
    

    status_df = pd.DataFrame(status, columns=['Date', 'Action', 'Price', 'Position', 'Current Liquidity'])  # 将status转换为DataFrame
    status_df.drop(columns=['Price'], inplace=True)
    status_df.set_index('Date', inplace=True)  # 设置日期为索引
    new_df = pd.concat([df['Price'], status_df.reindex(df.index)], axis=1)
    new_df.fillna(method='ffill', inplace=True)  # 根据前面的数填充position
    new_df.fillna({'Position':0, 'Current Liquidity':initial_capital}, inplace=True)  # 根据前面的数填充position
    new_df['Capital'] = new_df['Current Liquidity'] + new_df['Position'] * new_df['Price']

    returns = new_df['Capital'].pct_change().dropna()  # 计算每日收益率
    max_drawdown = (new_df['Capital'].cummax() - new_df['Capital']).max()  # 计算最大回撤
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # 计算Sharpe Ratio，假设每年252个交易日

    print(f"最大回撤: {max_drawdown:.2f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    plt.figure(figsize=(14, 6))
    plt.plot(new_df.index, new_df['Capital'], label='Capital', color='blue')
    plt.title('Capital Over Time')
    plt.xlabel('Date')
    plt.ylabel('Capital')

    # ✅ 计算右上角相对位置放置文字
    x_pos = new_df.index[int(len(new_df) * 0.15)]  # 取最后15%时间点
    y_pos = new_df['Capital'].max() * 0.95         # 取Y轴的95%位置

    # ✅ 绘制文字：无箭头，居右对齐
    info_text = f"Max Drawdown: {max_drawdown:.2f}\nSharpe Ratio: {sharpe_ratio:.2f}"
    plt.text(x_pos, y_pos, info_text,
            fontsize=10, color='black',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
            verticalalignment='top')

    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"aerthos_quant/predictions/capital_chart/{signal_type}.png")
    plt.close()
    return total_profit, new_df


def plot_with_mplfinance(df, signal_type='signal_rsi'):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df[df.index >= '2021-01-01']  # 过滤出2021年后的数据

    # 重命名列为 OHLC 标准格式
    df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Price': 'close'}, inplace=True)

    # 构造一个 y 序列：买入信号点为低点 -2，卖出信号点为高点 +2，其余为 NaN
    y_arrow_buy = np.where(df["buy_"+signal_type], df['low'] - 2, np.nan)
    y_arrow_sell = np.where(df["sell_"+signal_type], df['high'] + 2, np.nan)
    y_arrow = np.where(~np.isnan(y_arrow_buy), y_arrow_buy, y_arrow_sell)

    apds = [
        mpf.make_addplot(
            y_arrow_buy,
            type='scatter',
            markersize=80,
            marker='^',
            color='green'
        ),
        mpf.make_addplot(
            y_arrow_sell,
            type='scatter',
            markersize=80,
            marker='v',
            color='red'
        )
    ]

    mpf.plot(
        df,
        type='candle',
        style='charles',
        addplot=apds,
        title=f"Signal Chart for {signal_type.upper()}",
        ylabel='Price',
        figsize=(14, 6),
        warn_too_much_data=len(df) + 1000,  # 关闭 warning
        savefig="aerthos_quant/predictions/pics/"+signal_type+".png"
    )
    
plot_with_mplfinance(Signal_Generator(), signal_type='signal_rsi')
plot_with_mplfinance(Signal_Generator(), signal_type='signal_macd')
plot_with_mplfinance(Signal_Generator(), signal_type='signal_vol')
plot_with_mplfinance(Signal_Generator(), signal_type='signal_bb')
plot_with_mplfinance(Signal_Generator(), signal_type='signal_ma')
total_profit, new_df = backtest(Signal_Generator(), signal_type='signal_rsi')
total_profit, new_df = backtest(Signal_Generator(), signal_type='signal_macd')
total_profit, new_df = backtest(Signal_Generator(), signal_type='signal_vol')
total_profit, new_df = backtest(Signal_Generator(), signal_type='signal_bb')
total_profit, new_df = backtest(Signal_Generator(), signal_type='signal_ma')

