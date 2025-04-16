import ta

def add_indicators(df):

    df['rsi'] = ta.momentum.RSIIndicator(close=df['Price']).rsi()
    macd = ta.trend.MACD(close=df['Price'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(close=df['Price'])
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_upper'] = bb.bollinger_hband()
    df['ma5'] = df['Price'].rolling(window=5).mean()
    df['ma10'] = df['Price'].rolling(window=10).mean()
    df['vol_ma5'] = df['Vol'].rolling(window=5).mean()
    return df