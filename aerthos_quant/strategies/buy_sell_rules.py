import numpy as np

def generate_signal_columns(df):
    df['buy_signal_rsi'] = (df['rsi'] < 30) & (df['macd_diff'] < 0)
    df['sell_signal_rsi'] = (df['rsi'] > 70) & (df['macd_diff'] > 0)

    df['buy_signal_macd'] = (df['macd_diff'] > 0) & (df['macd_diff'].shift(1) < 0)
    df['sell_signal_macd'] = (df['macd_diff'] < 0) & (df['macd_diff'].shift(1) > 0)

    df['buy_signal_vol'] = (df['Vol'] > 1.6 * df['vol_ma5']) & (df['Vol'] > 1.6 * df['vol_ma5'].shift(1))
    df['sell_signal_vol'] = (df['Vol'] < 0.6 * df['vol_ma5']) & (df['Vol'] < 0.6 * df['vol_ma5'].shift(1))

    df['buy_signal_bb'] = (df['Price'] > df['bb_upper']) & (df['Price'].shift(1) < df['bb_upper'].shift(1))
    df['sell_signal_bb'] = (df['Price'] < df['bb_lower']) & (df['Price'].shift(1) > df['bb_lower'].shift(1))

    df['buy_signal_ma'] = (df['ma5'] > df['ma10']) & (df['ma5'].shift(1) < df['ma10'].shift(1))
    df['sell_signal_ma'] = (df['ma5'] < df['ma10']) & (df['ma5'].shift(1) > df['ma10'].shift(1))

    return df


def predict_direction(df, model, feature_cols, signal_type='ml'): 
    df = df.copy()
    df = df.dropna(subset=feature_cols)

    predictions = model.predict(df[feature_cols])
    buy_col = f'buy_signal_{signal_type}'
    sell_col = f'sell_signal_{signal_type}'
    df[buy_col] = (predictions == 1) & (np.roll(predictions, 1) == 0)
    df[sell_col] = (predictions == 0) & (np.roll(predictions, 1) == 1)

    print(f"✅ ML 信号列生成: {buy_col}, {sell_col}")
    print("✅ df columns:", df.columns.tolist())

    return df