from sklearn.model_selection import train_test_split

def prepare_ml_data(df, n=5, feature_cols=None):
    df = df.copy()
    if feature_cols is None:
        feature_cols = ['rsi', 'macd_diff', 'macd_signal', 'bb_upper', 'bb_lower', 'ma5', 'ma10', 'Vol', 'vol_ma5']
    label_col = f'label_direction_{n}d'

    df = df.dropna(subset=feature_cols + [label_col])  # 丢弃缺失值样本
    X = df[feature_cols]
    y = df[label_col]

    return train_test_split(X, y, test_size=0.2, shuffle=False)  # 时间序列不打乱