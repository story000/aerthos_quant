def make_direction_label(df, n=5, threshold=0.0):
    df = df.copy()
    future_return = df['Price'].shift(-n) / df['Price'] - 1
    df[f'label_direction_{n}d'] = (future_return > threshold).astype(int)
    return df