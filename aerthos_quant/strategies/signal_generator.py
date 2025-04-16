from .feature_engine import add_indicators
from .buy_sell_rules import generate_signal_columns, predict_direction
from ..ml.preprocessing import prepare_ml_data
from ..ml.training import train_direction_classifier
from ..ml.labeling import make_direction_label

def generate_signals(df, indicators=True):
    if indicators:
        df = add_indicators(df)
    df = generate_signal_columns(df)
    return df

def generate_ml_signals(df, indicators=True):
    if indicators:
        df = add_indicators(df)
    df= make_direction_label(df, n=5, threshold=0.01)
    feature_cols = ['rsi', 'macd_diff', 'macd_signal', 'bb_upper', 'bb_lower', 'ma5', 'ma10', 'Vol', 'vol_ma5']
    X_train, X_val, y_train, y_val = prepare_ml_data(df, feature_cols=feature_cols)
    model = train_direction_classifier(X_train, y_train, X_val, y_val)
    df = predict_direction(df, model, feature_cols, signal_type='ml')
    return df

