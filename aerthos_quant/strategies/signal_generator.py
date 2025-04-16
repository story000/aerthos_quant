from .indicator_engine import add_indicators
from .buy_sell_rules import generate_signal_columns

def generate_signals(df):
    df = add_indicators(df)
    df = generate_signal_columns(df)
    return df
