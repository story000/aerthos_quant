
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