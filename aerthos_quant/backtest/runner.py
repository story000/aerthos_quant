def run_batch_backtest(df, signal_types, backtest_func):
    for signal_type in signal_types:
        profit, _ = backtest_func(df.copy(), signal_type)
        print(f"{signal_type} 总收益: {profit:.2f}")
