from aerthos_quant.strategies.signal_generator import backtest, Alpha_MA_Crossover
from aerthos_quant.api.supabase import fetch_table
import matplotlib.pyplot as plt

buy_signals, sell_signals, df = Alpha_MA_Crossover()
initial_capital = 100000
total_profit, status_df = backtest(buy_signals, sell_signals, df, initial_capital)

print("初始资金: ", initial_capital)
print("总收益: ", total_profit)
annualized_return = (1 + total_profit / initial_capital) ** (252 / len(status_df)) - 1
print(f"年化收益率: {annualized_return:.2%}")
import numpy as np

# 计算每日收益率
status_df['Daily_Return'] = status_df['Capital'].pct_change()

# 计算夏普比率
risk_free_rate = 0.04  # 假设的无风险利率
sharpe_ratio = (status_df['Daily_Return'].mean() - risk_free_rate / 252) / status_df['Daily_Return'].std() * np.sqrt(252)
print(f"夏普比率: {sharpe_ratio:.2f}")

# 计算最大回撤
cumulative_returns = (1 + status_df['Daily_Return']).cumprod()
drawdown = cumulative_returns / cumulative_returns.cummax() - 1
max_drawdown = drawdown.min()
print(f"最大回撤: {max_drawdown:.2%}")




plt.figure(figsize=(12, 6))
plt.plot(status_df.index, status_df['Capital'], label='Capital', color='blue')
plt.title('Capital Change')
plt.xlabel('Date')
plt.ylabel('Capital')
plt.axhline(y=initial_capital, color='r', linestyle='--', label='Initial Capital')
plt.legend()
plt.grid()
plt.savefig('capital_plot.png', bbox_inches='tight', pad_inches=0.1)
plt.close()
