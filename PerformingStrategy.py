import yfinance as yf
import pandas as pd
from strategies.selection import universe_all, universe_top_price, universe_alternate
from strategies.weights import weights_equal, weights_vol, weights_momentum, weights_markowitz
from strategies.strategy import Strategy
from backtest.runner import run_strategy

# --- Parâmetros ---
ini_date = '2024-01-01'
end_date = '2024-09-01'
initial_investment = 100000
rebalance_freq = 'M'  # mensal
transaction_cost_bps = 5
lot_size = 1
allow_fractional = False

# --- Baixa todos os preços necessários ---
todos_ativos = ["WEGE3.SA", "VALE3.SA", "PETR4.SA", "ITUB4.SA"]
df_tickers = yf.download(todos_ativos, start=ini_date, end=end_date)
precos = df_tickers['Close'].dropna()

# --- Define datas de rebalanceamento ---
calendario = precos.resample(rebalance_freq).first().index.intersection(precos.index)
calendario = calendario.insert(0, precos.index[0])  # garante rebalance no início
calendario = calendario.sort_values()

# --- Exemplos de estratégias ---
strategy1 = Strategy(universe_top_price, weights_equal)
strategy2 = Strategy(universe_all, weights_vol)
strategy3 = Strategy(universe_alternate, weights_momentum)
strategy4 = Strategy(universe_all, weights_markowitz)

res1 = run_strategy(strategy1, precos, calendario, start_date='2024-03-01', weights_window=None)
res2 = run_strategy(strategy2, precos, calendario, start_date='2024-03-01', weights_window=21)
res3 = run_strategy(strategy3, precos, calendario, start_date='2024-03-01', weights_window=21)
res4 = run_strategy(strategy4, precos, calendario, start_date='2024-03-01', weights_window=42)

print('Top Price + Equal Weighted:')
for k, v in res1.items():
    print(f'--- {k} ---')
    print(v)
print('All + Vol Weighted:')
for k, v in res2.items():
    print(f'--- {k} ---')
    print(v)
print('Alternate + Momentum Weighted:')
for k, v in res3.items():
    print(f'--- {k} ---')
    print(v)
print('All + Markowitz Weighted:')
for k, v in res4.items():
    print(f'--- {k} ---')
    print(v)

print('q')