import yfinance as yf
import pandas as pd
from src.Allocation import simulate_portfolio





# Defining weights
w = {'IMAB11.SA':0.8,
     'SPXI11.SA':0.2}

# Getting tickers from dict
lista_tickers = list(w.keys())
df_tickers = yf.download(lista_tickers,
                      # period="1y",
                      # interval="1d",
                      start = '2019-05-21',
                      end = '2025-09-05')

portfolio_df = df_tickers['Close'].copy()

res = simulate_portfolio(
    prices=portfolio_df,                       # DataFrame de preços (colunas = tickers)
    weights=w,
    initial_investment=10000,
    rebalance_freq="Q",
    when="first",
    transaction_cost_bps=5,               # 5 bps por lado
    allow_fractional=False,
    lot_size=1
)

print('a')