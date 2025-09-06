import yfinance as yf
import pandas as pd
from src.Allocation import simulate_portfolio
from src.Metrics import strategy_stats
import sgs
from datetime import datetime




ini_date = '2025-05-16'
end_date = '2025-09-05'

# Defining weights
w1 = {'IMAB11.SA':0.8,
     'SPXI11.SA':0.2}

w2 = {'IMAB11.SA':0.4,
     'SPXI11.SA':0.6}

dict_strategies = {'80_20':w1,
                   '40_60':w2}


# Getting CDI
df_cdi = sgs.dataframe(12, start= datetime.strptime(ini_date, "%Y-%m-%d").strftime("%d/%m/%Y"), end= datetime.strptime(end_date, "%Y-%m-%d").strftime("%d/%m/%Y"))
df_cdi[12] = df_cdi[12]/100 # deixando em porcentagem

dict_results = {}
for k, v in dict_strategies.items():

    # Weights
    weights = v
    # Getting tickers from dict
    lista_tickers = list(v.keys())
    df_tickers = yf.download(lista_tickers,
                          # period="1y",
                          # interval="1d",
                          start = ini_date,
                          end = end_date)

    portfolio_df = df_tickers['Close'].copy()

    res = simulate_portfolio(
        prices=portfolio_df,                       # DataFrame de preços (colunas = tickers)
        weights=weights,
        initial_investment=10000,
        rebalance_freq="Q",
        when="first",
        transaction_cost_bps=5,               # 5 bps por lado
        allow_fractional=False,
        lot_size=1,
        # rf_daily=df_cdi[12]
    )

    # Saving results in dict
    dict_results[k] = res

# Analysing portfolio
stats = strategy_stats(data=dict_results['80_20'].portfolio_value,
                       rf_daily=df_cdi[12])



print('a')