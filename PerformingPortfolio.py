import yfinance as yf
import pandas as pd
from src.Allocation import simulate_portfolio
from src.Metrics import strategy_stats
import sgs
from datetime import datetime
from collections import OrderedDict




ini_date = '2022-09-12'
end_date = '2025-09-11'

# Defining weights
assets = ("IMAB11.SA", "SPXI11.SA")

dict_strategies = OrderedDict()

# ponto central
dict_strategies["80_20"] = {assets[0]: 0.80, assets[1]: 0.2}

# # expandindo simetricamente a partir de 50 em passos de 5
# for p in range(55, 101, 1):   # 55, 60, ..., 100
#     q = 100 - p               # 45, 40, ..., 0
#     # lado "IMAB mais pesado"
#     dict_strategies[f"{p}_{q}"] = {
#         assets[0]: p/100,
#         assets[1]: q/100
#     }
#     # lado espelhado "SPXI mais pesado"
#     dict_strategies[f"{q}_{p}"] = {
#         assets[0]: q/100,
#         assets[1]: p/100
#     }



# Getting CDI
df_cdi = sgs.dataframe(12, start= datetime.strptime(ini_date, "%Y-%m-%d").strftime("%d/%m/%Y"), end= datetime.strptime(end_date, "%Y-%m-%d").strftime("%d/%m/%Y"))
df_cdi[12] = df_cdi[12]/100 # deixando em porcentagem

# Getting assets prices
# Getting tickers from dict
lista_tickers = list(assets)
df_tickers = yf.download(lista_tickers,
                      # period="1y",
                      # interval="1d",
                      start = ini_date,
                      end = end_date)

portfolio_df = df_tickers['Close'].copy()

dict_results = {}
for k, v in dict_strategies.items():

    # Weights
    weights = v
    # Getting tickers from dict
    lista_tickers = list(v.keys())


    portfolio_df = df_tickers['Close'].copy()

    res = simulate_portfolio(
        prices=portfolio_df,                       # DataFrame de preços (colunas = tickers)
        weights=weights,
        initial_investment=100000,
        rebalance_freq="Q",
        when="first",
        transaction_cost_bps=5,               # 5 bps por lado
        allow_fractional=False,
        lot_size=1,
        # rf_daily=df_cdi[12]
    )

    # Saving results in dict
    dict_results[k] = res


# Analysing portfolios
dict_portfolio_valulation = {}
for k, v in dict_results.items():

    stats = strategy_stats(data=v.portfolio_value,
                           rf_daily=df_cdi[12])
    dict_portfolio_valulation[k] = {'annualized_return':stats['annualized_return'],
                                    'annualized_vol':stats['annualized_vol']}

df_return_and_vol = pd.DataFrame.from_dict(dict_portfolio_valulation, orient="index").reset_index()

print('a')