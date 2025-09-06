import numpy as np
import pandas as pd
import yfinance as yf

# Let's create an array that holds random portfolio weights
# Note that portfolio weights must add up to 1
import random



def generate_portfolio_weights(n):
    weights = []
    for i in range(n):
        weights.append(random.random())

    # let's make the sum of all weights add up to 1
    weights = weights / np.sum(weights)
    return weights



# Call the function (Run this cell multiple times to generate different outputs)
weights = generate_portfolio_weights(4)
print(weights)


# Function to scale stock prices based on their initial starting price
# The objective of this function is to set all prices to start at a value of 1
def price_scaling(raw_prices_df):
    scaled_prices_df = raw_prices_df.copy()
    for i in raw_prices_df.columns[0:]:
          scaled_prices_df[i] = raw_prices_df[i]/raw_prices_df[i][0]
    return scaled_prices_df


# Let's define the "weights" list similar to the slides
weights = [0.6, 0.4]


lista_tickers = ['BOVV11.SA', 'IMAB11.SA']
df_tickers = yf.download(lista_tickers,
                      period="1y", interval="1d")
close_price_df = df_tickers['Close'].copy()



# Scale stock prices using the "price_scaling" function that we defined earlier (make all stock values start at 1)
portfolio_df = close_price_df.copy()


# Assume that we have $1,000,000 that we would like to invest in one or more of the selected stocks
# Let's create a function that receives the following arguments:
# (1) Stocks closing prices
# (2) Random weights
# (3) Initial investment amount
# The function will return a DataFrame that contains the following:
# (1) Daily value (position) of each individual stock over the specified time period
# (2) Total daily value of the portfolio
# (3) Percentage daily return

def asset_allocation(df, weights, initial_investment):
    portfolio_df = df.copy()

    # Scale stock prices using the "price_scaling" function that we defined earlier (Make them all start at 1)
    scaled_df = price_scaling(df)

    for i, stock in enumerate(scaled_df.columns[0:]):
        portfolio_df[stock] = scaled_df[stock] * weights[i] * initial_investment

    # Sum up all values and place the result in a new column titled "portfolio value [$]"
    # Note that we excluded the date column from this calculation
    portfolio_df['Portfolio Value [$]'] = portfolio_df[portfolio_df != 'Date'].sum(axis=1, numeric_only=True)

    # Calculate the portfolio percentage daily return and replace NaNs with zeros
    portfolio_df['Portfolio Daily Return [%]'] = portfolio_df['Portfolio Value [$]'].pct_change(1) * 100
    portfolio_df.replace(np.nan, 0, inplace=True)

    return portfolio_df

import pandas as pd
import numpy as np

def simulate_portfolio(
    prices: pd.DataFrame,
    weights: dict,                    # agora só aceita dict {'PETR4.SA':0.4, ...}
    initial_investment: float = 10_000,
    freq: str | None = 'M',           # 'D', 'W', 'W-FRI', 'M', 'Q', 'A'... ou None (buy-and-hold)
    when: str = 'first',              # 'first' (primeiro pregão) ou 'last' (último pregão do período)
    transaction_cost_bps: float = 0.0 # custo de transação em bps
):
    """
    Simula a evolução de uma carteira a partir de preços, com rebalanceamento.
    - prices: DataFrame de preços (index datetime, colunas = tickers)
    - weights: dict {ticker: peso}, não precisa ter todas as colunas, faltantes recebem 0
    """

    assert isinstance(prices.index, pd.DatetimeIndex), "O índice de prices precisa ser DatetimeIndex"
    prices = prices.sort_index().copy()

    # Garantir correspondência entre colunas e dict
    w = pd.Series(0.0, index=prices.columns, dtype=float)  # começa com 0 para todos
    for k, v in weights.items():
        if k in w.index:
            w.loc[k] = float(v)
        else:
            print(f"Aviso: ticker '{k}' não está nas colunas do DataFrame e será ignorado.")

    if w.sum() == 0:
        raise ValueError("Soma dos pesos é 0. Nenhum ticker válido foi informado.")

    w = w / w.sum()  # normaliza para 100%

    # === Mesma lógica do rebalanceamento que você já tinha ===
    if freq is None:
        rebal_dates = [prices.index[0]]
    else:
        grp = prices.groupby(prices.index.to_period(freq))
        idx = grp.head(1).index if when == 'first' else grp.tail(1).index
        if prices.index[0] not in idx:
            idx = idx.union(pd.DatetimeIndex([prices.index[0]]))
        rebal_dates = pd.DatetimeIndex(sorted(idx.unique()))

    cols = list(prices.columns)
    values = pd.DataFrame(index=prices.index, columns=cols, dtype=float)
    weights_realized = pd.DataFrame(index=prices.index, columns=[f"w_{c}" for c in cols], dtype=float)

    p0 = prices.iloc[0]
    capital = float(initial_investment)
    target_val = capital * w
    shares = (target_val / p0.replace(0, np.nan)).fillna(0.0)

    cost_rate = transaction_cost_bps / 10_000.0

    for t in prices.index:
        pt = prices.loc[t]
        cur_vals = shares * pt
        total = cur_vals.sum()
        values.loc[t, cols] = cur_vals.values
        weights_realized.loc[t] = (cur_vals / (total if total != 0 else 1e-12)).values

        if t in rebal_dates and t != prices.index[0]:
            target_vals_pre_cost = total * w
            turnover_one_side = 0.5 * (target_vals_pre_cost - cur_vals).abs().sum()
            cost = turnover_one_side * cost_rate if cost_rate > 0 else 0.0
            total_after_cost = max(total - cost, 0.0)
            target_vals = total_after_cost * w
            shares = (target_vals / pt.replace(0, np.nan)).fillna(0.0)

    out = values.copy()
    out["Total"] = out.sum(axis=1)
    out["Return_Daily_%"] = out["Total"].pct_change().fillna(0.0) * 100.0
    out = out.join(weights_realized)

    return out


w = {
    "ABEV3.SA": 2.471/100,
    "ALOS3.SA": 0.524/100,
    "ASAI3.SA": 0.662/100,
    "AURE3.SA": 0.162/100,
    "AZZA3.SA": 0.219/100,
    "B3SA3.SA": 3.158/100,
    "BBAS3.SA": 2.847/100,
    "BBDC3.SA": 0.992/100,
    "BBDC4.SA": 4.024/100,
    "BBSE3.SA": 0.947/100,
    "BEEF3.SA": 0.122/100,
    "BPAC11.SA": 2.951/100,
    "BRAP4.SA": 0.192/100,
    "BRAV3.SA": 0.433/100,
    "BRFS3.SA": 0.589/100,
    "BRKM5.SA": 0.117/100,
    "CEAB3.SA": 0.109/100,
    "CMIG4.SA": 0.990/100,
    "CMIN3.SA": 0.401/100,
    "COGN3.SA": 0.248/100,
    "CPFE3.SA": 0.346/100,
    "CPLE6.SA": 0.941/100,
    "CSAN3.SA": 0.314/100,
    "CSNA3.SA": 0.259/100,
    "CURY3.SA": 0.216/100,
    "CVCB3.SA": 0.045/100,
    "CXSE3.SA": 0.396/100,
    "CYRE3.SA": 0.337/100,
    "DIRR3.SA": 0.236/100,
    "EGIE3.SA": 0.477/100,
    "ELET3.SA": 3.841/100,
    "ELET6.SA": 0.604/100,
    "EMBR3.SA": 2.617/100,
    "ENEV3.SA": 1.353/100,
    "ENGI11.SA": 0.744/100,
    "EQTL3.SA": 2.134/100,
    "FLRY3.SA": 0.312/100,
    "GGBR4.SA": 0.995/100,
    "GOAU4.SA": 0.276/100,
    "HAPV3.SA": 0.608/100,
    "HYPE3.SA": 0.335/100,
    "IGTI11.SA": 0.233/100,
    "IRBR3.SA": 0.183/100,
    "ISAE4.SA": 0.430/100,
    "ITSA4.SA": 3.070/100,
    "ITUB4.SA": 8.307/100,
    "KLBN11.SA": 0.680/100,
    "LREN3.SA": 0.762/100,
    "MGLU3.SA": 0.136/100,
    "MOTV3.SA": 0.667/100,
    "MRFG3.SA": 0.220/100,
    "MRVE3.SA": 0.133/100,
    "MULT3.SA": 0.409/100,
    "NATU3.SA": 0.354/100,
    "PCAR3.SA": 0.073/100,
    "PETR3.SA": 4.177/100,
    "PETR4.SA": 6.423/100,
    "POMO4.SA": 0.287/100,
    "PRIO3.SA": 1.353/100,
    "PSSA3.SA": 0.439/100,
    "RADL3.SA": 1.057/100,
    "RAIL3.SA": 0.828/100,
    "RAIZ4.SA": 0.067/100,
    "RDOR3.SA": 2.012/100,
    "RECV3.SA": 0.166/100,
    "RENT3.SA": 1.640/100,
    "SANB11.SA": 0.476/100,
    "SBSP3.SA": 3.420/100,
    "SLCE3.SA": 0.162/100,
    "SMFT3.SA": 0.488/100,
    "STBP3.SA": 0.273/100,
    "SUZB3.SA": 1.584/100,
    "TAEE11.SA": 0.357/100,
    "TIMS3.SA": 0.857/100,
    "TOTS3.SA": 1.076/100,
    "UGPA3.SA": 0.986/100,
    "USIM5.SA": 0.094/100,
    "VALE3.SA": 11.103/100,
    "VAMO3.SA": 0.101/100,
    "VBBR3.SA": 1.252/100,
    "VIVA3.SA": 0.167/100,
    "VIVT3.SA": 1.170/100,
    "WEGE3.SA": 2.620/100,
    "YDUQ3.SA": 0.161/100
}

lista_tickers = list(w.keys())
df_tickers = yf.download(lista_tickers,
                      # period="1y",
                      # interval="1d",
                      start = '2025-09-01',
                      end = '2025-09-05')
portfolio_df = df_tickers['Close'].copy()

portfolio = simulate_portfolio(portfolio_df, weights=w, initial_investment=100000, freq=None)

# teste_bh = simulate_portfolio(portfolio_df, weights, initial_investment=1000000, freq=None)
#
# teste_d = simulate_portfolio(portfolio_df, weights, 1000000, freq='D', when='first')


print('a')