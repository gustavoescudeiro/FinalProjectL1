import yfinance as yf
import pandas as pd
import numpy as np
from src.Allocation import simulate_portfolio
from sklearn.linear_model import LinearRegression

# Universe: 20 ações do Ibovespa (sem BOVA11) + BOVA11 para comparação
ativos = [
    'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'B3SA3.SA',
    'ABEV3.SA', 'BBAS3.SA', 'BBDC4.SA', 'WEGE3.SA', 'RENT3.SA',
    'MGLU3.SA', 'LREN3.SA', 'SUZB3.SA', 'GGBR4.SA', 'PRIO3.SA',
    'CSNA3.SA', 'HAPV3.SA', 'ELET3.SA', 'BRFS3.SA', 'RAIL3.SA', 'KLBN11.SA',
    'BOVA11.SA'  # só para pegar o retorno do índice
]
data_ini = '2024-01-01'
data_fim = '2025-09-01'
df = yf.download(ativos, start=data_ini, end=data_fim)['Close']
prices = df.dropna(axis=1, thresh=int(0.9*len(df)))
ativos_validos = [a for a in prices.columns if a != 'BOVA11.SA']

# Calcula retornos diários
rets = prices.pct_change().dropna()
ret_mkt = rets['BOVA11.SA']
rets = rets.drop(columns=['BOVA11.SA'])

# Calcula beta de cada ativo (regressão simples)
betas = {}
for a in rets.columns:
    X = ret_mkt.values.reshape(-1,1)
    y = rets[a].values
    reg = LinearRegression().fit(X, y)
    betas[a] = reg.coef_[0]
betas = pd.Series(betas)

# Seleciona 5 low beta e 5 high beta
low_beta = betas.nsmallest(5).index.tolist()
high_beta = betas.nlargest(5).index.tolist()

# Calcula pesos para portfólio neutro em beta
beta_L = betas[low_beta].mean()
beta_H = betas[high_beta].mean()
w_L = abs(beta_H) / (abs(beta_L) + abs(beta_H))
w_H = abs(beta_L) / (abs(beta_L) + abs(beta_H))

pesos = {a: w_L/len(low_beta) for a in low_beta}
pesos.update({a: -w_H/len(high_beta) for a in high_beta})

# Adiciona zeros para os demais ativos
for a in ativos_validos:
    if a not in pesos:
        pesos[a] = 0.0

# Simula portfólio BAB
res = simulate_portfolio(
    prices=prices[ativos_validos],
    weights=pesos,
    initial_investment=100000,
    rebalance_freq='D',
    leverage=1.0,
    financing_rate_daily=0.0,
    allow_fractional=True,
    lot_size=1
)

# Calcula retornos do portfólio
ret_port = res.portfolio_value.pct_change().dropna()

# Regressão do retorno do portfólio contra o mercado
X = ret_mkt.loc[ret_port.index].values.reshape(-1,1)
y = ret_port.values
reg = LinearRegression().fit(X, y)
print(f'Beta do portfólio BAB: {reg.coef_[0]:.4f}')
print(f'Intercepto: {reg.intercept_:.4%}')
print(f'Retorno anualizado BAB: {(1+ret_port.mean())**252-1:.2%}')
print(f'Valor final BAB: {res.portfolio_value.iloc[-1]:.2f}')
print('Pesos low beta:', {a: pesos[a] for a in low_beta})
print('Pesos high beta:', {a: pesos[a] for a in high_beta})

# Retorno acumulado do BOVA11 para comparação
bova = prices['BOVA11.SA']
ret_bova = bova.pct_change().dropna()
ret_acum_bova = (bova.iloc[-1] / bova.iloc[0]) - 1
print(f'Retorno acumulado BOVA11: {ret_acum_bova:.2%}')
print(f'Retorno anualizado BOVA11: {(1+ret_bova.mean())**252-1:.2%}')

