import yfinance as yf
import pandas as pd
from src.Allocation import simulate_portfolio

# Baixar preços reais de um ativo
ativo = ['BOVA11.SA']
data_ini = '2025-01-01'
data_fim = '2025-09-01'
df = yf.download(ativo, start=data_ini, end=data_fim)
prices = df['Close'].dropna(how='all')

# Pesos: 100% no ativo
pesos = {a: 1.0 for a in prices.columns}

# Sem custo de financiamento
financing_rate = 0.0

# Teste com leverage 1x
res1 = simulate_portfolio(
    prices=prices,
    weights=pesos,
    initial_investment=100000,
    rebalance_freq='Y',
    leverage=1.0,
    financing_rate_daily=financing_rate,
    allow_fractional=True,
    lot_size=1
)

# Teste com leverage 2x
res2 = simulate_portfolio(
    prices=prices,
    weights=pesos,
    initial_investment=100000,
    rebalance_freq='Y',
    leverage=2.0,
    financing_rate_daily=financing_rate,
    allow_fractional=True,
    lot_size=1
)

ret1 = (res1.portfolio_value.iloc[-1] / res1.portfolio_value.iloc[0]) - 1
ret2 = (res2.portfolio_value.iloc[-1] / res2.portfolio_value.iloc[0]) - 1
print(f'Retorno percentual final 1x: {ret1:.2%}')
print(f'Retorno percentual final 2x: {ret2:.2%}')
print('Valor final 1x:', res1.portfolio_value.iloc[-1])
print('Valor final 2x:', res2.portfolio_value.iloc[-1])
print('Maior alavancagem 1x:', ((res1.positions * prices).sum(axis=1) / res1.portfolio_value).max())
print('Maior alavancagem 2x:', ((res2.positions * prices).sum(axis=1) / res2.portfolio_value).max())