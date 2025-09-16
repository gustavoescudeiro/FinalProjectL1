import yfinance as yf
import pandas as pd
from src.Allocation import simulate_portfolio
from src.Metrics import strategy_stats
import sgs
from datetime import datetime

# Parâmetros

# Parâmetros de simulação
ini_date_estrategia = '2024-01-31'  # data de início da estratégia (primeiro rebalanceamento)
lookback = 60  # dias para cálculo do momentum
ini_date_precos = (pd.to_datetime(ini_date_estrategia) - pd.Timedelta(days=lookback*2)).strftime('%Y-%m-%d')
end_date = '2025-09-01'
ativos_universo = [
    'IMAB11.SA', 'SPXI11.SA', 'BOVA11.SA', 'IVVB11.SA', 'HASH11.SA', 'SMAL11.SA'
]

# Baixa preços com antecedência suficiente para o cálculo do momentum
df_tickers = yf.download(ativos_universo, start=ini_date_precos, end=end_date)
precos = df_tickers['Close'].dropna(how='all')

# Calcula momentum (retorno dos últimos 60 dias)
momentum = precos.pct_change(lookback)

# Cria calendário de rebalanceamento mensal a partir da data de início da estratégia
calendario = precos.loc[ini_date_estrategia:].resample('M').last().index.intersection(precos.index)


# Para cada data de rebalanceamento, seleciona os 2 ativos com maior momentum
pesos_dict = {}
for data in calendario:
    if data not in momentum.index:
        continue
    mom = momentum.loc[data].dropna()
    top2 = mom.nlargest(2).index.tolist()
    if len(top2) < 2:
        continue
    # Aloca 50% em cada um dos dois ativos
    pesos = {a: 0.5 if a in top2 else 0.0 for a in precos.columns}
    pesos_dict[data] = pesos



# Monta DataFrame de pesos apenas a partir da data de início da estratégia
pesos_df = pd.DataFrame.from_dict(pesos_dict, orient='index').reindex(precos.index).fillna(0.0)

# Getting CDI
df_cdi = sgs.dataframe(12, start= datetime.strptime(ini_date_estrategia, "%Y-%m-%d").strftime("%d/%m/%Y"), end= datetime.strptime(end_date, "%Y-%m-%d").strftime("%d/%m/%Y"))
df_cdi[12] = df_cdi[12]/100 # deixando em porcentagem


# Alinha preços e CDI ao período da estratégia
precos_sim = precos.loc[ini_date_estrategia:]
pesos_df_sim = pesos_df.loc[ini_date_estrategia:]
rf_cdi_sim = df_cdi[12].reindex(precos_sim.index).fillna(0.0)

# Simula portfólio
res = simulate_portfolio(
    prices=precos_sim,
    weights=pesos_df_sim,
    initial_investment=100000,
    rebalance_freq=None,
    transaction_cost_bps=5,
    allow_fractional=False,
    lot_size=1,
    rf_daily=rf_cdi_sim
)

# Analisa resultado
stats = strategy_stats(data=res.portfolio_value, rf_daily=df_cdi[12])
print('Retorno anualizado:', stats['annualized_return'])
print('Volatilidade anualizada:', stats['annualized_vol'])
print('Valor final da carteira:', res.portfolio_value.iloc[-1])
