import yfinance as yf
import pandas as pd
import numpy as np
from src.Allocation import simulate_portfolio
from src.Metrics import strategy_stats
import sgs
from datetime import datetime

# Parâmetros
ini_date = '2015-01-01'
end_date = '2025-01-01'
ativos_universo = [
    'ABEV3.SA','ALOS3.SA','ASAI3.SA','AURE3.SA','AZZA3.SA','B3SA3.SA','BBAS3.SA','BBDC3.SA','BBDC4.SA','BBSE3.SA','BEEF3.SA','BPAC11.SA','BRAP4.SA','BRAV3.SA','BRFS3.SA','BRKM5.SA','CEAB3.SA','CMIG4.SA','CMIN3.SA','COGN3.SA','CPFE3.SA','CPLE6.SA','CSAN3.SA','CSNA3.SA','CURY3.SA','CVCB3.SA','CXSE3.SA','CYRE3.SA','DIRR3.SA','EGIE3.SA','ELET3.SA','ELET6.SA','EMBR3.SA','ENEV3.SA','ENGI11.SA','EQTL3.SA','FLRY3.SA','GGBR4.SA','GOAU4.SA','HAPV3.SA','HYPE3.SA','IGTI11.SA','IRBR3.SA','ISAE4.SA','ITSA4.SA','ITUB4.SA','KLBN11.SA','LREN3.SA','MGLU3.SA','MOTV3.SA','MRFG3.SA','MRVE3.SA','MULT3.SA','NATU3.SA','PETR3.SA','PETR4.SA','POMO4.SA','PRIO3.SA','PSSA3.SA','RADL3.SA','RAIL3.SA','RAIZ4.SA','RDOR3.SA','RECV3.SA','RENT3.SA','SANB11.SA','SBSP3.SA','SLCE3.SA','SMFT3.SA','STBP3.SA','SUZB3.SA','TAEE11.SA','TIMS3.SA','TOTS3.SA','UGPA3.SA','USIM5.SA','VALE3.SA','VAMO3.SA','VBBR3.SA','VIVA3.SA','VIVT3.SA','WEGE3.SA','YDUQ3.SA']
ativos_universo = [
    'ABEV3.SA','ALOS3.SA','AMER3.SA','ASAI3.SA','AURE3.SA','AZUL4.SA','AZZA3.SA','B3SA3.SA','BBAS3.SA','BBDC3.SA','BBDC4.SA','BBSE3.SA','BEEF3.SA','BPAC11.SA','BRAP4.SA','BRFS3.SA','BRKM5.SA','BRML3.SA','CASH3.SA','CBAV3.SA','CCRO3.SA','CEAB3.SA','CESP6.SA','CMIG4.SA','CMIN3.SA','COGN3.SA','CPFE3.SA','CPLE6.SA','CRFB3.SA','CSAN3.SA','CSNA3.SA','CVCB3.SA','CYRE3.SA','DXCO3.SA','ECOR3.SA','EGIE3.SA','ELET3.SA','ELET6.SA','EMBR3.SA','ENEV3.SA','ENGI11.SA','EQTL3.SA','EZTC3.SA','FLRY3.SA','GGBR4.SA','GOAU4.SA','GOLL4.SA','HAPV3.SA','HYPE3.SA','IGTI11.SA','IRBR3.SA','ITSA4.SA','ITUB4.SA','JBSS3.SA','JHSF3.SA','KLBN11.SA','LREN3.SA','LWSA3.SA','MGLU3.SA','MRFG3.SA','MRVE3.SA','MULT3.SA','NATU3.SA','PETR3.SA','PETR4.SA','PETZ3.SA','POMO4.SA','PRIO3.SA','PSSA3.SA','QUAL3.SA','RADL3.SA','RAIL3.SA','RAIZ4.SA','RDOR3.SA','RECV3.SA','RENT3.SA','RRRP3.SA','SANB11.SA','SBSP3.SA','SLCE3.SA','SMFT3.SA','SOMA3.SA','STBP3.SA','SUZB3.SA','TAEE11.SA','TIMS3.SA','TOTS3.SA','UGPA3.SA','USIM5.SA','VALE3.SA','VAMO3.SA','VBBR3.SA','VIVA3.SA','VIVT3.SA','WEGE3.SA','YDUQ3.SA']
benchmark = 'BOVA11.SA'

# Baixa preços
df_tickers = yf.download(ativos_universo + [benchmark], start=ini_date, end=end_date)
precos = df_tickers['Close'].dropna(how='all')

# Simula betas (exemplo: rolling 60 dias contra BOVA11)
retornos = precos.pct_change()
ret_bench = retornos[benchmark]
retornos = retornos.loc[ret_bench.index]
betas = {}
window = 220
for ativo in ativos_universo:
    if ativo == benchmark:
        continue
    serie_ativo = retornos[ativo].reindex(ret_bench.index).dropna()
    idx_valid = serie_ativo.index.intersection(ret_bench.index)
    if len(idx_valid) == 0:
        betas[ativo] = pd.Series(index=ret_bench.index, dtype=float)
        continue
    beta_serie = (
        serie_ativo.loc[idx_valid].rolling(window).cov(ret_bench.loc[idx_valid]) /
        ret_bench.loc[idx_valid].rolling(window).var()
    )
    betas[ativo] = beta_serie.reindex(ret_bench.index)
betas_df = pd.DataFrame(betas)
print(betas_df)

# Cria DataFrame de pesos: long nos 2 menores beta, short nos 2 maiores
pesos_dict = {}
# Parâmetro de alavancagem BAB
alavancagem_bab = 1.0  # 1.0 = 100% comprado e 100% vendido
# Rebalance mensal: datas de rebalance são o último pregão de cada mês
rebalance_dates = precos.resample('M').last().index.intersection(precos.index)
rebalance_validas = []
ativos_low_dict = {}
ativos_high_dict = {}
for data in rebalance_dates:
    if data not in betas_df.index:
        continue
    betas_hoje = betas_df.loc[data].dropna().drop(benchmark, errors='ignore')
    quantil = 0.3
    n_total = len(betas_hoje)
    n_q = max(1, int(n_total * quantil))
    if n_total < 2 * n_q:
        continue
    # Considera apenas betas positivos
    betas_positivos = betas_hoje[betas_hoje > 0]
    if len(betas_positivos) < 2:
        continue
    betas_ordenados = betas_positivos.sort_values()
    n_q = max(1, int(len(betas_ordenados) * quantil))
    if len(betas_ordenados) < 2 * n_q:
        continue
    ativos_low = betas_ordenados.index[:n_q].tolist()
    ativos_high = betas_ordenados.index[-n_q:].tolist()
    beta_low = betas_ordenados.iloc[:n_q].values
    beta_high = betas_ordenados.iloc[-n_q:].values
    pesos = {a: 0.0 for a in precos.columns if a != benchmark}
    nL = len(beta_low)
    nH = len(beta_high)
    if nL == 0 or nH == 0:
        for a in ativos_low:
            pesos[a] = 1.0 / max(1, nL)
        for a in ativos_high:
            pesos[a] = -1.0 / max(1, nH)
        pesos_dict[data] = pesos
        rebalance_validas.append(data)
        continue
    beta_long = np.mean(beta_low)
    beta_short = np.mean(beta_high)
    # Frazzini & Pedersen (2014): w_long = 1/beta_long, w_short = -1/beta_short
    w_long = 1.0 / beta_long if abs(beta_long) > 1e-8 else 1.0
    w_short = -1.0 / beta_short if abs(beta_short) > 1e-8 else -1.0
    # Distribui igualmente entre os ativos de cada grupo
    for a in ativos_low:
        pesos[a] = w_long / nL
    for a in ativos_high:
        pesos[a] = w_short / nH
    pesos_dict[data] = pesos
    rebalance_validas.append(data)
if len(rebalance_validas) == 0:
    raise ValueError('Não há datas mensais com betas válidos suficientes para rebalance. Tente aumentar o período ou reduzir o window.')
pesos_df = pd.DataFrame.from_dict(pesos_dict, orient='index').reindex(rebalance_validas).fillna(0.0)
# Normaliza pesos para soma dos absolutos = 1 em cada data
pesos_df = pesos_df.div(pesos_df.abs().sum(axis=1), axis=0)
# DataFrames de validação: pesos dos ativos escolhidos em cada decil
pesos_low_df = pesos_df.apply(lambda row: row[ativos_low_dict[row.name]] if row.name in ativos_low_dict else pd.Series(dtype=float), axis=1)
pesos_high_df = pesos_df.apply(lambda row: row[ativos_high_dict[row.name]] if row.name in ativos_high_dict else pd.Series(dtype=float), axis=1)
print('Pesos dos ativos do menor decil (comprados) por data:')
print(pesos_low_df)
print('Pesos dos ativos do maior decil (vendidos) por data:')
print(pesos_high_df)

# DataFrame de checagem: beta ponderado da carteira por data
beta_check = []
for data in pesos_df.index:
    if data in betas_df.index:
        betas_hoje = betas_df.loc[data]
        pesos_hoje = pesos_df.loc[data]
        beta_carteira = (betas_hoje * pesos_hoje).sum()
        beta_check.append({'data': data, 'beta_carteira': beta_carteira})
beta_check_df = pd.DataFrame(beta_check).set_index('data')
print('Beta ponderado da carteira por data:')
print(beta_check_df)

# DataFrames de validação: ativos escolhidos em cada decil
df_low = pd.DataFrame.from_dict(ativos_low_dict, orient='index')
df_high = pd.DataFrame.from_dict(ativos_high_dict, orient='index')
print('Ativos do menor decil (comprados) por data:')
print(df_low)
print('Ativos do maior decil (vendidos) por data:')
print(df_high)

# # CDI para custo de financiamento
df_cdi = sgs.dataframe(12, start= datetime.strptime(ini_date, "%Y-%m-%d").strftime("%d/%m/%Y"), end= datetime.strptime(end_date, "%Y-%m-%d").strftime("%d/%m/%Y"))
df_cdi[12] = df_cdi[12]/100 # deixando em porcentagem
financing_spread = 0.01 / 252  # spread de 1% a.a. sobre CDI
financing_rate = df_cdi[12] + financing_spread

# Alinha preços, pesos e CDI ao período da estratégia
if len(rebalance_validas) > 0:
    precos_sim = precos.loc[rebalance_validas[0]:, [a for a in precos.columns if a != benchmark]]
    pesos_df_sim = pesos_df.loc[rebalance_validas[0]:]
    rf_cdi_sim = df_cdi[12].reindex(precos_sim.index).fillna(0.0)
financing_rate_sim = financing_rate.reindex(precos_sim.index).fillna(0.0)

# Simula portfólio Betting Against Beta
res = simulate_portfolio(
    prices=precos_sim,
    weights=pesos_df_sim,
    initial_investment=100000,
    rebalance_freq=None,
    transaction_cost_bps=5,
    allow_fractional=False,
    lot_size=1,
    rf_daily=rf_cdi_sim,
    leverage=1.0,
    financing_rate_daily=financing_rate_sim
)

# Analisa resultado
# Analisa resultado
stats = strategy_stats(data=res.portfolio_value, rf_daily=rf_cdi_sim)
print('Retorno anualizado:', stats['annualized_return'])
print('Volatilidade anualizada:', stats['annualized_vol'])
print('Valor final da carteira:', res.portfolio_value.iloc[-1])
print('Caixa final:', res.cash.iloc[-1])
print('Maior alavancagem (valor investido / equity):', ((res.positions * precos_sim).abs().sum(axis=1) / res.portfolio_value).max())

# Comparando BOVA11
df_comparacao = pd.concat([res.portfolio_value,precos['BOVA11.SA']],axis=1)
df_comparacao = df_comparacao.dropna()
df_comparacao['retorno_portfolio'] = df_comparacao[0].pct_change()   # retornos simples
df_comparacao['cota_portfolio_acum'] = (1 + df_comparacao['retorno_portfolio']).cumprod()
df_comparacao['retorno_mercado'] = df_comparacao['BOVA11.SA'].pct_change()   # retornos simples
df_comparacao['cota_mercado_acum'] = (1 + df_comparacao['retorno_mercado']).cumprod()
print('a')
