import yfinance as yf
import pandas as pd
from src.Allocation import simulate_portfolio
from src.Metrics import strategy_stats
import sgs
from datetime import datetime

# Parâmetros

# Parâmetros de simulação
ini_date = '2025-09-01'  # data de início da estratégia (primeiro rebalanceamento)
end_date = '2025-09-15'

pesos = {
    'ABEV3.SA': 0.024710,
    'ALOS3.SA': 0.005240,
    'ASAI3.SA': 0.006620,
    'AURE3.SA': 0.001620,
    'AZZA3.SA': 0.002190,
    'B3SA3.SA': 0.031580,
    'BBAS3.SA': 0.028470,
    'BBDC3.SA': 0.009920,
    'BBDC4.SA': 0.040240,
    'BBSE3.SA': 0.009470,
    'BEEF3.SA': 0.001220,
    'BPAC11.SA': 0.029510,
    'BRAP4.SA': 0.001920,
    'BRAV3.SA': 0.004330,
    'BRFS3.SA': 0.005890,
    'BRKM5.SA': 0.001170,
    'CEAB3.SA': 0.001090,
    'CMIG4.SA': 0.009900,
    'CMIN3.SA': 0.004010,
    'COGN3.SA': 0.002480,
    'CPFE3.SA': 0.003460,
    'CPLE6.SA': 0.009410,
    'CSAN3.SA': 0.003140,
    'CSNA3.SA': 0.002590,
    'CURY3.SA': 0.002160,
    'CVCB3.SA': 0.000450,
    'CXSE3.SA': 0.003960,
    'CYRE3.SA': 0.003370,
    'DIRR3.SA': 0.002360,
    'EGIE3.SA': 0.004770,
    'ELET3.SA': 0.038410,
    'ELET6.SA': 0.006040,
    'EMBR3.SA': 0.026170,
    'ENEV3.SA': 0.013530,
    'ENGI11.SA': 0.007440,
    'EQTL3.SA': 0.021340,
    'FLRY3.SA': 0.003120,
    'GGBR4.SA': 0.009950,
    'GOAU4.SA': 0.002760,
    'HAPV3.SA': 0.006080,
    'HYPE3.SA': 0.003350,
    'IGTI11.SA': 0.002330,
    'IRBR3.SA': 0.001830,
    'ISAE4.SA': 0.004300,
    'ITSA4.SA': 0.030700,
    'ITUB4.SA': 0.083070,
    'KLBN11.SA': 0.006800,
    'LREN3.SA': 0.007620,
    'MGLU3.SA': 0.001360,
    'MOTV3.SA': 0.006670,
    'MRFG3.SA': 0.002200,
    'MRVE3.SA': 0.001330,
    'MULT3.SA': 0.004090,
    'NATU3.SA': 0.003540,
    'PCAR3.SA': 0.000730,
    'PETR3.SA': 0.041770,
    'PETR4.SA': 0.064230,
    'POMO4.SA': 0.002870,
    'PRIO3.SA': 0.013530,
    'PSSA3.SA': 0.004390,
    'RADL3.SA': 0.010570,
    'RAIL3.SA': 0.008280,
    'RAIZ4.SA': 0.000670,
    'RDOR3.SA': 0.020120,
    'RECV3.SA': 0.001660,
    'RENT3.SA': 0.016400,
    'SANB11.SA': 0.004760,
    'SBSP3.SA': 0.034200,
    'SLCE3.SA': 0.001620,
    'SMFT3.SA': 0.004880,
    'STBP3.SA': 0.002730,
    'SUZB3.SA': 0.015840,
    'TAEE11.SA': 0.003570,
    'TIMS3.SA': 0.008570,
    'TOTS3.SA': 0.010760,
    'UGPA3.SA': 0.009860,
    'USIM5.SA': 0.000940,
    'VALE3.SA': 0.111030,
    'VAMO3.SA': 0.001010,
    'VBBR3.SA': 0.012520,
    'VIVA3.SA': 0.001670,
    'VIVT3.SA': 0.011700,
    'WEGE3.SA': 0.026200,
    'YDUQ3.SA': 0.001610
}


# Baixa preços com antecedência suficiente para o cálculo do momentum
df_tickers = yf.download(list(pesos.keys()), start=ini_date, end=end_date)
precos = df_tickers['Close'].dropna(how='all')




# Monta DataFrame de pesos apenas a partir da data de início da estratégia
pesos_df = pd.DataFrame([pesos], index=[pd.Timestamp("2025-09-01")])


# Getting CDI
df_cdi = sgs.dataframe(12, start= datetime.strptime(ini_date, "%Y-%m-%d").strftime("%d/%m/%Y"), end= datetime.strptime(end_date, "%Y-%m-%d").strftime("%d/%m/%Y"))
df_cdi[12] = df_cdi[12]/100 # deixando em porcentagem



# Simula portfólio
res = simulate_portfolio(
    prices=precos,
    weights=pesos_df,
    initial_investment=100000,
    rebalance_freq=None,
    transaction_cost_bps=0,
    allow_fractional=False,
    lot_size=1
)

# Analisa resultado
stats = strategy_stats(data=res.portfolio_value, rf_daily=df_cdi[12])
print('Retorno anualizado:', stats['annualized_return'])
print('Volatilidade anualizada:', stats['annualized_vol'])
print('Valor final da carteira:', res.portfolio_value.iloc[-1])
