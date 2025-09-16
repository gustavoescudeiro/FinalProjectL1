
from backtest.runner import run_strategy
import yfinance as yf
import pandas as pd
from strategies.generic_strategy import GenericStrategy
from pandas.tseries.offsets import BDay
import sgs
from datetime import datetime



def weights_user_fixed(ativos, prices_window, date, pesos_dict):
    # Retorna os pesos fixos para os ativos do universo
    if not ativos:
        return {}
    # Garante que todos os ativos do universo recebam peso do dicionário (mesmo que zero)
    pesos = {a: pesos_dict.get(a, 0.0) for a in ativos}
    total = sum(pesos.values())
    # Se todos os pesos são zero, retorna pesos iguais
    if total == 0:
        return {a: 1/len(ativos) for a in ativos}
    # Senão, normaliza os pesos
    return {a: pesos[a]/total for a in ativos}

class GenericStrategyUserFixed(GenericStrategy):
    def __init__(self, ativos_usuario, pesos_dict):
        super().__init__(ativos_usuario, self.weights_fn)
        self.pesos_dict = pesos_dict

    def weights_fn(self, ativos, prices_window, date):
        return weights_user_fixed(ativos, prices_window, date, self.pesos_dict)

if __name__ == "__main__":
    import yfinance as yf
    import pandas as pd
    from strategies.weights import weights_equal, weights_vol, weights_markowitz
    from strategies.generic_strategy import GenericStrategy
    from backtest.runner import run_strategy

    ativos_usuario = ["IMAB11.SA", "SPXI11.SA"]
    ini_date = '2022-09-12'
    end_date = '2025-09-11'
    initial_investment = 100000
    rebalance_freq = 'M'
    transaction_cost_bps = 0
    lot_size = 1
    allow_fractional = False

    # Getting CDI
    df_cdi = sgs.dataframe(12, start=datetime.strptime(ini_date, "%Y-%m-%d").strftime("%d/%m/%Y"), end=datetime.strptime(end_date, "%Y-%m-%d").strftime("%d/%m/%Y"))
    df_cdi[12] = df_cdi[12] / 100

    ini_date_prices = pd.to_datetime(ini_date) - BDay(60)
    df_tickers = yf.download(ativos_usuario, start=str(ini_date_prices.date()), end=end_date)
    precos = df_tickers['Close'].dropna()
    # Cria calendário de rebalanceamento
    ini_date_dt = pd.to_datetime(ini_date)
    if ini_date_dt not in precos.index:
        raise ValueError(f"Não há preços para a data inicial {ini_date}")
    calendario_rebal = precos.resample(rebalance_freq).first().index.intersection(precos.index)
    calendario_rebal = calendario_rebal[calendario_rebal > ini_date_dt]
    calendario = pd.Index([ini_date_dt]).append(calendario_rebal)

    # # Teste Equal Weighted
    # strategy_equal = GenericStrategy(ativos_usuario, weights_equal)
    # res_equal = run_strategy(strategy_equal, precos, calendario, start_date=ini_date, weights_window=None, initial_investment=initial_investment, transaction_cost_bps=transaction_cost_bps, allow_fractional=allow_fractional, lot_size=lot_size)
    # print('GenericStrategy - Equal Weighted:')
    # for k, v in res_equal.items():
    #     print(f'--- {k} ---')
    #     print(v)
    #
    # # Teste Vol Weighted
    # strategy_vol = GenericStrategy(ativos_usuario, weights_vol, weights_window=21)
    # res_vol = run_strategy(strategy_vol, precos, calendario, start_date=ini_date, weights_window=21, initial_investment=initial_investment, transaction_cost_bps=transaction_cost_bps, allow_fractional=allow_fractional, lot_size=lot_size)
    # print('GenericStrategy - Vol Weighted:')
    # for k, v in res_vol.items():
    #     print(f'--- {k} ---')
    #     print(v)
    # #
    # # # Teste Markowitz Weighted
    # # strategy_markowitz = GenericStrategy(ativos_usuario, weights_markowitz, weights_window=42)
    # # res_markowitz = run_strategy(strategy_markowitz, precos, calendario, start_date=ini_date, weights_window=42, initial_investment=initial_investment, transaction_cost_bps=transaction_cost_bps, allow_fractional=allow_fractional, lot_size=lot_size)
    # # print('GenericStrategy - Markowitz Weighted:')
    # # for k, v in res_markowitz.items():
    # #     print(f'--- {k} ---')
    # #     print(v)

    # Teste User Fixed Weights
    pesos_dict = {"IMAB11.SA": 0.8, "SPXI11.SA": 0.2}
    strategy_user_fixed = GenericStrategyUserFixed(ativos_usuario, pesos_dict)
    res_user_fixed = run_strategy(strategy_user_fixed, precos, calendario, start_date=ini_date, weights_window=None, initial_investment=initial_investment, transaction_cost_bps=transaction_cost_bps, allow_fractional=allow_fractional, lot_size=lot_size, rf_daily=df_cdi[12])
    print('GenericStrategy - User Fixed Weights (IMAB11=80%, SPXI=20%):')
    for k, v in res_user_fixed.items():
        print(f'--- {k} ---')
        print(v)