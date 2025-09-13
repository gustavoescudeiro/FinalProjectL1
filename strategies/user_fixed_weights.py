# Estratégia long only com seleção e pesos fixos definidos pelo usuário

def user_universe_n(date, prices, ativos_usuario):
    # Sempre retorna os n ativos definidos pelo usuário que estão disponíveis
    return [a for a in ativos_usuario if a in prices.columns]

def user_weights_fixed(ativos, prices_window, date, pesos_dict):
    # Retorna os pesos fixos para os ativos do universo
    total = sum(pesos_dict.get(a, 0.0) for a in ativos)
    if total == 0:
        return {a: 1/len(ativos) for a in ativos} if ativos else {}
    return {a: pesos_dict.get(a, 0.0)/total for a in ativos}

class UserFixedStrategy:
    def __init__(self, ativos_usuario, pesos_dict):
        self.ativos_usuario = ativos_usuario
        self.pesos_dict = pesos_dict

    def get_universe(self, date, prices):
        return user_universe_n(date, prices, self.ativos_usuario)

    def get_weights(self, ativos, prices_window, date):
        return user_weights_fixed(ativos, prices_window, date, self.pesos_dict)

# Exemplo de teste da estratégia UserFixedStrategy
if __name__ == "__main__":
    from backtest.runner import run_strategy
    import yfinance as yf
    import pandas as pd

    ativos_usuario = ["IMAB11.SA", "SPXI11.SA"]
    pesos_dict = {"IMAB11.SA": 0.8, "SPXI11.SA": 0.2}
    strategy = UserFixedStrategy(ativos_usuario, pesos_dict)

    ini_date = '2025-05-16'
    end_date = '2025-09-05'
    initial_investment = 100000
    rebalance_freq = 'M'
    transaction_cost_bps = 5
    lot_size = 1
    allow_fractional = False

    df_tickers = yf.download(ativos_usuario, start=ini_date, end=end_date)
    precos = df_tickers['Close'].dropna()
    calendario = precos.resample(rebalance_freq).first().index.intersection(precos.index)
    calendario = calendario.insert(0, precos.index[0])
    calendario = calendario.sort_values()

    res = run_strategy(strategy, precos, calendario, start_date=ini_date, weights_window=None, initial_investment=initial_investment, transaction_cost_bps=transaction_cost_bps, allow_fractional=allow_fractional, lot_size=lot_size)

    print('User Fixed Strategy (80% IMAB11, 20% SPXI11):')
    for k, v in res.items():
        print(f'--- {k} ---')
        print(v)
