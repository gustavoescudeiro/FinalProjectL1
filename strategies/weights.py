# Funções de estabelecimento de pesos
from src.Optimization import markowitz_optimize

def weights_equal(ativos, prices_window, date):
    n = len(ativos)
    return {a: 1/n for a in ativos} if n > 0 else {}

def weights_vol(ativos, prices_window, date):
    if len(ativos) == 0:
        return {}
    vol = prices_window[ativos].pct_change().std()
    inv_vol = 1 / vol.replace(0, 1e-6)
    w = inv_vol / inv_vol.sum()
    return w.to_dict()

def weights_momentum(ativos, prices_window, date):
    if len(ativos) == 0:
        return {}
    ret = prices_window[ativos].pct_change().sum()
    pos_ret = ret.clip(lower=0)
    if pos_ret.sum() == 0:
        return weights_equal(ativos, prices_window, date)
    w = pos_ret / pos_ret.sum()
    return w.to_dict()

def weights_markowitz(ativos, prices_window, date):
    if len(ativos) == 0 or prices_window.shape[0] < 2:
        return {}
    returns = prices_window[ativos].pct_change().dropna()
    if returns.shape[0] < 2:
        return weights_equal(ativos, prices_window, date)
    try:
        res = markowitz_optimize(
            data=returns,
            rf=0.0,
            is_returns=True,
            periods_per_year=252,
            min_weight=0.0,
            max_weight=1.0
        )
        w_dict = res.get("w_max_sharpe", {})
        s = sum(max(0.0, w_dict.get(n, 0.0)) for n in ativos)
        if s <= 0:
            return {n: 0.0 for n in ativos}
        return {n: max(0.0, w_dict.get(n, 0.0))/s for n in ativos}
    except Exception:
        return weights_equal(ativos, prices_window, date)
