import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yfinance as yf

# ================== Funções ==================

def _feasible_w0(n, lo, hi):
    """
    Gera um ponto inicial viável: começa nos pisos (lo) e distribui o restante
    até os tetos (hi). 'lo' e 'hi' podem ser escalares ou arrays (tamanho n).
    """
    lo = np.full(n, lo, dtype=float) if np.isscalar(lo) else np.asarray(lo, float)
    hi = np.full(n, hi, dtype=float) if np.isscalar(hi) else np.asarray(hi, float)

    if lo.sum() > 1 + 1e-12:
        raise ValueError(f"Inviável: soma dos mínimos = {lo.sum():.4f} > 1.")
    if hi.sum() < 1 - 1e-12:
        raise ValueError(f"Inviável: soma dos máximos = {hi.sum():.4f} < 1.")

    w = lo.copy()
    slack = 1.0 - w.sum()
    if slack <= 1e-12:
        return w / w.sum()

    for _ in range(8):
        room = hi - w
        mask = room > 1e-12
        if not mask.any():
            break
        add_each = slack / mask.sum()
        inc = np.minimum(room[mask], add_each)
        w[mask] += inc
        slack -= inc.sum()
        if slack <= 1e-12:
            break

    return w / w.sum()

def markowitz_optimize(
    data: pd.DataFrame,
    rf: float = 0.0,                 # taxa livre de risco ANUAL
    is_returns: bool = False,        # True se 'data' já for retornos; False se forem preços
    periods_per_year: int = 252,
    min_weight: float = 0.0,         # piso por ativo (ex.: 0.05 = 5%)
    max_weight: float = 1.0          # teto por ativo (ex.: 0.40 = 40%)
):
    """
    Retorna pesos de Mínima Variância e Máximo Sharpe + métricas.
    data: DataFrame (index datetime, colunas = ativos) com PREÇOS ou RETORNOS.
    """
    # 1) Retornos por período
    if is_returns:
        r = data.dropna(how="any")
    else:
        r = data.sort_index().pct_change().dropna(how="any")

    names = list(r.columns)
    n = len(names)

    # 2) μ e Σ (anuais)
    mu = r.mean().values * periods_per_year
    Sigma = r.cov().values * periods_per_year

    # 3) Bounds + ponto inicial viável
    assert n * min_weight <= 1 + 1e-12, f"Inviável: n*min_weight = {n*min_weight:.2f} > 1"
    assert n * max_weight >= 1 - 1e-12, f"Inviável: n*max_weight = {n*max_weight:.2f} < 1"
    bnds = [(min_weight, max_weight)] * n
    w0 = _feasible_w0(n, min_weight, max_weight)

    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

    def stats(w):
        ret = float(w @ mu)
        vol = float(np.sqrt(w @ Sigma @ w))
        shp = (ret - rf) / (vol + 1e-12) if vol > 0 else np.nan
        return ret, vol, shp

    # 4) Mínima Variância
    res_mv = minimize(lambda w: w @ Sigma @ w, w0,
                      method='SLSQP', bounds=bnds, constraints=cons,
                      options={'maxiter': 300, 'ftol': 1e-9})
    if not res_mv.success:
        raise RuntimeError(f"MinVar não convergiu: {res_mv.message}")
    w_mv = res_mv.x

    # 5) Máximo Sharpe
    res_ms = minimize(lambda w: -((w @ mu - rf) / (np.sqrt(w @ Sigma @ w) + 1e-12)),
                      w0, method='SLSQP', bounds=bnds, constraints=cons,
                      options={'maxiter': 300, 'ftol': 1e-9})
    if not res_ms.success:
        raise RuntimeError(f"MaxSharpe não convergiu: {res_ms.message}")
    w_ms = res_ms.x

    # 6) Métricas
    r_mv, v_mv, s_mv = stats(w_mv)
    r_ms, v_ms, s_ms = stats(w_ms)

    return {
        "assets": names,
        "w_min_variance": dict(zip(names, w_mv)),
        "w_max_sharpe": dict(zip(names, w_ms)),
        "min_variance_stats": {"ret": r_mv, "vol": v_mv, "sharpe": s_mv},
        "max_sharpe_stats": {"ret": r_ms, "vol": v_ms, "sharpe": s_ms},
    }

# ================== Aplicação (exemplo) ==================
if __name__ == "__main__":
    tickers = ["WEGE3.SA","LREN3.SA","VALE3.SA","PETR4.SA","EQTL3.SA","EGIE3.SA"]
    start = "2025-05-16"
    end   = "2025-09-05"
    rf    = 0.0        # taxa livre de risco ANUAL
    min_w = 0.05       # piso de 5% por ativo
    max_w = 1.0        # sem teto; se quiser limitar concentração, use 0.35/0.40

    # Baixar preços ajustados
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    # Extrair 'Close' mesmo em MultiIndex
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw.copy()
        if "Close" in prices.columns:
            prices = prices["Close"].to_frame() if prices["Close"].ndim == 1 else prices[["Close"]]
            # Se for Series (1 ticker apenas), transforme em DF
            if prices.shape[1] == 1 and tickers and prices.columns[0] != tickers[0]:
                prices.columns = tickers[:1]
        # Se já vier no formato colunas=tickers, mantém

    # Garantir apenas os tickers desejados e remover colunas extras
    prices = prices.reindex(columns=tickers)

    # Rodar Markowitz simples com piso
    res = markowitz_optimize(
        data=prices,
        rf=rf,
        is_returns=False,
        periods_per_year=252,
        min_weight=min_w,
        max_weight=max_w
    )

    # Mostrar resultados
    w_ms = pd.Series(res["w_max_sharpe"]).sort_values(ascending=False)
    w_mv = pd.Series(res["w_min_variance"]).sort_values(ascending=False)

    print("\n== Pesos Máximo Sharpe ==")
    print((w_ms*100).round(2).astype(str) + "%")
    print("\n== Pesos Mínima Variância ==")
    print((w_mv*100).round(2).astype(str) + "%")

    print("\n== Métricas Máximo Sharpe ==")
    print({k: round(v, 4) for k, v in res["max_sharpe_stats"].items()})
    print("\n== Métricas Mínima Variância ==")
    print({k: round(v, 4) for k, v in res["min_variance_stats"].items()})

    # Checagens rápidas
    print("\nMin peso (Máx Sharpe):", float(w_ms.min()))
    print("Min peso (Min Var):    ", float(w_mv.min()))
    print("Soma pesos (Máx Sharpe):", float(w_ms.sum()))
    print("Soma pesos (Min Var):    ", float(w_mv.sum()))
