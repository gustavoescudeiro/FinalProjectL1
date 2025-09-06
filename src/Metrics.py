import numpy as np
import pandas as pd

def strategy_stats(data: pd.DataFrame | pd.Series,
                   col: str | None = None,
                   rf_daily: pd.Series | pd.DataFrame | None = None,
                   rf_col: str | None = None,
                   periods_per_year: int = 252,
                   annualize_by_calendar: bool = False,
                   input_is_return: bool = False,
                   ddof: int = 0,
                   which_month: str | pd.Timestamp | None = None):
    """
    Saídas:
      - total_return
      - annualized_return (CAGR)
      - annualized_vol (a partir de retornos por período)
      - sharpe_ratio (anualizado; excesso diário sobre rf_daily)
      - monthly_returns (retornos simples por mês, EoM)
      - monthly_metrics (DataFrame com métricas anualizadas por mês)
      - month_result (linha do mês solicitado em which_month, se informado)

    'rf_daily' deve ser retorno **diário** do RF (mesma frequência/índice).
    """
    # --- Seleciona/normaliza a série principal ---
    if isinstance(data, pd.Series):
        s = data.copy()
    else:
        if col is None:
            if data.shape[1] != 1:
                raise ValueError("Informe 'col' quando 'data' tiver múltiplas colunas.")
            col = data.columns[0]
        s = data[col].copy()

    s = s.sort_index()
    s.index = pd.to_datetime(s.index, errors="raise")
    s = s.dropna()

    # --- Retornos por período ---
    if input_is_return:
        r = s.dropna()
        total_return = float((1.0 + r).prod() - 1.0) if len(r) else np.nan
        first_day, last_day = (r.index[0], r.index[-1]) if len(r) else (None, None)
    else:
        v = s
        r = v.pct_change().dropna()
        total_return = float(v.iloc[-1] / v.iloc[0] - 1.0) if len(v) > 1 else np.nan
        first_day, last_day = (v.index[0], v.index[-1]) if len(v) else (None, None)

    # --- Annualized return (CAGR) ---
    if first_day is not None and last_day is not None:
        if annualize_by_calendar:
            days = (last_day - first_day).days
            ann_return = float((1 + total_return)**(365 / days) - 1) if days > 0 else np.nan
        else:
            n = len(r)
            ann_return = float((1 + total_return)**(periods_per_year / n) - 1) if n > 0 else np.nan
    else:
        ann_return = np.nan

    # --- Annualized vol ---
    ann_vol = float(r.std(ddof=ddof) * np.sqrt(periods_per_year)) if len(r) > 1 else np.nan

    # --- Risk-free diário alinhado ---
    if rf_daily is None:
        rf_p = pd.Series(0.0, index=r.index)
    else:
        rf_series = rf_daily.copy()
        if isinstance(rf_series, pd.DataFrame):
            if rf_col is None:
                if rf_series.shape[1] != 1:
                    raise ValueError("rf_daily tem múltiplas colunas. Informe 'rf_col'.")
                rf_col = rf_series.columns[0]
            rf_series = rf_series[rf_col]
        rf_series.index = pd.to_datetime(rf_series.index, errors="raise")
        rf_p = rf_series.reindex(r.index).fillna(0.0).astype(float)

    # --- Sharpe anualizado (global) ---
    excess = r - rf_p
    mu = excess.mean()
    sigma = excess.std(ddof=ddof)
    sharpe = float(np.sqrt(periods_per_year) * mu / sigma) if sigma > 0 else np.nan

    # --- Retornos mensais simples (compostos dentro do mês) ---
    if input_is_return:
        ret_m = (1 + r).resample("M").prod().sub(1).dropna()
    else:
        ret_m = s.resample("M").last().pct_change().dropna()

    # --- Métricas ANUALIZADAS por mês ---
    # Vol e Sharpe usam retornos/excessos DIÁRIOS dentro do mês
    g_r = r.groupby(pd.Grouper(freq="M"))
    g_excess = excess.groupby(pd.Grouper(freq="M"))

    # Vol anualizada do mês: std(dia no mês) * sqrt(252)
    vol_ann_month = g_r.std(ddof=ddof) * np.sqrt(periods_per_year)

    # Sharpe anualizado do mês: sqrt(252) * mean(excess_dia) / std(excess_dia)
    mu_ex_m = g_excess.mean()
    sd_ex_m = g_excess.std(ddof=ddof)
    sharpe_ann_month = np.sqrt(periods_per_year) * (mu_ex_m / sd_ex_m)
    sharpe_ann_month = sharpe_ann_month.replace([np.inf, -np.inf], np.nan)

    # Retorno anualizado do mês: (1 + retorno_mensal)^(12) - 1
    ann_return_month = (1 + ret_m)**12 - 1

    monthly_metrics = pd.DataFrame({
        "retorno_mensal": ret_m,
        "retorno_anualizado_mes": ann_return_month,
        "vol_anualizada_mes": vol_ann_month.reindex(ret_m.index),       # alinhar por mês
        "sharpe_anualizado_mes": sharpe_ann_month.reindex(ret_m.index)  # alinhar por mês
    })

    # --- Seleção do "resultado do mês da métrica" ---
    month_result = None
    if which_month is not None and len(monthly_metrics):
        if isinstance(which_month, str) and which_month.lower() == "last":
            month_result = monthly_metrics.iloc[-1]
        else:
            # aceitar "YYYY-MM" ou uma data qualquer dentro do mês
            ts = pd.Timestamp(which_month)
            month_end = ts.to_period("M").to_timestamp("M")
            if month_end in monthly_metrics.index:
                month_result = monthly_metrics.loc[month_end]
            else:
                # tentar por período diretamente
                try:
                    month_result = monthly_metrics.loc[ts]
                except KeyError:
                    month_result = None  # não encontrado

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_vol": ann_vol,
        "sharpe_ratio": sharpe,
        "monthly_returns": ret_m,
        "monthly_metrics": monthly_metrics,
        "month_result": month_result
    }
