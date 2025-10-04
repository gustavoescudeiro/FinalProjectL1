from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd



@dataclass
class SimulationResult:
    portfolio_value: pd.Series       # valor total do portfólio ao longo do tempo
    cash: pd.Series                  # caixa ao longo do tempo
    positions: pd.DataFrame          # quantidade de cada ativo ao longo do tempo
    trades: pd.DataFrame             # trades (quantidades) executados nas datas de rebalance
    weights_realized: pd.DataFrame   # pesos realizados por data
    assets_pnl: pd.DataFrame         # resultado financeiro de cada ativo por data

def _round_to_lot(qty: np.ndarray, lot: int, allow_fractional: bool) -> np.ndarray:
    if allow_fractional:
        return qty
    lot = max(int(lot), 1)
    return (np.floor(qty / lot) * lot).astype(float)

def simulate_portfolio(
    prices: pd.DataFrame,
    weights: dict[str, float] | list[float] | np.ndarray | None = None,   # alvos percentuais
    shares: dict[str, float] | list[float] | np.ndarray | None = None,    # alvos de quantidade
    initial_investment: float = 10_000.0,
    rebalance_freq: str | None = 'M',        # 'D','W','M','Q','A' ou None (buy-and-hold)
    when: str = 'first',                     # 'first' ou 'last' (pregão do período para rebalance)
    transaction_cost_bps: float = 0.0,       # custo por lado em bps (0.0001 = 1bp)
    allow_fractional: bool = False,          # True permite fracionário
    lot_size: int | dict[str, int] = 1,      # 1, 100, ou dict por ticker
    cash_symbol: str = "CASH",               # label do cash
    # >>> NOVOS PARÂMETROS PARA REMUNERAR O CAIXA <<<
    rf_daily: pd.Series | None = None,       # série de retornos diários (ex.: CDI), alinhada por data
    rf_timing: Literal["start","end"] = "start",  # aplica antes ("start") ou depois ("end") dos trades do dia
    rf_apply_to_negative: bool = True,        # se True, também aplica (cobra) quando caixa for negativo
    leverage: float = 1.0,                    # alavancagem global (default 1.0)
    financing_rate_daily: float | pd.Series | None = None,  # custo diário de financiamento (juros sobre caixa negativo)
    maturities: dict[str, str] | pd.DataFrame | None = None,  # datas de vencimento dos ativos (dict: ticker -> str ou DataFrame: colunas 'ticker', 'maturity')
    maturity_action: str = 'cash'  # 'cash' (venda vira caixa) ou 'rebalance' (venda + rebalanceamento imediato)
) -> SimulationResult:
    """
    Simula carteira com suporte a:
      - Alocação por 'weights' (percentual do patrimônio)
      - Alocação por 'shares' (quantidade de ativos)
    Pode rebalancear por frequência ou fazer buy-and-hold.
    Custos (bps) aplicados ao valor negociado em cada trade (por lado).

    Remuneração do caixa:
      - Passe 'rf_daily' como Série de retornos diários (decimal), indexada por data.
      - 'rf_timing' define se o juro incide antes ('start') ou depois ('end') dos trades do dia.
      - Se 'rf_apply_to_negative' for True, o mesmo retorno é aplicado quando o caixa for negativo
        (simulando custo de carregamento); caso False, não incide sobre caixa negativo.

    Regras:
      - Informe exatamente um: 'weights' OU 'shares'.
      - 'shares' com rebalance tenta manter as quantidades alvo nas datas de rebalance.
      - Se faltar caixa, a compra é reduzida proporcionalmente (weights) ou por lote (shares).
      - Arredondamento controlado por 'allow_fractional' e 'lot_size'.

    'prices' deve ser DataFrame: índice datetime e colunas = tickers (preços).
    """
    if (weights is None) == (shares is None):
        raise ValueError("Informe exatamente um: 'weights' OU 'shares'.")

    # Índice de datas seguro
    prices = prices.sort_index()
    prices.index = pd.to_datetime(prices.index, errors="raise")
    tickers = list(prices.columns)

    # --- Alinhamento do risk-free diário (se fornecido) ---
    if rf_daily is not None:
        if not isinstance(rf_daily, pd.Series):
            raise ValueError("rf_daily deve ser um pd.Series de retornos diários (decimal).")
        rf_daily = rf_daily.copy()
        rf_daily.index = pd.to_datetime(rf_daily.index, errors="raise")
        # Alinhar ao índice de prices; onde não houver dado, assume 0
        rf_daily = rf_daily.reindex(prices.index).fillna(0.0)

    # Normalizar alvos
    if weights is not None:
        if isinstance(weights, pd.DataFrame):
            weights_df = weights.copy()
            def get_w_for_date(dt):
                if dt not in weights_df.index:
                    return np.zeros(len(tickers))
                row = weights_df.loc[dt]
                arr = np.array([row.get(t, 0.0) for t in tickers], dtype=float)
                # Permite pesos negativos (short)
                s = arr.sum()
                if s == 0:
                    return arr  # tudo zero, sem alocação
                if s > 1.0 + 1e-8:
                    raise ValueError(f"A soma dos weights ({s:.4f}) é maior que 1. Os pesos devem somar 1 ou menos.")
                return arr / s
            mode = "weights_df"
        elif isinstance(weights, dict):
            w = np.array([weights.get(t, 0.0) for t in tickers], dtype=float)
            # Permite pesos negativos (short)
            s = w.sum()
            if s == 0:
                raise ValueError("A soma dos weights é zero.")
            if s > 1.0 + 1e-8:
                raise ValueError(f"A soma dos weights ({s:.4f}) é maior que 1. Os pesos devem somar 1 ou menos.")
            w = w / s
            mode = "weights"
        else:
            w = np.asarray(weights, dtype=float)
            if w.size != len(tickers):
                raise ValueError("weights deve ter mesmo comprimento das colunas de 'prices'.")
            # Permite pesos negativos (short)
            s = w.sum()
            if s == 0:
                raise ValueError("A soma dos weights é zero.")
            if s > 1.0 + 1e-8:
                raise ValueError(f"A soma dos weights ({s:.4f}) é maior que 1. Os pesos devem somar 1 ou menos.")
            w = w / s
            mode = "weights"
    else:
        if isinstance(shares, dict):
            s_target = np.array([shares.get(t, 0.0) for t in tickers], dtype=float)
        else:
            s_target = np.asarray(shares, dtype=float)
            if s_target.size != len(tickers):
                raise ValueError("shares deve ter mesmo comprimento das colunas de 'prices'.")
        if (s_target < 0).any():
            raise ValueError("shares não podem ser negativos.")
        mode = "shares"

    # Datas de rebalance
    if isinstance(weights, pd.DataFrame):
        rebalance_dates = [d for d in weights.index if (weights.loc[d] != 0).any()]
        # Garante rebalance no primeiro dia, mesmo que não haja linha de pesos
        first_day = prices.index[0]
        if first_day not in rebalance_dates:
            rebalance_dates = [first_day] + rebalance_dates
    elif rebalance_freq is None:
        rebalance_dates = prices.index[[0]]
    else:
        if when == "first":
            rb = prices.resample(rebalance_freq).first().index.intersection(prices.index)
        else:
            rb = prices.resample(rebalance_freq).last().index.intersection(prices.index)
        rebalance_dates = rb.union(prices.index[:1]).sort_values()

    # Estruturas
    pos = pd.DataFrame(0.0, index=prices.index, columns=tickers)
    cash = pd.Series(0.0, index=prices.index, name=cash_symbol)
    trades = pd.DataFrame(0.0, index=prices.index, columns=tickers)
    tc = float(transaction_cost_bps) / 10_000.0

    # lot size por ticker
    if isinstance(lot_size, dict):
        lot_map = {t: int(max(1, lot_size.get(t, 1))) for t in tickers}
    else:
        lot_map = {t: int(max(1, lot_size)) for t in tickers}

    # Auxiliares
    def apply_rounding(target_qty: np.ndarray) -> np.ndarray:
        out = target_qty.copy().astype(float)
        if not allow_fractional:
            for i, t in enumerate(tickers):
                out[i] = _round_to_lot(out[i], lot_map[t], allow_fractional=False)
        return out

    def target_from_weights(t: pd.Timestamp, equity_value: float, leverage_: float) -> np.ndarray:
        px = prices.loc[t].values.astype(float)
        px = np.nan_to_num(px, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

        if mode == 'weights_df':
            w_now = get_w_for_date(t)
        else:
            w_now = w
        if w_now.sum() == 0:
            return np.zeros(len(tickers))
        target_value = equity_value * leverage_ * w_now
        qty = target_value / np.clip(px, 1e-12, None)
        return apply_rounding(qty)

    def enforce_cash_constraint(px: np.ndarray, curr_qty: np.ndarray, desired_qty: np.ndarray, avail_cash: float) -> tuple[np.ndarray, float]:
        delta = desired_qty - curr_qty
        notional = (np.abs(delta) * px).sum()
        total_cost = notional * tc
        cash_after = avail_cash - (delta.clip(min=0) * px).sum() - total_cost + (delta.clip(max=0) * (-px)).sum()
        # Permite caixa negativo até o limite de margem (valor investido <= equity * leverage)
        # Aqui, equity = caixa + valor dos ativos antes do rebalance
        # valor investido = sum(abs(desired_qty) * px)
        valor_ativos_antes = (curr_qty * px).sum()
        equity = avail_cash + valor_ativos_antes
        valor_investido = (desired_qty * px).sum()
        max_investido = equity * leverage
        if valor_investido <= max_investido + 1e-6:
            return desired_qty, cash_after
        # Se exceder o limite de margem, reduz proporcionalmente
        fator = max_investido / max(valor_investido, 1e-12)
        q_adj = curr_qty + (desired_qty - curr_qty) * fator
        q_adj = apply_rounding(q_adj)
        delta2 = q_adj - curr_qty
        notional2 = (np.abs(delta2) * px).sum()
        total_cost2 = notional2 * tc
        cash_after2 = avail_cash - (delta2.clip(min=0) * px).sum() - total_cost2 + (delta2.clip(max=0) * (-px)).sum()
        return q_adj, cash_after2

    # Inicialização
    cash.iloc[0] = initial_investment
    curr_qty = np.zeros(len(tickers), dtype=float)
    rebalance_set = set(pd.to_datetime(rebalance_dates))
    first_day = prices.index[0]

    # Loop

    for t in prices.index:
        px = prices.loc[t].values.astype(float)
        px = np.nan_to_num(px, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

        # --- VENDA AUTOMÁTICA DE ATIVOS VENCIDOS ---
        rebalance_now = False
        if maturities is not None:
            if isinstance(maturities, dict):
                venc_dict = maturities
            elif isinstance(maturities, pd.DataFrame):
                venc_dict = dict(zip(maturities['ticker'], maturities['maturity']))
            else:
                venc_dict = {}
            for i, ticker in enumerate(tickers):
                if ticker in venc_dict:
                    data_venc = pd.to_datetime(venc_dict[ticker])
                    if pd.to_datetime(t) >= data_venc and curr_qty[i] != 0:
                        # Vende o ativo vencido ao preço do dia
                        valor_venda = curr_qty[i] * px[i]
                        cash.loc[t] += valor_venda
                        trades.loc[t, ticker] = -curr_qty[i]
                        curr_qty[i] = 0
                        if maturity_action == 'rebalance':
                            rebalance_now = True

        # ---- Remuneração do caixa (timing = start) ----
        if rf_daily is not None and rf_timing == "start" and t != first_day:
            r = float(rf_daily.loc[t])
            c0 = cash.loc[t]
            if (c0 >= 0) or rf_apply_to_negative:
                cash.loc[t] = c0 * (1.0 + r)

        # Custo de financiamento para caixa negativo (antes dos trades)
        if financing_rate_daily is not None and cash.loc[t] < 0:
            if isinstance(financing_rate_daily, pd.Series):
                rate = float(financing_rate_daily.loc[t])
            else:
                rate = float(financing_rate_daily)
            cash.loc[t] = cash.loc[t] * (1.0 + rate)

        # Rebanceamento nas datas-alvo ou na data de vencimento se maturity_action == 'rebalance'
        if t in rebalance_set or rebalance_now:
            # Identifica ativos válidos (não vencidos)
            ativos_validos = np.ones(len(tickers), dtype=bool)
            if maturities is not None:
                if isinstance(maturities, dict):
                    venc_dict = maturities
                elif isinstance(maturities, pd.DataFrame):
                    venc_dict = dict(zip(maturities['ticker'], maturities['maturity']))
                else:
                    venc_dict = {}
                for i, ticker in enumerate(tickers):
                    if ticker in venc_dict:
                        data_venc = pd.to_datetime(venc_dict[ticker])
                        if pd.to_datetime(t) >= data_venc:
                            ativos_validos[i] = False

            # Calcula pesos/quantidades apenas para ativos válidos
            if mode == "weights" or mode == "weights_df":
                equity_now = cash.loc[t] + (curr_qty * px).sum()
                if mode == "weights_df":
                    w_now = get_w_for_date(t)
                else:
                    w_now = w
                # Zera pesos dos ativos vencidos
                w_valid = w_now * ativos_validos
                s = w_valid.sum()
                if s > 0:
                    w_valid = w_valid / s
                desired_qty = np.zeros(len(tickers))
                target_value = equity_now * leverage * w_valid
                desired_qty[ativos_validos] = target_value[ativos_validos] / np.clip(px[ativos_validos], 1e-12, None)
                desired_qty = apply_rounding(desired_qty)
            else:
                # shares: zera alvos dos vencidos
                s_target_valid = s_target.copy()
                s_target_valid[~ativos_validos] = 0.0
                desired_qty = apply_rounding(s_target_valid)

            desired_qty, cash_after = enforce_cash_constraint(px, curr_qty, desired_qty, cash.loc[t])

            delta = desired_qty - curr_qty
            trades.loc[t] = delta
            notional = (np.abs(delta) * px).sum()
            fees = notional * tc

            # Atualiza caixa após trades + custos
            cash.loc[t] = cash.loc[t] - (delta.clip(min=0) * px).sum() - fees + (delta.clip(max=0) * (-px)).sum()
            curr_qty = desired_qty

        # Custo de financiamento para caixa negativo (após trades)
        if financing_rate_daily is not None and cash.loc[t] < 0:
            if isinstance(financing_rate_daily, pd.Series):
                rate = float(financing_rate_daily.loc[t])
            else:
                rate = float(financing_rate_daily)
            cash.loc[t] = cash.loc[t] * (1.0 + rate)

        # ---- Remuneração do caixa (timing = end) ----
        if rf_daily is not None and rf_timing == "end":
            r = float(rf_daily.loc[t])
            c0 = cash.loc[t]
            if (c0 >= 0) or rf_apply_to_negative:
                cash.loc[t] = c0 * (1.0 + r)

        # Carregar posições/cash para a próxima linha
        pos.loc[t] = curr_qty
        if t != prices.index[-1]:
            cash.iloc[cash.index.get_loc(t) + 1] = cash.loc[t]

    # Saídas
    portfolio_value = (pos * prices).sum(axis=1) + cash
    weights_realized = (pos * prices).div(portfolio_value, axis=0).fillna(0.0)

    # Cálculo do P&L diário de cada ativo
    pos_shift = pos.shift(1).fillna(0.0)
    price_diff = prices.diff().fillna(0.0)
    assets_pnl = pos_shift * price_diff

    return SimulationResult(
        portfolio_value=portfolio_value,
        cash=cash,
        positions=pos,
        trades=trades.replace(0.0, np.nan),
        weights_realized=weights_realized,
        assets_pnl=assets_pnl
    )



