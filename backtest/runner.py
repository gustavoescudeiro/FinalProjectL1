# Função para rodar o backtest de uma estratégia

def run_strategy(strategy, precos, calendario, start_date=None, weights_window=None, initial_investment=100000, transaction_cost_bps=5, allow_fractional=False, lot_size=1):
    import pandas as pd
    if start_date is not None:
        calendario = calendario[calendario >= pd.to_datetime(start_date)]
    portfolio_value = pd.Series(dtype=float)
    cash = pd.Series(dtype=float)
    positions = pd.DataFrame(dtype=float)
    trades = pd.DataFrame(dtype=float)
    weights_realized = pd.DataFrame(dtype=float)
    for i, data_rebal in enumerate(calendario):
        ativos = strategy.get_universe(data_rebal, precos)
        if weights_window is not None:
            end_idx = precos.index.get_loc(data_rebal)
            start_idx = max(0, end_idx - weights_window)
            prices_window = precos[ativos].iloc[start_idx:end_idx+1]
        else:
            prices_window = precos[ativos].loc[:data_rebal]
        weights = strategy.get_weights(ativos, prices_window, data_rebal)
        if i < len(calendario) - 1:
            prox_data = calendario[i+1]
            precos_periodo = precos[ativos].loc[data_rebal:prox_data]
        else:
            precos_periodo = precos[ativos].loc[data_rebal:]
        from src.Allocation import simulate_portfolio
        res = simulate_portfolio(
            prices=precos_periodo,
            weights=weights,
            initial_investment=initial_investment if i == 0 else portfolio_value.iloc[-1],
            rebalance_freq=None,
            transaction_cost_bps=transaction_cost_bps,
            allow_fractional=allow_fractional,
            lot_size=lot_size
        )
        portfolio_value = pd.concat([portfolio_value, res.portfolio_value])
        cash = pd.concat([cash, res.cash])
        positions = pd.concat([positions, res.positions])
        trades = pd.concat([trades, res.trades])
        weights_realized = pd.concat([weights_realized, res.weights_realized])
    portfolio_value = portfolio_value[~portfolio_value.index.duplicated(keep='first')]
    cash = cash[~cash.index.duplicated(keep='first')]
    positions = positions[~positions.index.duplicated(keep='first')]
    trades = trades[~trades.index.duplicated(keep='first')]
    weights_realized = weights_realized[~weights_realized.index.duplicated(keep='first')]
    return {
        'portfolio_value': portfolio_value,
        'cash': cash,
        'positions': positions,
        'trades': trades,
        'weights_realized': weights_realized
    }
