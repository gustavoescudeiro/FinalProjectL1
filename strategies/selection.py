# Funções de seleção de ativos (universo)
def universe_all(date, prices):
    return list(prices.columns)

def universe_top_price(date, prices):
    if date not in prices.index:
        return list(prices.columns)
    top2 = prices.loc[date].sort_values(ascending=False).head(2).index.tolist()
    return top2

def universe_alternate(date, prices):
    month = date.month
    if month % 2 == 0:
        return [a for a in ["WEGE3.SA", "VALE3.SA"] if a in prices.columns]
    else:
        return [a for a in ["PETR4.SA", "ITUB4.SA"] if a in prices.columns]
