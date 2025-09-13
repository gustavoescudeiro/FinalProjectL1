# Classe Strategy para orquestrar seleção e pesos
class Strategy:
    def __init__(self, universe_fn, weights_fn):
        self.universe_fn = universe_fn  # (date, prices) -> [ativos]
        self.weights_fn = weights_fn    # (ativos, prices_window, date) -> {ativo: peso}

    def get_universe(self, date, prices):
        return self.universe_fn(date, prices)

    def get_weights(self, ativos, prices_window, date):
        return self.weights_fn(ativos, prices_window, date)
