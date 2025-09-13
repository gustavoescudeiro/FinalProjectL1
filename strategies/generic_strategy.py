# Estratégia genérica: seleção fixa de ativos e função de pesos parametrizável
class GenericStrategy:
    def __init__(self, ativos_usuario, weights_fn, weights_window=21):
        self.ativos_usuario = ativos_usuario
        self.weights_fn = weights_fn  # função de pesos (ex: weights_equal, weights_vol, weights_markowitz)
        self.weights_window = weights_window

    def get_universe(self, date, prices):
        # Seleção fixa, mas pode ser qualquer função
        return [a for a in self.ativos_usuario if a in prices.columns]

    def get_weights(self, ativos, prices_window, date):
        # Janela de preços para cálculo dos pesos
        if date not in prices_window.index:
            end_idx = len(prices_window) - 1
        else:
            end_idx = prices_window.index.get_loc(date)
        start_idx = max(0, end_idx - self.weights_window + 1)
        window = prices_window.iloc[start_idx:end_idx+1]
        # Sempre chama a função de pesos, mesmo se a janela for pequena
        return self.weights_fn(ativos, window, date)
