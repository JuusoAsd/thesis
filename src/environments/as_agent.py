from src.environments.mm_env import AgentBaseClass


class ASAgent(AgentBaseClass):
    """
    Agent that behaves based on Avellaneda-Stoikov model
    Requires trades and order book midprice and uses those to calculate the indifference price and spread
    """

    def __init__(self, env, alpha=0.5, kappa=0.5):
        super().__init__(env)
        self.alpha = alpha
        self.kappa = kappa
        self.indifference_price = 0
        self.spread = 0

    def reset(self):
        self.indifference_price = 0
        self.spread = 0
