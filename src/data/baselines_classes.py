import numpy as np

"""
BuyAndHold:
Buy equal amounts (1/N) of every asset on day one, then never trade again. where N is the number of assets

"Turnover" = how much of the portfolio you trade on a given day = 0 = no trade

Note: the weights do NOT stay equal. We keep the same shares, so assets that go
up become a bigger part of the portfolio and assets that go down become smaller.
"""
class BuyAndHold:
    def __init__(self):
        self.name = "Buy-and-Hold"

    def get_weights(self, env):
        return env.current_weights.copy()  # NO TRADING, Buy equally and just wait


"""
EqualWeight:
Keep the portfolio at equal amounts (1/N) of every asset.
Every rebalance_days trading days (default 21, about once a month) it rebalances:
SELL what went up and BUY more of what went DOWN, so we rebalance to 1/N.
On all other days it holds, like Buy-and-Hold.

"Turnover" = small on rebalance days; 0 on all other days -> very low on average

Note: between rebalance days the weights drift away from 1/N as prices move,
then get reset to 1/N on the next rebalance day.
"""
class EqualWeight:
    """Rebalance back to 1/N in every asset every `rebalance_days` trading days."""
    def __init__(self, rebalance_days: int = 21): # 252 trading days per year / 12 months = 21 trading days per month
        self.rebalance_days = rebalance_days
        self.name = f"Equal-Weight ({rebalance_days}d)"

    def get_weights(self, env):
        if env.current_step % self.rebalance_days == 0:
            # array([0.143, 0.143, 0.143, 0.143, 0.143, 0.143, 0.143]) bc we have 7 assets
            return np.ones(env.num_assets) / env.num_assets # 1/N
        return env.current_weights.copy() # On all other days it holds, like Buy-and-Hold.


"""
RandomWeights:
Pick a NEW RANDOM portfolio every day, ignoring prices completely.
The weights need to be positive (or very very very close to zero).
The weights need to sum to 1.
Hence, use Dirichlet distribution: produces random vectors of positive numbers that sum to 1.

"Turnover" = very high, because most of the portfolio changes every day
"""
class RandomWeights:
    """New random weights every day, drawn uniformly over all portfolios (Dirichlet(1,...,1))."""
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.name = "Random Weights"

    def get_weights(self, env):
        return self.rng.dirichlet(np.ones(env.num_assets))




# TESTing
# Quick test: python -m src.data.baselines_classes   (run from the repo root)
# env = PortfolioEnv(prices, "2008-01-01", "2019-01-01")
# self,
#         prices: pd.DataFrame, # no default - need to fill 
#         start_date: str,      # no default - need to fill 
#         end_date: str,        # no default - need to fill
#         window_size: int = 30,
#         transaction_cost_rate: float = 0.001,
#         episode_mode: str = "full",
#         num_trading_days: int | None = None,

if __name__ == "__main__":
    import pandas as pd
    from src.data import config
    from src.data.env import PortfolioEnv

    # without the index_col=0, Date is treated as if the 8th asset
    prices = pd.read_csv(config.PROCESSED_DATA_DIR / "train_prices.csv", index_col=0, parse_dates=True)

    # so policy is not in the env, so we need to pass it in
    # next: do it for the PPO etc...
    for policy in [BuyAndHold(), EqualWeight(21), RandomWeights()]:
        env = PortfolioEnv(prices, "2008-01-01", "2019-01-01")
        env.reset(seed=0)
        done = False
        while not done:
            target = policy.get_weights(env)  # baseline picks today's weights
            obs, reward, terminated, truncated, info = env._step_weights(target)
            done = terminated or truncated
        print(policy.name, "final value:", round(info["portfolio_value"], 3))


# from step_weighs():
# info = {
        #     "date": date,
        #     "current_step": self.current_step,
        #     "portfolio_value": self.portfolio_value, ******* THIS!
        #     "net_return": float(np.expm1(portfolio_log_return)), (THIS IS ONLY FOR THE CURRENT STEP, NOT CUMULATIVE)
        #     "portfolio_log_return": float(portfolio_log_return),
        #     "turnover_ratio": turnover_ratio,
        #     "transaction_cost": transaction_cost,
        #     "transaction_cost_ratio": transaction_cost_ratio,
        #     "target_weights": target_weights.copy(),
        #     "weights": self.current_weights.copy(),
        # }

        # return observation, float(reward), terminated, truncated, info

# reasonable:
# (.venv) (base) lizababaoglu@Lizas-MacBook-Pro ECE1508-RL-FinanceAnalyzer % python -m src.data.baselines_classes
# Buy-and-Hold final value: 1.567
# Equal-Weight (21d) final value: 1.64
# Random Weights final value: 0.157