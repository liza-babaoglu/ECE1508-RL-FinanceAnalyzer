# use this:
# python builtin_policies.py --algo ppo
# python builtin_policies.py --algo dqn

import argparse
from datetime import datetime
import os
import pandas as pd
from stable_baselines3 import DQN, PPO # can easily add more i think, but lets do research into how they work for the report
from src.data import config
from src.data.env import DiscretePortfolioEnv, PortfolioEnv

TIMESTEPS = 100000
SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["ppo", "dqn"], default="ppo")
    args = ap.parse_args()

    train_prices = pd.read_csv(config.PROCESSED_DATA_DIR / "train_prices.csv", index_col=0, parse_dates=True)

    # Training env: random 252-day episodes inside 2008-2018
    # PPO: continuous weights (PortfolioEnv), DQN: pick 1 of 29 portfolios (DiscretePortfolioEnv)
    env_class = DiscretePortfolioEnv if args.algo == "dqn" else PortfolioEnv

    # instead of hardcoding  num_trading_days=252 use config.TRADING_DAYS_PER_YEAR
    # instead of hardcoding "2008-01-01", "2019-01-01",
    start_date, end_date = config.TRAIN_SPLIT     # ("2008-01-01", "2019-01-01")

    train_env = env_class(train_prices, start_date, end_date, episode_mode="random", num_trading_days=config.TRADING_DAYS_PER_YEAR)

    if args.algo == "ppo":
        model = PPO("MlpPolicy", train_env, seed=SEED, device="cpu")
    else:
        model = DQN("MlpPolicy", train_env, seed=SEED, device="cpu",
                    buffer_size=100000, learning_starts=1000, exploration_fraction=0.2)

    print(f"Started training {args.algo.upper()} ...")
    model.learn(total_timesteps=TIMESTEPS)  # stable_baselines3 built-in training loop !!!

    # fixing the overwritting issue, but remember to delete to save space
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")    # e.g. 20261104-2056
    run_dir = "runs/" + args.algo                         # runs/ppo or runs/dqn
    os.makedirs(run_dir, exist_ok=True)
    model_path = f"{run_dir}/{args.algo}_{timestamp}"     # runs/ppo/ppo_20261104-2200.zip
    model.save(model_path)
    print(f"Saved to {model_path}.zip")



    ## TESTing PPO on validation --- sanity check
    val_prices = pd.read_csv(config.PROCESSED_DATA_DIR / "val_prices.csv", index_col=0, parse_dates=True)
    val_start, val_end = config.VAL_SPLIT
    val_env = env_class(val_prices, val_start, val_end)

    obs, info = val_env.reset(seed=0)
    done = False
    while not done:
        action, state = model.predict(obs, deterministic=True)   # no exploration
        obs, reward, terminated, truncated, info = val_env.step(action)
        done = terminated or truncated

    print(f"{args.algo.upper()} validation final value:", round(info["portfolio_value"], 3))


if __name__ == "__main__":
    main()


# Started training PPO ...
# Saved to runs/ppo/ppo_20261104-2205.zip
# PPO validation final value: 0.833 
# this is pretty low, it says we lost money... maybe because of the validation set being during the covid crash? or we need more training... 

### ran baselines on validation years -- check baselines_classes
# PPO  did worse than doing nothing (Buy-and-Hold) and worse than random. The current PPO hasn’t learned a useful strategy.

