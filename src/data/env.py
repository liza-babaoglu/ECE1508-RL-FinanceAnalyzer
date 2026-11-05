import gymnasium as gym
import numpy as np
import pandas as pd


def build_portfolio_menu(num_assets: int):
    """Build portfolios: single assets or 50/50 pairs or equal weight."""

    if isinstance(num_assets, (bool, np.bool_)) or not isinstance(num_assets, (int, np.integer)) or num_assets < 1:
        raise ValueError("num_assets must be a positive integer.")

    portfolio_candidates = list(np.eye(num_assets, dtype=np.float64))

    # Type 1: 50/50 pairs for every 2 assets
    for first_asset_index in range(num_assets):
        for second_asset_index in range(first_asset_index + 1, num_assets):
            pair_weights = np.zeros(num_assets, dtype=np.float64)
            pair_weights[first_asset_index] = 0.5
            pair_weights[second_asset_index] = 0.5
            portfolio_candidates.append(pair_weights)

    # Type 2: Equal Weight
    portfolio_candidates.append(
        np.full(num_assets, 1.0 / num_assets, dtype=np.float64)
    )

    # Type 3: Unique assets, remove duplicates for the case of 1 or 2 assets.
    unique_portfolios = []
    for candidate_weights in portfolio_candidates:
        if not any(np.array_equal(candidate_weights, existing_weights)
                   for existing_weights in unique_portfolios):
            unique_portfolios.append(candidate_weights)

    return np.stack(unique_portfolios)


class PortfolioEnv(gym.Env):
    def __init__(
        self,
        prices: pd.DataFrame,
        start_date: str,
        end_date: str,
        window_size: int = 30,
        transaction_cost_rate: float = 0.001,
        episode_mode: str = "full",
        num_trading_days: int | None = None,

        
        action_score_scale: float = 5.0,    #Used for scaling the [-1,1], this number probarbly need to be tuned.
        observation_return_scale: float = 100.0,  # Scale log returns used as market observations.
        reward_scale: float = 100.0,  # Reserved for the portfolio log reward in step().


        # NOTE：Full-period mode
        # Each episode proceeds chronologically through all eligible trading dates in the complete training period.

        # At each decision, the agent observes the most recent window_size daily returns available at the decision day's close, 
        # together with the current portfolio weights. The environment rebalances to the target weights, deducts transaction costs, 
        # and calculates the return through the next trading day's close.

        # After each step, the observation window advances to the next trading date, 
        # dropping the oldest return and adding the newly observed return. 
        # The episode ends after settling the final return in the training period.

        # Calling reset() restarts the environment from the same starting point with a portfolio value of 1.0 and equal asset weights. 
        # The model's learned parameters are retained.

        # NOTE: Random mode
        # At each reset(), the environment randomly selects a valid starting point within the training period. 
        # The start must have sufficient preceding price history to construct the observation window and enough subsequent data to complete num_trading_days consecutive trading steps. 
        # The entire evaluated segment must remain within the training period.

        # At each decision, the agent uses the most recent window_size daily returns and the current portfolio weights to choose its target allocation.
        # The environment rebalances, deducts costs, and settles the return. The observation window advances with the trading dates.

        # The episode ends after the specified number of trading steps. The next reset() samples a new starting point and restores the portfolio value to 1.0 with equal asset weights. 
        # Episodes may overlap, and the model's learned parameters are retained.
                
    ):
        super().__init__()


        # region Error and boundary Checking:

        self.prices = prices.copy()

        if self.prices.empty:
            raise ValueError("prices can't be empty.")

        if self.prices.index.hasnans:
            raise ValueError("Trading dates must not contain missing values.")


        # NOTE: This can be deleted, most likely not needed but just in case
        if not self.prices.index.is_monotonic_increasing:
            raise ValueError("Trading dates must be sorted in ascending order.")

        if not self.prices.index.is_unique:
            raise ValueError("Trading dates must not contain duplicates.")

        if not self.prices.columns.is_unique:
            raise ValueError("Asset names must not contain duplicates.")
        
        if not np.isfinite(transaction_cost_rate) or not (
            0 <= transaction_cost_rate < 1
        ):
            raise ValueError("transaction_cost_rate must be in [0, 1).")

        # endregion

        # region define self attributes

        self.window_size = window_size
        self.transaction_cost_rate = transaction_cost_rate

        self.asset_names = self.prices.columns.tolist()
        self.num_assets = len(self.asset_names)

        # position means 
        # self.num_positions = self.num_assets + 1
        # self.cash_position_index = self.num_assets
        
        #Currently no cash position
        self.num_positions = self.num_assets

        self.price_values = self.prices.to_numpy(dtype=np.float64)

        if not np.isfinite(self.price_values).all():
            raise ValueError("Prices must not contain missing or infinite values.")
        
        if (self.price_values <= 0).any():
            raise ValueError("All prices must be positive.")

        # endregion

        # Defining returns

        # Keep simple returns for portfolio growth and weight drift calculations.
        self.daily_asset_returns = self.price_values[1:] / self.price_values[:-1] - 1.0

        # Log returns are observation features only; they do not replace settlement returns.
        self.daily_asset_log_returns = np.log1p(self.daily_asset_returns)

        # Defining date ranges that are allowed for trading
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)

        # NOTE: Can be deleted
        if pd.isna(self.start_date) or pd.isna(self.end_date):
            raise ValueError("Trading boundaries must be valid dates.")

        # NOTE: Can be deleted
        if self.start_date >= self.end_date:
            raise ValueError("Start date must be earlier than end date.")



        # region: Calculate tradinng row

        # NOTE: first_available_trading_row and last_available_trading_row: 
        # The position of first day that has a reward under the allowed date range
        # The position of last that that has a reward under the allowed date range
        self.first_available_trading_row = int(self.prices.index.searchsorted(self.start_date,side = "left"))
        self.last_available_trading_row = int(self.prices.index.searchsorted(self.end_date,side = "left")) - 1

        self.num_available_trading_days =  self.last_available_trading_row - self.first_available_trading_row + 1
        if self.num_available_trading_days <= 0 :
             raise ValueError("No trading dates exist in the requested period.")

        required_history_rows = self.window_size + 1

        if self.first_available_trading_row < required_history_rows:
            raise ValueError(
                f"Window size {self.window_size} requires at least "
                f"{required_history_rows} historical price rows before "
                "the first available trading date, but only "
                f"{self.first_available_trading_row} are available."
            )

        # endregion


        # region -- Define Episode Mode

        if episode_mode not in ["full","random"]:
              raise ValueError("episode_mode must be 'full' or 'random'.")

        self.episode_mode = episode_mode

        if self.episode_mode == "full":
            if num_trading_days is not None:
                raise ValueError(
                    "Use num_trading_days=None in full mode. "
                    "Trading days are calculated from the date range."
                )

            self.num_trading_days = self.num_available_trading_days


        else:
            if type(num_trading_days) is not int or num_trading_days < 1:
                raise ValueError("Random mode requires a positive integer num_trading_days.")

            if num_trading_days > self.num_available_trading_days:
                raise ValueError(
                    f"Requested {num_trading_days} trading days, but only "
                    f"{self.num_available_trading_days} are available."
                )

            self.num_trading_days = num_trading_days
        # endregion



        #------------------------------------------------------------------
        # region -- action and observation


        self.action_score_scale = action_score_scale

        # NOTE: Can be deleted
        if not np.isfinite(observation_return_scale) or observation_return_scale <= 0:
            raise ValueError("Observation return scale must be positive and finite.")
        if not np.isfinite(reward_scale) or reward_scale <= 0:
            raise ValueError("Reward scale must be positive and finite.")

        # Observation scaling changes the input magnitude, not account returns.
        self.observation_return_scale = observation_return_scale
        self.reward_scale = reward_scale

        ### ACTION SPACE FOR GOD SAKE!!!
        self.action_space = gym.spaces.Box(low = -1.0, high = 1.0, shape = (self.num_positions,), dtype = np.float32)

        self.observation_size = self.window_size * self.num_assets + self.num_positions
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_size,),
            dtype=np.float32,
        )

        # endregion

    
    
    #Some Helper Functions:
    def reset(self, *, seed=None, options=None):
        """Start an episode with an equally invested portfolio and its first observation."""
        # Seed Gymnasium's generator before selecting a random episode start.
        super().reset(seed=seed)
        self._select_episode_range()

        # This row is the END date of the next return to be earned.
        # The initial decision and observation stop at the preceding price row.
        self.current_trading_row = self.first_trading_row
        self.current_step = 0

        # Net asset value starts at 1.0. There is no cash position.
        # Equal holdings are the initial endowment; reset itself charges no entry fee.
        self.portfolio_value = 1.0
        self.current_weights = np.full(
            self.num_positions,
            1.0 / self.num_assets,
            dtype=np.float64,
        )

        observation = self._get_observation()
        info = {
            "date": self.prices.index[self.current_trading_row - 1],
            "first_return_date": self.prices.index[self.first_trading_row],
            "last_return_date": self.prices.index[self.last_trading_row],
            "num_trading_days": self.num_trading_days,
            "portfolio_value": self.portfolio_value,
            "weights": self.current_weights.copy(),
        }
        return observation, info


    def step(self, action):
        """Convert the algorithm action into target weights, then use the shared settlement logic."""


        target_weights = self._action_to_weights(action)
        return self._step_weights(target_weights)

    def _step_weights(self, target_weights): # Adjust weight based on current close price, and see return till next day.
        """Settle one trading day from explicit weights, including exact zero weights."""

        # NOTE: General Steps
        # 1. calculate transaction cost to reach the target weight
        # 2. calculate how much the portfolio grows in 1 day
        # 3. update portfolio value and the reward
        # 4. Update the current weight since the portfolio value and close price changes

        # NOTE: 1. Check whether the step is possible
        # NOTE: When the current trading row > last_trading_row, impossible to trade
        if not hasattr(self, "current_trading_row"):
            raise RuntimeError("Call reset() before step().")

        if self.current_trading_row > self.last_trading_row:
            raise RuntimeError(
                "The episode has ended. Call reset() before another step()."
            )

        # target_weights = self._action_to_weights(action)

        target_weights = np.asarray(target_weights, dtype=np.float64)

        if target_weights.shape != (self.num_positions,):
            raise ValueError(
                f"Target weights must have shape ({self.num_positions},), "
                f"but received {target_weights.shape}."
            )
        if not np.isfinite(target_weights).all():
            raise ValueError("Target weights must be finite.")
        if (target_weights < 0).any():
            raise ValueError("Target weights must be nonnegative.")

        total_weight = float(target_weights.sum())
        if not np.isclose(total_weight, 1.0, rtol=0.0, atol=1e-6):
            raise ValueError("Target weights must sum to one.")

        target_weights = target_weights / total_weight

        # -------------------------------------------------------------------------------
        # NOTE: 2.Calculate transaction cost ratio = how much weight change * cost rate
        #       transaction_cost = transaction_cost_ratio * previous_ portfolio_value

        turnover_ratio = float(np.abs(target_weights - self.current_weights).sum()) #[0.5,0.5] -> [0.6,0.4] = 0.1 + 0.1 = 0.2

        transaction_cost_ratio = self.transaction_cost_rate * turnover_ratio

        if transaction_cost_ratio >= 1.0:
            raise ValueError("Transaction costs would consume the portfolio.")

        previous_portfolio_value = self.portfolio_value

        transaction_cost = previous_portfolio_value * transaction_cost_ratio

        # -------------------------------------------------------------------------------
        # NOTE: 3. Calculate the simple return for this step, next day - current day
        #    portfolio_growth_factor = target_weight @ how much asset grows in 1 day, ex: [0.5,0.5] @ [1.1,0.95] = 1.025

        asset_returns = self.daily_asset_returns[self.current_trading_row - 1] # For trading row t, the return is at row t-1
        asset_growth_factors = 1.0 + asset_returns  #Calculate the growth ex: [0.1,-0.05] -> [1.1,0.95]

        portfolio_growth_factor = float(target_weights @ asset_growth_factors) # Calculate the portfolio growth

        # -------------------------------------------------------------------------------
        # NOTE: 4. Update the portfolio value and calculate the reward

        next_portfolio_value = previous_portfolio_value * (1.0 - transaction_cost_ratio) * portfolio_growth_factor
        portfolio_log_return = np.log1p(-transaction_cost_ratio)+ np.log(portfolio_growth_factor)
        
        reward = portfolio_log_return * self.reward_scale

        # -------------------------------------------------------------------------------
        # NOTE: 5. Update the current weight, since the price changes
        self.current_weights = target_weights * asset_growth_factors / portfolio_growth_factor

        # -------------------------------------------------------------------------------
        # NOTE: 6. Update the portfolio value, move the counter
        self.portfolio_value = next_portfolio_value
        date = self.prices.index[self.current_trading_row]

        self.current_trading_row += 1
        self.current_step += 1

        # -------------------------------------------------------------------------------
        # 7. Return the result

        terminated = False
        truncated = self.current_trading_row > self.last_trading_row

        observation = self._get_observation()

        info = {
            "date": date,
            "current_step": self.current_step,
            "portfolio_value": self.portfolio_value,
            "net_return": float(np.expm1(portfolio_log_return)),
            "portfolio_log_return": float(portfolio_log_return),
            "turnover_ratio": turnover_ratio,
            "transaction_cost": transaction_cost,
            "transaction_cost_ratio": transaction_cost_ratio,
            "target_weights": target_weights.copy(),
            "weights": self.current_weights.copy(),
        }

        return observation, float(reward), terminated, truncated, info
    def _select_episode_range(self):        #Function used to select the start and end dates for each episode
        # if mode is full, first trading row is the first available trading row

        if self.episode_mode == "full":
            self.first_trading_row = self.first_available_trading_row
        else:
            # random mode

            # NOTE: Under random mode, the last row that allow gthe trading to be finished
            latest_start_row = self.last_available_trading_row - self.num_trading_days + 1

            # NOTE: Under random mode, the first trading day that's finished under this episode(random)
            self.first_trading_row = int(
            self.np_random.integers(
                low=self.first_available_trading_row,
                high=latest_start_row + 1,
                )
            )

        self.last_trading_row = self.first_trading_row + self.num_trading_days - 1

    def _action_to_weights(self,action): #Convert the action score into weights

        action = np.asarray(action, dtype=np.float64)

        if action.shape != (self.num_positions,):
            raise ValueError(
                f"Action must have shape ({self.num_positions},), "
                f"but received {action.shape}."
            )

        if not np.isfinite(action).all():
            raise ValueError("Action must not contain missing or infinite values.")

        bound_action = np.clip(action, -1.0, 1.0)
        action_scores = self.action_score_scale * bound_action
        action_scores = action_scores - np.max(action_scores)
        exponential_scores = np.exp(action_scores)
        target_weights = exponential_scores / exponential_scores.sum()

        return target_weights


    # Previous description: window days observation + current day + weight.
    def _get_observation(self):
        """Return historical scaled log returns and current asset weights."""
        history_end = self.current_trading_row - 1
        history_start = history_end - self.window_size

        # Previous version: unscaled simple returns as observation features.
        # historical_returns = self.daily_asset_returns[
        #     history_start:history_end
        # ]

        # Stop before the upcoming return: no future market data in the observation.
        historical_returns = self.daily_asset_log_returns[
            history_start:history_end
        ] * self.observation_return_scale
        # Do not clip: preserve the size of extreme historical market moves.

        if historical_returns.shape != (
            self.window_size,
            self.num_assets,
        ):
            raise ValueError("Not enough historical returns for observation.")


        flattened_returns = historical_returns.reshape(-1)


        observation = np.concatenate(
            [flattened_returns, self.current_weights]
        )

        return observation.astype(np.float32)

    
class DiscretePortfolioEnv(PortfolioEnv):

    def __init__(
        self,
        prices: pd.DataFrame,
        start_date: str,
        end_date: str,
        **kwargs,
    ):
        super().__init__(prices, start_date, end_date, **kwargs)

        self.portfolio_menu = build_portfolio_menu(self.num_assets)
        self.num_portfolios = len(self.portfolio_menu)

        self.action_space = gym.spaces.Discrete(self.num_portfolios)

    def _action_to_weights(self, action):
        
        if isinstance(action, (bool, np.bool_)) or not self.action_space.contains(action):
            raise ValueError(
                f"Action must be an integer portfolio index in "
                f"[0, {self.num_portfolios - 1}]."
            )

        portfolio_index = int(action)
        return self.portfolio_menu[portfolio_index].copy()
