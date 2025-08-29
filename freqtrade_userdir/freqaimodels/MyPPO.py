from freqtrade.freqai.prediction_models.ReinforcementLearner import ReinforcementLearner
from freqtrade.freqai.RL.Base5ActionRLEnv import Base5ActionRLEnv, Actions, Positions


class MyPPO(ReinforcementLearner):
    """
    Custom RL model for FreqAI. Use with:
      --freqaimodel MyPPO --strategy MyRLStrategy
    Configure PPO, policy, and policy_kwargs via freqai.model_training_parameters.
    """

    class MyRLEnv(Base5ActionRLEnv):
        """Minimal reward: realized/unrealized pnl minus tiny holding penalty."""

        def calculate_reward(self, action: int) -> float:
            if not self._is_valid(action):
                return -2.0

            pnl = self.get_unrealized_profit()

            # Encourage valid closes when profitable
            if action == Actions.Long_exit.value and self._position == Positions.Long:
                return float(pnl)
            if action == Actions.Short_exit.value and self._position == Positions.Short:
                return float(pnl)

            # Small penalty for sitting in positions
            hold_penalty = 0.0005 if self._position != Positions.Neutral else 0.0
            return float(pnl - hold_penalty)


