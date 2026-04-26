"""
Fixed parameters for the ML Mean Reversion Strategy.
These are risk constraints, not tunable parameters.
"""

# Portfolio risk constraints
MAX_POSITIONS = 3  # Maximum concurrent positions
MAX_WEIGHT = 0.5  # Maximum weight per position (50%)

# Backtest parameters
INITIAL_CAPITAL = 100_000
TRANSACTION_COST = 0.001  # 0.1% one-way, 0.2% round-trip

# ML parameters
ML_PROB_THRESHOLD = 0.55  # Higher threshold filters to higher-confidence trades only

# Walk-forward parameters
TEST_WINDOW_MONTHS = 12
MIN_TRAIN_YEARS = 3  # Sufficient for LR with ~4k samples