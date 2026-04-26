"""
Fixed parameters for the ML Mean Reversion Strategy.
These are risk constraints, not tunable parameters.
"""

# Portfolio risk constraints
MAX_POSITIONS = 3  # Maximum concurrent positions
MAX_WEIGHT = 0.5  # Maximum weight per position (50%)

# Backtest parameters
INITIAL_CAPITAL = 100_000
# Transaction cost is modeled as a one-way cost (entry OR exit).
# The backtest applies round-trip cost as: net_return = gross_return - 2 * TRANSACTION_COST
# Default: 0.05% one-way → 0.10% round-trip.
TRANSACTION_COST = 0.0005

# ML parameters
ML_PROB_THRESHOLD = 0.55  # Higher threshold filters to higher-confidence trades only

# Walk-forward parameters
TEST_WINDOW_MONTHS = 12
MIN_TRAIN_YEARS = 3  # Sufficient for LR with ~4k samples