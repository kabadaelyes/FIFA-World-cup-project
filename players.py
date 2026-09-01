"""
HIT140 Assessment 2 - Individual Analytic Task
Question: Among World Cup 2026 defenders, what proportion of defensive duels
did they win, and does this differ between younger and older defenders?
"""

import pandas as pd
import numpy as np
from scipy import stats

# STEP 1: Load the data
df = pd.read_csv("world-cup-players-2026-stats.csv")

# STEP 2: Data wrangling - keep only defenders who contested at least one duel
defenders = df[df["position"] == "Defender"].copy()
defenders = defenders[defenders["duels_total_overall"] > 0].copy()

print(f"Number of defenders with at least 1 duel: {len(defenders)}")

# STEP 3: Data preparation - create the proportion variable
defenders["duel_win_rate"] = (
    defenders["duels_won_total_overall"] / defenders["duels_total_overall"]
)