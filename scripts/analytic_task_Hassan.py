"""
HIT140 Assessment 2 - Individual Analytic Task
Question: Among World Cup 2026 defenders, what proportion of defensive duels
did they win, and does this differ between younger and older defenders?
"""

import pandas as pd
import numpy as np
import os
from scipy import stats

# STEP 1: Load the data (path built relative to this script's location)
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "..", "data", "raw", "players.csv")
df = pd.read_csv(csv_path)

# STEP 2: Data wrangling - keep only defenders who contested at least one duel
defenders = df[df["position"] == "Defender"].copy()
defenders = defenders[defenders["duels_total_overall"] > 0].copy()

print(f"Number of defenders with at least 1 duel: {len(defenders)}")

# STEP 3: Data preparation - create the proportion variable
defenders["duel_win_rate"] = (
    defenders["duels_won_total_overall"] / defenders["duels_total_overall"]
)

# STEP 4: Sampling from the population of defenders
sample_size = 100
sample = defenders.sample(n=min(sample_size, len(defenders)), random_state=42)


# STEP 5: Split into two groups by age (median split)
median_age = sample["age"].median()
younger = sample[sample["age"] <= median_age]["duel_win_rate"]
older = sample[sample["age"] > median_age]["duel_win_rate"]

print(f"\nMedian age used as split point: {median_age}")
print(f"Younger group n = {len(younger)}, Older group n = {len(older)}")

# STEP 6: Descriptive statistics
print("\n--- Descriptive statistics: duel win rate ---")
print("Overall sample:")
print(sample["duel_win_rate"].describe())

print("\nYounger defenders:")
print(younger.describe())

print("\nOlder defenders:")
print(older.describe())