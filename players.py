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