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