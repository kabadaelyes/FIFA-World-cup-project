"""
HIT140 Assessment 3 - Objective 2, Linear Regression 2.1
Predict the goal difference (home goals - away goals) of each
FIFA World Cup 2026 match from 8 pre-match explanatory variables.

Dataset: data/world_cup_prediction/match_prediction_features.csv (104 matches)

Every explanatory variable is a home-minus-away difference and is known
before kickoff. In-match stats (shots, possession, xG of that match) are
not used, because they would leak the result.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---------------------------------------------------------
# STEP 1: Load the data
# ---------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "..", "data", "world_cup_prediction",
                         "match_prediction_features.csv")
out_dir = os.path.join(script_dir, "..", "data", "processed", "model_2_1")
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(data_path)
print(f"Matches loaded: {len(df)}")

# ---------------------------------------------------------
# STEP 2: Build the target and the 8 explanatory variables
# ---------------------------------------------------------
# Target: goal difference from the home team's point of view.
# Matches decided on penalties count as 0 (scores exclude shootouts).
y = df["home_score"] - df["away_score"]

X = pd.DataFrame({
    "elo_diff":        df["home_elo"] - df["away_elo"],
    "squad_value_diff": (df["home_squad_avg_value_eur"]
                         - df["away_squad_avg_value_eur"]) / 1_000_000,  # in EUR millions
    "prev_scored_diff": df["home_prev_avg_goals_scored"] - df["away_prev_avg_goals_scored"],
    "prev_conceded_diff": df["home_prev_avg_goals_conceded"] - df["away_prev_avg_goals_conceded"],
    "caps_diff":       df["home_squad_total_caps"] - df["away_squad_total_caps"],
    "age_diff":        df["home_squad_avg_age"] - df["away_squad_avg_age"],
    "host_diff":       df["home_is_host"] - df["away_is_host"],
    "rest_days_diff":  df["home_rest_days"] - df["away_rest_days"],
})

model_data = X.copy()
model_data.insert(0, "match_id", df["match_id"])
model_data["goal_difference"] = y
model_data.to_csv(os.path.join(out_dir, "model_2_1_dataset.csv"), index=False)
print(f"Dataset shape: {X.shape[0]} rows x {X.shape[1]} explanatory variables")
print(f"Missing values: {int(model_data.isna().sum().sum())}")

# ---------------------------------------------------------
# STEP 3: Descriptive statistics and correlation with the target
# ---------------------------------------------------------
print("\n--- Descriptive statistics ---")
desc = model_data.drop(columns="match_id").describe().T
print(desc[["mean", "std", "min", "max"]].round(3))
desc.to_csv(os.path.join(out_dir, "descriptive_statistics.csv"))

print("\n--- Correlation with goal difference ---")
corr_with_y = X.corrwith(y).sort_values(key=abs, ascending=False)
print(corr_with_y.round(3))

# ---------------------------------------------------------
# STEP 4: Multicollinearity check (VIF)
# VIF for each variable = 1 / (1 - R^2) when it is regressed on the other 7.
# VIF above 5 would suggest a problem.
# ---------------------------------------------------------
vif = {}
for col in X.columns:
    others = X.drop(columns=col)
    r2 = LinearRegression().fit(others, X[col]).score(others, X[col])
    vif[col] = 1 / (1 - r2)
vif = pd.Series(vif, name="VIF").sort_values(ascending=False)
print("\n--- Variance Inflation Factors ---")
print(vif.round(2))
vif.to_csv(os.path.join(out_dir, "vif.csv"))

# ---------------------------------------------------------
# STEP 5: Train/test split (80/20)
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"\nTraining matches: {len(X_train)}, Test matches: {len(X_test)}")

# ---------------------------------------------------------
# STEP 6: Fit the linear regression model on the training set
# ---------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ---------------------------------------------------------
# STEP 7: Evaluate on the unseen test set
# ---------------------------------------------------------
y_pred = model.predict(X_test)
n_test, k = len(y_test), X.shape[1]
r2 = r2_score(y_test, y_pred)
adj_r2 = 1 - (1 - r2) * (n_test - 1) / (n_test - k - 1)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

# Baseline: always predict the training mean goal difference
baseline_pred = np.full(n_test, y_train.mean())
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))

train_r2 = model.score(X_train, y_train)

print("\n--- Model evaluation ---")
print(f"Train R^2:          {train_r2:.3f}")
print(f"Test R^2:           {r2:.3f}")
print(f"Test adjusted R^2:  {adj_r2:.3f}")
print(f"Test RMSE:          {rmse:.3f} goals")
print(f"Test MAE:           {mae:.3f} goals")
print(f"Baseline RMSE:      {baseline_rmse:.3f} goals (predicting the mean)")

# ---------------------------------------------------------
# STEP 7b: Repeated 5-fold cross-validation
# With only 104 matches, one 21-match test set is very sensitive to which
# matches land in it. Cross-validation tests every match once per repeat
# and averages over 20 repeats, giving a more reliable estimate.
# ---------------------------------------------------------
cv = RepeatedKFold(n_splits=5, n_repeats=20, random_state=42)
cv_r2 = cross_val_score(LinearRegression(), X, y, cv=cv, scoring="r2")
cv_rmse = -cross_val_score(LinearRegression(), X, y, cv=cv,
                           scoring="neg_root_mean_squared_error")
cv_mae = -cross_val_score(LinearRegression(), X, y, cv=cv,
                          scoring="neg_mean_absolute_error")
print("\n--- Repeated 5-fold cross-validation (20 repeats) ---")
print(f"CV R^2:   {cv_r2.mean():.3f} (sd {cv_r2.std():.3f})")
print(f"CV RMSE:  {cv_rmse.mean():.3f} goals")
print(f"CV MAE:   {cv_mae.mean():.3f} goals")
print(f"Std dev of goal difference: {y.std():.3f} goals")

metrics = pd.DataFrame({
    "metric": ["train_R2", "test_R2", "test_adj_R2", "test_RMSE",
               "test_MAE", "baseline_RMSE", "cv_R2_mean", "cv_R2_sd",
               "cv_RMSE_mean", "cv_MAE_mean"],
    "value": [train_r2, r2, adj_r2, rmse, mae, baseline_rmse,
              cv_r2.mean(), cv_r2.std(), cv_rmse.mean(), cv_mae.mean()],
})
metrics.to_csv(os.path.join(out_dir, "evaluation_metrics.csv"), index=False)

# ---------------------------------------------------------
# STEP 8: Coefficients with standard errors and p-values
# (ordinary least squares on all 104 matches, for interpretation)
# ---------------------------------------------------------
full = LinearRegression().fit(X, y)
X_design = np.column_stack([np.ones(len(X)), X.values])
residuals = y.values - full.predict(X)
dof = len(X) - X_design.shape[1]
sigma2 = residuals @ residuals / dof
se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X_design.T @ X_design)))
coefs = np.concatenate([[full.intercept_], full.coef_])
t_vals = coefs / se
p_vals = 2 * stats.t.sf(np.abs(t_vals), dof)

coef_table = pd.DataFrame({
    "coefficient": coefs, "std_error": se, "t_value": t_vals, "p_value": p_vals,
}, index=["intercept"] + list(X.columns))
full_r2 = full.score(X, y)
print("\n--- Coefficients (all 104 matches) ---")
print(coef_table.round(4))
print(f"R^2 on all matches: {full_r2:.3f}")
coef_table.to_csv(os.path.join(out_dir, "coefficients.csv"))

# ---------------------------------------------------------
# STEP 9: Visualisations
# ---------------------------------------------------------
# Actual vs predicted (test set)
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(y_test, y_pred, color="#1F6FD1", alpha=0.8)
lims = [min(y_test.min(), y_pred.min()) - 0.5, max(y_test.max(), y_pred.max()) + 0.5]
ax.plot(lims, lims, linestyle="--", color="grey", label="Perfect prediction")
ax.set_xlabel("Actual goal difference")
ax.set_ylabel("Predicted goal difference")
ax.set_title("Model 2.1: Actual vs Predicted (test set)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "actual_vs_predicted.png"), dpi=200)
plt.close()

# Residuals vs fitted (all matches) to check linearity and constant variance
fitted = full.predict(X)
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(fitted, residuals, color="#E8720C", alpha=0.8)
ax.axhline(0, linestyle="--", color="grey")
ax.set_xlabel("Fitted goal difference")
ax.set_ylabel("Residual")
ax.set_title("Model 2.1: Residuals vs Fitted")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "residuals_vs_fitted.png"), dpi=200)
plt.close()

print(f"\nOutputs saved to: {os.path.abspath(out_dir)}")
