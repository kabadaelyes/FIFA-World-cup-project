
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold, cross_validate, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==================================================
# 1. PATHS AND SETTINGS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_DIR
    / "data"
    / "world_cup_prediction"
    / "match_prediction_features.csv"
)

OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "model_2_2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "goals_scored"

# Exactly eight explanatory variables.
FEATURES = [
    "team_elo",
    "team_squad_value_millions",
    "team_is_host",
    "team_rest_days",
    "team_fifa_ranking",
    "opponent_fifa_ranking",
    "venue_elevation",
    "referee_average_cards",
]


# ==================================================
# 2. LOAD DATA
# ==================================================

print("Loading match dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Original dataset shape: {df.shape}")

required_source_columns = [
    "match_id",
    "home_team_name",
    "away_team_name",
    "home_score",
    "away_score",
    "home_elo",
    "away_elo",
    "home_fifa_rank",
    "away_fifa_rank",
    "home_squad_avg_value_eur",
    "away_squad_avg_value_eur",
    "home_is_host",
    "away_is_host",
    "home_rest_days",
    "away_rest_days",
    "venue_elevation_meters",
    "referee_avg_cards",
]

missing_source_columns = [
    col for col in required_source_columns if col not in df.columns
]

if missing_source_columns:
    raise ValueError(
        "The input CSV is missing these required columns:\n"
        + "\n".join(missing_source_columns)
    )

if df["match_id"].isna().any():
    raise ValueError("The dataset contains missing match IDs.")

if df["match_id"].duplicated().any():
    raise ValueError("Expected one row per match in the source dataset.")

if df[["home_score", "away_score"]].isna().any().any():
    raise ValueError("The dataset contains missing match scores.")


# ==================================================
# 3. CREATE TWO TEAM-LEVEL ROWS PER MATCH
# ==================================================

rows = []

for _, match in df.iterrows():

    common_features = {
        "match_id": match["match_id"],
        "venue_elevation": match["venue_elevation_meters"],
        "referee_average_cards": match["referee_avg_cards"],
    }

    # Home team perspective.
    rows.append({
        **common_features,
        "team": match["home_team_name"],
        "opponent": match["away_team_name"],
        "team_elo": match["home_elo"],
        "team_squad_value_millions":
            match["home_squad_avg_value_eur"] / 1_000_000,
        "team_is_host": match["home_is_host"],
        "team_rest_days": match["home_rest_days"],
        "team_fifa_ranking": match["home_fifa_rank"],
        "opponent_fifa_ranking": match["away_fifa_rank"],
        TARGET: match["home_score"],
    })

    # Away team perspective.
    rows.append({
        **common_features,
        "team": match["away_team_name"],
        "opponent": match["home_team_name"],
        "team_elo": match["away_elo"],
        "team_squad_value_millions":
            match["away_squad_avg_value_eur"] / 1_000_000,
        "team_is_host": match["away_is_host"],
        "team_rest_days": match["away_rest_days"],
        "team_fifa_ranking": match["away_fifa_rank"],
        "opponent_fifa_ranking": match["home_fifa_rank"],
        TARGET: match["away_score"],
    })

model_df = pd.DataFrame(rows)

assert len(FEATURES) == 8
assert len(model_df) == 2 * len(df)

print(f"Constructed dataset shape: {model_df.shape}")
print(f"Unique matches: {model_df['match_id'].nunique()}")
print(f"Number of predictors: {len(FEATURES)}")

model_df.to_csv(
    OUTPUT_DIR / "model_2_2_dataset.csv",
    index=False,
)


# ==================================================
# 4. CLEAN AND CHECK DATA
# ==================================================

numeric_columns = FEATURES + [TARGET]

for column in numeric_columns:
    model_df[column] = pd.to_numeric(
        model_df[column],
        errors="coerce",
    )

print("\nMissing values before cleaning:")
print(model_df[numeric_columns].isna().sum())

# Do not invent values to replace missing observations.
model_df = model_df.dropna(
    subset=["match_id"] + numeric_columns
).copy()

if model_df.empty:
    raise ValueError("No usable rows remain after cleaning.")

if model_df["match_id"].nunique() < 5:
    raise ValueError("At least five unique matches are required.")

print(f"\nUsable dataset shape: {model_df.shape}")
print(f"Usable unique matches: {model_df['match_id'].nunique()}")

# Save descriptive statistics.
model_df[FEATURES + [TARGET]].describe().T.to_csv(
    OUTPUT_DIR / "descriptive_statistics.csv"
)

# Save Pearson correlations.
model_df[FEATURES + [TARGET]].corr(
    method="pearson"
).to_csv(
    OUTPUT_DIR / "correlations.csv"
)


# ==================================================
# 5. EXPLORATORY DATA ANALYSIS
# ==================================================

# Distribution of goals scored.
plt.figure(figsize=(8, 5))

plt.hist(
    model_df[TARGET],
    bins=np.arange(-0.5, 8.5, 1),
    edgecolor="black",
)

plt.xlabel("Goals scored by team")
plt.ylabel("Number of team-match observations")
plt.title("Distribution of Team Goals Scored")
plt.xticks(range(0, 9))
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "goals_distribution.png",
    dpi=300,
)
plt.close()


# Correlation heatmap.
correlation_matrix = model_df[FEATURES + [TARGET]].corr()

fig, ax = plt.subplots(figsize=(11, 8))

image = ax.imshow(
    correlation_matrix,
    aspect="auto",
    vmin=-1,
    vmax=1,
)

ax.set_xticks(range(len(correlation_matrix.columns)))
ax.set_xticklabels(
    correlation_matrix.columns,
    rotation=70,
    ha="right",
)

ax.set_yticks(range(len(correlation_matrix.index)))
ax.set_yticklabels(correlation_matrix.index)

fig.colorbar(image, ax=ax, label="Pearson correlation")
ax.set_title("Correlation Matrix: Model 2.2")
fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "correlation_heatmap.png",
    dpi=300,
)
plt.close(fig)


# ==================================================
# 6. GROUPED CROSS-VALIDATION
# ==================================================

X = model_df[FEATURES]
y = model_df[TARGET]
groups = model_df["match_id"]

number_of_folds = min(5, groups.nunique())
cv = GroupKFold(n_splits=number_of_folds)

# Both teams from the same match stay in the same fold.
models = {
    "Dummy baseline": DummyRegressor(strategy="mean"),

    "Linear Regression": make_pipeline(
        StandardScaler(),
        LinearRegression(),
    ),

    "Ridge Regression": make_pipeline(
        StandardScaler(),
        Ridge(alpha=1.0),
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        random_state=42,
    ),
}

results = []

print("\nEvaluating regression models...")

for name, model in models.items():

    scores = cross_validate(
        model,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "mse": "neg_mean_squared_error",
            "r2": "r2",
        },
        n_jobs=-1,
        error_score="raise",
    )

    fold_mae = -scores["test_mae"]
    fold_rmse = np.sqrt(-scores["test_mse"])
    fold_r2 = scores["test_r2"]

    results.append({
        "model": name,
        "mean_MAE": fold_mae.mean(),
        "std_MAE": fold_mae.std(),
        "mean_RMSE": fold_rmse.mean(),
        "std_RMSE": fold_rmse.std(),
        "mean_R2": fold_r2.mean(),
    })

metrics = pd.DataFrame(results).sort_values("mean_MAE")

metrics.to_csv(
    OUTPUT_DIR / "evaluation_metrics.csv",
    index=False,
)

print("\nGrouped cross-validation results:")
print(metrics.round(4).to_string(index=False))

best_name = metrics.iloc[0]["model"]
best_model = models[best_name]

print(f"\nBest model by mean MAE: {best_name}")


# ==================================================
# 7. OUT-OF-FOLD PREDICTIONS
# ==================================================

predictions = cross_val_predict(
    best_model,
    X,
    y,
    groups=groups,
    cv=cv,
    n_jobs=-1,
)

model_df["predicted_goals"] = predictions
model_df["residual"] = model_df[TARGET] - predictions

model_df[
    [
        "match_id",
        "team",
        "opponent",
        TARGET,
        "predicted_goals",
        "residual",
    ]
].to_csv(
    OUTPUT_DIR / "out_of_fold_predictions.csv",
    index=False,
)

mae = mean_absolute_error(y, predictions)
rmse = np.sqrt(mean_squared_error(y, predictions))
r2 = r2_score(y, predictions)

print("\nBest model out-of-fold metrics:")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")


# Actual vs predicted graph.
plt.figure(figsize=(7, 6))

plt.scatter(y, predictions, alpha=0.65)

lower = min(y.min(), predictions.min())
upper = max(y.max(), predictions.max())

plt.plot(
    [lower, upper],
    [lower, upper],
    linestyle="--",
)

plt.xlabel("Actual goals")
plt.ylabel("Out-of-fold predicted goals")
plt.title(f"Actual vs Predicted Goals — {best_name}")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "actual_vs_predicted.png",
    dpi=300,
)
plt.close()


# Residuals vs predicted graph.
plt.figure(figsize=(7, 5))

plt.scatter(
    predictions,
    model_df["residual"],
    alpha=0.65,
)

plt.axhline(0, linestyle="--")
plt.xlabel("Out-of-fold predicted goals")
plt.ylabel("Residual (actual - predicted)")
plt.title("Residuals vs Predicted Goals")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "residuals_vs_fitted.png",
    dpi=300,
)
plt.close()


# ==================================================
# 8. LINEAR REGRESSION COEFFICIENTS
# ==================================================

linear_model = make_pipeline(
    StandardScaler(),
    LinearRegression(),
)

linear_model.fit(X, y)

linear_regression = linear_model.named_steps["linearregression"]

coefficients = pd.DataFrame({
    "feature": FEATURES,
    "standardised_coefficient": linear_regression.coef_,
})

coefficients.to_csv(
    OUTPUT_DIR / "linear_regression_coefficients.csv",
    index=False,
)

print("\nLinear regression coefficients:")
print(coefficients.round(4).to_string(index=False))


# ==================================================
# 9. RANDOM FOREST FEATURE IMPORTANCE
# ==================================================

forest = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)

forest.fit(X, y)

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": forest.feature_importances_,
}).sort_values(
    "importance",
    ascending=False,
)

importance.to_csv(
    OUTPUT_DIR / "random_forest_feature_importance.csv",
    index=False,
)

plt.figure(figsize=(9, 6))

plt.barh(
    importance["feature"],
    importance["importance"],
)

plt.gca().invert_yaxis()
plt.xlabel("Feature importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "feature_importance.png",
    dpi=300,
)
plt.close()


# ==================================================
# 10. FINISH
# ==================================================

print("\nModel 2.2 finished.")
print(f"Results saved to: {OUTPUT_DIR}")

print("\nFiles created:")
for path in sorted(OUTPUT_DIR.iterdir()):
    print(f"- {path.name}")

