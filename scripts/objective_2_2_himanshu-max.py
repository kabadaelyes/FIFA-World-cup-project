
# HIT140 - Objective 2.2
# Predict the number of goals scored by a team in a World Cup match

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance


# --------------------------------------------------
# 1. FILES AND VARIABLES
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_FOLDER = ROOT / "data" / "world_cup_prediction"
OUTPUT_FOLDER = ROOT / "data" / "processed" / "objective_2_2"

FEATURES = [
    "Elo_rating",
    "FIFA_ranking",
    "Average_age",
    "Total_value",
    "Rest_days",
    "Previous_average_goals_scored",
    "Previous_average_goals_conceded",
    "Team_is_host"
]

# Matching column names in Elyes's match dataset
MATCH_COLUMNS = {
    "Elo_rating": ("home_elo", "away_elo"),
    "FIFA_ranking": ("home_fifa_rank", "away_fifa_rank"),
    "Average_age": ("home_squad_avg_age", "away_squad_avg_age"),
    "Total_value": ("home_squad_total_value_eur", "away_squad_total_value_eur"),
    "Rest_days": ("home_rest_days", "away_rest_days"),
    "Previous_average_goals_scored": (
        "home_prev_avg_goals_scored",
        "away_prev_avg_goals_scored"
    ),
    "Previous_average_goals_conceded": (
        "home_prev_avg_goals_conceded",
        "away_prev_avg_goals_conceded"
    ),
    "Team_is_host": ("home_is_host", "away_is_host")
}


# --------------------------------------------------
# 2. LOAD AND PREPARE THE DATA
# --------------------------------------------------

def load_data():

    # Max's dataset contains the eight predictors.
    max_data = pd.read_csv(
        DATA_FOLDER / "variables_for_goal_prediction.csv"
    )

    # Elyes's dataset contains the match scores.
    matches = pd.read_csv(
        DATA_FOLDER / "match_prediction_features.csv"
    )

    if len(max_data) != 208 or len(matches) != 104:
        raise ValueError("Incorrect number of rows in the datasets.")

    if len(FEATURES) != 8:
        raise ValueError("Exactly eight predictors are required.")

    if not {"Team", *FEATURES}.issubset(max_data.columns):
        raise ValueError("Required columns are missing from Max's dataset.")

    required = {
        "match_id", "home_team_name", "away_team_name",
        "home_score", "away_score"
    }

    for home_col, away_col in MATCH_COLUMNS.values():
        required.update([home_col, away_col])

    if not required.issubset(matches.columns):
        raise ValueError("Required columns are missing from the match dataset.")

    if matches["match_id"].isna().any():
        raise ValueError("Some match IDs are missing.")

    if matches["match_id"].duplicated().any():
        raise ValueError("Duplicate match IDs found.")

    # Max's rows contain home teams first, then away teams.
    teams = pd.concat(
        [matches["home_team_name"], matches["away_team_name"]],
        ignore_index=True
    )

    if not max_data["Team"].astype(str).str.strip().equals(
        teams.astype(str).str.strip()
    ):
        raise ValueError("Team order does not match between datasets.")

    # Check that the predictor values match both datasets.
    for feature, (home_col, away_col) in MATCH_COLUMNS.items():

        expected = pd.concat(
            [matches[home_col], matches[away_col]],
            ignore_index=True
        )

        actual = pd.to_numeric(max_data[feature], errors="coerce")
        expected = pd.to_numeric(expected, errors="coerce")

        if not np.allclose(
            actual, expected, rtol=1e-5, atol=1e-6
        ):
            raise ValueError(f"Values do not match for {feature}")

    # Add match IDs and goals scored to Max's dataset.
    data = max_data.copy()

    data.insert(
        0,
        "match_id",
        pd.concat([matches["match_id"]] * 2, ignore_index=True)
    )

    data["Goals"] = pd.concat(
        [matches["home_score"], matches["away_score"]],
        ignore_index=True
    )

    data[FEATURES + ["Goals"]] = data[
        FEATURES + ["Goals"]
    ].apply(pd.to_numeric, errors="coerce")

    if data[FEATURES + ["Goals"]].isna().any().any():
        raise ValueError("Missing or invalid numerical values found.")

    if not np.isfinite(data[FEATURES + ["Goals"]].to_numpy()).all():
        raise ValueError("Infinite values found.")

    if (data["Goals"] < 0).any():
        raise ValueError("Goals cannot be negative.")

    if (data["Goals"] % 1 != 0).any():
        raise ValueError("Goals must be whole numbers.")

    if not data["Team_is_host"].isin([0, 1]).all():
        raise ValueError("Host indicator must be 0 or 1.")

    if data.groupby("match_id").size().ne(2).any():
        raise ValueError("Each match must have two team records.")

    return data


# --------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS
# --------------------------------------------------

def explore_data(data):

    # Descriptive statistics
    statistics = data[FEATURES + ["Goals"]].describe()

    statistics.to_csv(
        OUTPUT_FOLDER / "descriptive_statistics.csv"
    )

    # Correlation analysis
    correlations = data[FEATURES + ["Goals"]].corr()

    correlations.to_csv(
        OUTPUT_FOLDER / "correlations.csv"
    )

    # Graph 1: Goals distribution
    goal_counts = data["Goals"].value_counts().sort_index()

    plt.figure(figsize=(7, 4))
    plt.bar(goal_counts.index, goal_counts.values)
    plt.xticks(goal_counts.index)
    plt.xlabel("Goals scored")
    plt.ylabel("Number of team-match records")
    plt.title("Distribution of Goals Scored")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_FOLDER / "goals_distribution.png",
        dpi=160
    )
    plt.close()

    # Graph 2: Correlation heatmap
    plt.figure(figsize=(9, 7))

    plt.imshow(
        correlations,
        vmin=-1,
        vmax=1,
        cmap="coolwarm"
    )

    plt.colorbar(label="Correlation")

    plt.xticks(
        range(len(correlations)),
        correlations.columns,
        rotation=45,
        ha="right",
        fontsize=8
    )

    plt.yticks(
        range(len(correlations)),
        correlations.columns,
        fontsize=8
    )

    plt.title("Correlation Between Variables")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_FOLDER / "correlation_heatmap.png",
        dpi=160
    )
    plt.close()


# --------------------------------------------------
# 4. TRAIN AND COMPARE REGRESSION MODELS
# --------------------------------------------------

def compare_models(train):

    # Compare ordinary linear regression with Ridge regression.
    models = {
        "Mean baseline": DummyRegressor(),

        "Linear regression": make_pipeline(
            StandardScaler(),
            LinearRegression()
        ),

        "Ridge 1": make_pipeline(
            StandardScaler(),
            Ridge(alpha=1)
        ),

        "Ridge 10": make_pipeline(
            StandardScaler(),
            Ridge(alpha=10)
        ),

        "Ridge 100": make_pipeline(
            StandardScaler(),
            Ridge(alpha=100)
        )
    }

    # Both teams from the same match stay in the same fold.
    cv = GroupKFold(n_splits=5)

    results = []

    for name, model in models.items():

        errors = -cross_val_score(
            model,
            train[FEATURES],
            train["Goals"],
            groups=train["match_id"],
            cv=cv,
            scoring="neg_mean_absolute_error"
        )

        results.append({
            "Model": name,
            "CV_MAE": errors.mean(),
            "CV_SD": errors.std()
        })

    comparison = pd.DataFrame(results).sort_values("CV_MAE")

    comparison.to_csv(
        OUTPUT_FOLDER / "model_comparison.csv",
        index=False
    )

    # Graph 3: Compare model performance
    plt.figure(figsize=(8, 4))
    plt.barh(comparison["Model"], comparison["CV_MAE"])
    plt.gca().invert_yaxis()
    plt.xlabel("Cross-validation MAE (lower is better)")
    plt.title("Regression Model Comparison")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_FOLDER / "model_comparison.png",
        dpi=160
    )
    plt.close()

    # Choose the regression model with the lowest CV error.
    regression_results = comparison[
        comparison["Model"] != "Mean baseline"
    ]

    best_name = regression_results.iloc[0]["Model"]
    best_model = models[best_name]

    return best_name, best_model, comparison


# --------------------------------------------------
# 5. EVALUATE THE FINAL MODEL
# --------------------------------------------------

def evaluate_model(model, model_name, train, test):

    model.fit(train[FEATURES], train["Goals"])

    predicted = model.predict(test[FEATURES])
    actual = test["Goals"].to_numpy()

    residuals = actual - predicted

    # Compare the selected model against a simple baseline.
    baseline = DummyRegressor()

    baseline.fit(train[FEATURES], train["Goals"])

    baseline_predictions = baseline.predict(test[FEATURES])

    baseline_mae = mean_absolute_error(
        actual, baseline_predictions
    )

    # Evaluation metrics
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    metrics = pd.DataFrame([{
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Baseline_MAE": baseline_mae
    }])

    metrics.to_csv(
        OUTPUT_FOLDER / "metrics.csv",
        index=False
    )

    # Save actual and predicted goals.
    predictions = test[
        ["match_id", "Team", "Goals"]
    ].copy()

    predictions["Predicted_goals"] = predicted
    predictions["Residual"] = residuals

    predictions.to_csv(
        OUTPUT_FOLDER / "test_predictions.csv",
        index=False
    )

    # Standardised regression coefficients
    coefficients = model.steps[-1][1].coef_

    coefficient_table = pd.DataFrame({
        "Feature": FEATURES,
        "Standardised_coefficient": coefficients
    })

    coefficient_table.to_csv(
        OUTPUT_FOLDER / "coefficients.csv",
        index=False
    )

    # Feature importance: how much error increases
    # when each predictor is shuffled.
    importance = permutation_importance(
        model,
        test[FEATURES],
        test["Goals"],
        scoring="neg_mean_absolute_error",
        n_repeats=10,
        random_state=42
    )

    feature_scores = pd.DataFrame({
        "Feature": FEATURES,
        "MAE_increase": importance.importances_mean
    }).sort_values("MAE_increase", ascending=False)

    feature_scores.to_csv(
        OUTPUT_FOLDER / "feature_importance.csv",
        index=False
    )

    # Graph 4: Feature importance
    plt.figure(figsize=(8, 4))

    plt.barh(
        feature_scores["Feature"],
        feature_scores["MAE_increase"]
    )

    plt.gca().invert_yaxis()
    plt.xlabel("Increase in MAE when feature is shuffled")
    plt.title("Feature Importance")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_FOLDER / "feature_importance.png",
        dpi=160
    )
    plt.close()

    # Graph 5: Actual vs predicted goals
    plt.figure(figsize=(6, 5))

    plt.scatter(actual, predicted, alpha=0.7)

    limits = [
        min(actual.min(), predicted.min()),
        max(actual.max(), predicted.max())
    ]

    plt.plot(
        limits, limits,
        "--",
        label="Perfect prediction"
    )

    plt.xlabel("Actual goals")
    plt.ylabel("Predicted goals")
    plt.title("Actual vs Predicted Goals")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_FOLDER / "actual_vs_predicted.png",
        dpi=160
    )
    plt.close()

    # Graph 6: Residual analysis
    plt.figure(figsize=(6, 5))

    plt.scatter(predicted, residuals, alpha=0.7)
    plt.axhline(0, linestyle="--")

    plt.xlabel("Predicted goals")
    plt.ylabel("Residual (actual - predicted)")
    plt.title("Residual Analysis")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_FOLDER / "residual_plot.png",
        dpi=160
    )
    plt.close()

    return metrics


# --------------------------------------------------
# 6. RUN THE ANALYSIS
# --------------------------------------------------

def main():

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    data = load_data()

    data.to_csv(
        OUTPUT_FOLDER / "team_match_208.csv",
        index=False
    )

    explore_data(data)

    # Split by match to avoid sharing a match
    # between training and testing.
    split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42
    )

    train_index, test_index = next(
        split.split(data, groups=data["match_id"])
    )

    train = data.iloc[train_index]
    test = data.iloc[test_index]

    best_name, best_model, comparison = compare_models(train)

    metrics = evaluate_model(
        best_model,
        best_name,
        train,
        test
    )

    print("\nObjective 2.2 - Goals Scored Prediction")

    print(
        f"\nDataset: {len(data)} rows, "
        f"{data['match_id'].nunique()} matches, "
        f"{len(FEATURES)} predictors"
    )

    print("\nCross-validation results:")
    print(comparison.to_string(index=False))

    print("\nFinal test results:")
    print(metrics.to_string(index=False))

    print("\nResults saved in:", OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
