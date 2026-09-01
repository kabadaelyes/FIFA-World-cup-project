# Himanshu Bhattarai
# HIT140 FIFA World Cup 2026 project
#
# Question:
# What proportion of teams with fewer than two cards per
# group-stage match qualified for the knockout stage,
# compared with teams with two or more cards per match?

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# Find the main project folder and load the match dataset
project_folder = Path(__file__).resolve().parent.parent
data_file = project_folder / "data" / "raw" / "matches.csv"

matches = pd.read_csv(data_file)


# Game Weeks 1, 2 and 3 are the group stage
group_matches = matches[
    matches["Game Week"].isin([1, 2, 3])
].copy()

# Blank Game Week values represent knockout matches
knockout_matches = matches[
    matches["Game Week"].isna()
].copy()

print("Total matches:", len(matches))
print("Group-stage matches:", len(group_matches))


# Teams appearing in the knockout stage had qualified
# No knockout-stage cards are used in this analysis
qualified_teams = set(
    knockout_matches["home_team_name"]
).union(
    set(knockout_matches["away_team_name"])
)


# Select home-team card data
home_cards = group_matches[
    [
        "home_team_name",
        "home_team_yellow_cards",
        "home_team_red_cards"
    ]
].copy()

home_cards.columns = [
    "team",
    "yellow_cards",
    "red_cards"
]


# Select away-team card data
away_cards = group_matches[
    [
        "away_team_name",
        "away_team_yellow_cards",
        "away_team_red_cards"
    ]
].copy()

away_cards.columns = [
    "team",
    "yellow_cards",
    "red_cards"
]


# Put home and away teams into the same table
team_cards = pd.concat(
    [home_cards, away_cards],
    ignore_index=True
)

# A yellow card counts as one and a red card counts as two
team_cards["total_cards"] = (
    team_cards["yellow_cards"]
    + 2 * team_cards["red_cards"]
)


# Combine each team's three group-stage matches
team_data = (
    team_cards
    .groupby("team")
    .agg(
        matches_played=("team", "count"),
        yellow_cards=("yellow_cards", "sum"),
        red_cards=("red_cards", "sum"),
        total_cards=("total_cards", "sum")
    )
    .reset_index()
)

# Calculate average cards per group-stage match
team_data["cards_per_match"] = (
    team_data["total_cards"]
    / team_data["matches_played"]
)


# Check that every team has three group matches
if not (team_data["matches_played"] == 3).all():
    raise ValueError(
        "At least one team does not have three group matches."
    )


# Add qualification information
# Qualified is 1 and eliminated is 0
team_data["qualified"] = (
    team_data["team"]
    .isin(qualified_teams)
    .astype(int)
)

team_data["qualification_status"] = (
    team_data["qualified"].map({
        1: "Qualified",
        0: "Eliminated"
    })
)


# Create the two card groups
team_data["discipline_group"] = "2 or more"

team_data.loc[
    team_data["cards_per_match"] < 2,
    "discipline_group"
] = "Fewer than 2"


# Round cards per match to two decimal places
team_data["cards_per_match"] = (
    team_data["cards_per_match"].round(2)
)

print("Number of teams:", len(team_data))
print("Qualified teams:", team_data["qualified"].sum())

print("\nCard groups:")
print(team_data["discipline_group"].value_counts())


# Take a proportionate stratified sample of 36 teams
# The same random state produces the same sample each time
fewer_sample = team_data[
    team_data["discipline_group"] == "Fewer than 2"
].sample(
    n=28,
    random_state=140
)

more_sample = team_data[
    team_data["discipline_group"] == "2 or more"
].sample(
    n=8,
    random_state=140
)

sample = pd.concat(
    [fewer_sample, more_sample],
    ignore_index=True
)

print("\nSample size:", len(sample))
print(sample["discipline_group"].value_counts())


# Calculate descriptive statistics and qualification proportions
summary = (
    sample
    .groupby("discipline_group")
    .agg(
        sample_size=("team", "count"),
        qualified_teams=("qualified", "sum"),
        mean_cards=("cards_per_match", "mean"),
        median_cards=("cards_per_match", "median"),
        standard_deviation=("cards_per_match", "std"),
        qualification_percentage=("qualified", "mean")
    )
    .reset_index()
)

summary["qualification_percentage"] = (
    summary["qualification_percentage"] * 100
)


# Calculate a 95% confidence interval for each proportion
lower_values = []
upper_values = []

for _, row in summary.iterrows():

    interval = stats.binomtest(
        k=int(row["qualified_teams"]),
        n=int(row["sample_size"])
    ).proportion_ci(
        confidence_level=0.95,
        method="wilson"
    )

    lower_values.append(interval.low * 100)
    upper_values.append(interval.high * 100)

summary["ci_lower"] = lower_values
summary["ci_upper"] = upper_values


# Round the results to make the table easier to read
columns_to_round = [
    "mean_cards",
    "median_cards",
    "standard_deviation",
    "qualification_percentage",
    "ci_lower",
    "ci_upper"
]

summary[columns_to_round] = summary[
    columns_to_round
].round(2)

print("\nSummary results:")
print(summary.to_string(index=False))


# Calculate the difference between qualification percentages
fewer_percentage = summary.loc[
    summary["discipline_group"] == "Fewer than 2",
    "qualification_percentage"
].iloc[0]

more_percentage = summary.loc[
    summary["discipline_group"] == "2 or more",
    "qualification_percentage"
].iloc[0]

percentage_difference = (
    fewer_percentage - more_percentage
)

print(
    "\nPercentage-point difference:",
    round(percentage_difference, 2)
)


# Perform the required two-sample t-test
# The mean of a 1/0 variable is its proportion
fewer_values = sample.loc[
    sample["discipline_group"] == "Fewer than 2",
    "qualified"
]

more_values = sample.loc[
    sample["discipline_group"] == "2 or more",
    "qualified"
]

t_statistic, p_value = stats.ttest_ind(
    fewer_values,
    more_values,
    equal_var=False
)

alpha = 0.05

if p_value < alpha:
    decision = "Reject the null hypothesis"
else:
    decision = "Do not reject the null hypothesis"

print("\nWelch two-sample t-test:")
print("t-statistic:", round(t_statistic, 4))
print("p-value:", round(p_value, 4))
print("Decision:", decision)


# Create the folder for my processed results
output_folder = (
    project_folder
    / "data"
    / "processed"
    / "himanshu_discipline"
)

output_folder.mkdir(
    parents=True,
    exist_ok=True
)


# Save the processed data for all 48 teams
team_data.to_csv(
    output_folder / "01_team_details.csv",
    index=False
)

# Save the 36 teams used in the sample
sample.to_csv(
    output_folder / "02_stratified_sample.csv",
    index=False
)

# Save descriptive statistics and confidence intervals
summary.to_csv(
    output_folder / "03_discipline_results.csv",
    index=False
)


# Save the t-test result
test_result = pd.DataFrame({
    "t_statistic": [round(t_statistic, 4)],
    "p_value": [round(p_value, 4)],
    "alpha": [alpha],
    "decision": [decision],
    "percentage_difference": [
        round(percentage_difference, 2)
    ]
})

test_result.to_csv(
    output_folder / "04_t_test_result.csv",
    index=False
)


# Create one graph showing the qualification percentages
graph_data = summary.set_index(
    "discipline_group"
).loc[
    ["Fewer than 2", "2 or more"]
].reset_index()

bars = plt.bar(
    graph_data["discipline_group"],
    graph_data["qualification_percentage"],
    color=["steelblue", "darkorange"]
)

plt.title("Qualification by Group-Stage Cards")
plt.xlabel("Cards per group-stage match")
plt.ylabel("Teams that qualified (%)")
plt.ylim(0, 100)

plt.bar_label(
    bars,
    fmt="%.1f%%",
    padding=3
)

plt.tight_layout()

plt.savefig(
    output_folder / "05_qualification_chart.png",
    dpi=300
)

plt.show()

print("\nAnalysis complete.")
print("Files saved in:", output_folder)