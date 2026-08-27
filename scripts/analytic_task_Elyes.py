import pandas as pd
from pathlib import Path
from scipy import stats


# Find the project folder and load the dataset
project_folder = Path(__file__).resolve().parent.parent
file_path = project_folder / "data" / "raw" / "matches.csv"

matches = pd.read_csv(file_path)


# Keep only group-stage matches
group_stage = matches[
    matches["Game Week"].isin([1, 2, 3])
].copy()

print("Total matches:", len(matches))
print("Group-stage matches:", len(group_stage))


# Get possession for each team
home_possession = group_stage[
    ["home_team_name", "home_team_possession"]
].copy()

home_possession.columns = ["team", "possession"]

away_possession = group_stage[
    ["away_team_name", "away_team_possession"]
].copy()

away_possession.columns = ["team", "possession"]


# Combine home and away data
team_possession = pd.concat(
    [home_possession, away_possession],
    ignore_index=True
)


# Calculate each team's group-stage possession
team_averages = (
    team_possession
    .groupby("team")["possession"]
    .agg(["mean", "median", "std"])
    .reset_index()
)

team_averages.columns = [
    "team",
    "average_possession",
    "median_possession",
    "sd_possession"
]


# Check that every team has three matches
matches_per_team = team_possession["team"].value_counts()

print("Number of teams:", len(team_averages))
print("Teams with 3 matches:", (matches_per_team == 3).sum())


# Find teams that reached the knockout stage
knockout_stage = matches[
    matches["Game Week"].isna()
]

knockout_teams = pd.concat([
    knockout_stage["home_team_name"],
    knockout_stage["away_team_name"]
]).drop_duplicates()


# Add qualification status
team_averages["qualification"] = "Eliminated"

team_averages.loc[
    team_averages["team"].isin(knockout_teams),
    "qualification"
] = "Qualified"


print("\nQualification:")
print(team_averages["qualification"].value_counts())


# Save the processed population dataset
processed_folder = project_folder / "data" / "processed"
processed_folder.mkdir(exist_ok=True)

output_file = processed_folder / "task1_team_possession.csv"

team_averages.to_csv(output_file, index=False)


# Take a stratified sample of 30 teams
qualified = team_averages[
    team_averages["qualification"] == "Qualified"
].sample(n=15, random_state=42)

eliminated = team_averages[
    team_averages["qualification"] == "Eliminated"
].sample(n=15, random_state=42)

sample = pd.concat(
    [qualified, eliminated],
    ignore_index=True
)


print("\nSample size:", len(sample))
print(sample["qualification"].value_counts())


# Get possession values for each group
qualified_possession = sample[
    sample["qualification"] == "Qualified"
]["average_possession"]

eliminated_possession = sample[
    sample["qualification"] == "Eliminated"
]["average_possession"]


# Descriptive statistics
print("\nDescriptive statistics:")

print("\nQualified:")
print("Mean:", qualified_possession.mean())
print("Median:", qualified_possession.median())
print("SD:", qualified_possession.std())

print("\nEliminated:")
print("Mean:", eliminated_possession.mean())
print("Median:", eliminated_possession.median())
print("SD:", eliminated_possession.std())


# Difference between the means
difference = (
    qualified_possession.mean()
    - eliminated_possession.mean()
)


# 95% confidence interval
standard_error = (
    (qualified_possession.std() ** 2 / len(qualified_possession))
    + (eliminated_possession.std() ** 2 / len(eliminated_possession))
) ** 0.5

df = len(qualified_possession) + len(eliminated_possession) - 2

t_value = stats.t.ppf(0.975, df)

margin_of_error = t_value * standard_error

lower = difference - margin_of_error
upper = difference + margin_of_error


print("\n95% Confidence Interval:")
print("Difference:", difference)
print("Lower:", lower)
print("Upper:", upper)


# One-tailed two-sample t-test
t_stat, two_tailed_p = stats.ttest_ind(
    qualified_possession,
    eliminated_possession,
    equal_var=True
)

one_tailed_p = two_tailed_p / 2

print("\nTwo-sample t-test:")
print("t-statistic:", t_stat)
print("One-tailed p-value:", one_tailed_p)