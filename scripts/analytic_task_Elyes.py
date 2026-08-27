import pandas as pd
from pathlib import Path


# Read the matches dataset
project_folder = Path(__file__).resolve().parent.parent
file_path = project_folder / "data" / "raw" / "matches.csv"

matches = pd.read_csv(file_path)


# Check the dataset
print("Dataset shape:")
print(matches.shape)

print("\nColumns:")
print(matches.columns.tolist())

print("\nFirst 5 rows:")
print(matches.head())


# Check the number of matches in each game week
print("\nGame Week counts:")
print(matches["Game Week"].value_counts(dropna=False).sort_index())


# Find all the teams
home_teams = set(matches["home_team_name"].dropna())
away_teams = set(matches["away_team_name"].dropna())

all_teams = home_teams.union(away_teams)

print("\nNumber of teams:")
print(len(all_teams))


# Keep only group-stage matches
group_stage = matches[matches["Game Week"].isin([1, 2, 3])].copy()

print("\nGroup-stage dataset shape:")
print(group_stage.shape)

print("\nGroup-stage Game Week counts:")
print(group_stage["Game Week"].value_counts().sort_index())


# Get possession for each team in each match
home_possession = group_stage[
    ["home_team_name", "home_team_possession"]
].copy()

home_possession.columns = ["team", "possession"]


away_possession = group_stage[
    ["away_team_name", "away_team_possession"]
].copy()

away_possession.columns = ["team", "possession"]


# Combine home and away teams
team_match_possession = pd.concat(
    [home_possession, away_possession],
    ignore_index=True
)


print("\nTeam-match dataset shape:")
print(team_match_possession.shape)

print("\nFirst 10 observations:")
print(team_match_possession.head(10))


# Check that every team has 3 group-stage matches
print("\nMatches per team:")
print(team_match_possession["team"].value_counts().sort_index())


# Calculate statistics for each team's group-stage possession
team_averages = (
    team_match_possession
    .groupby("team")["possession"]
    .agg(["mean", "median", "std", "min", "max"])
    .reset_index()
)

team_averages = team_averages.rename(
    columns={
        "mean": "average_possession",
        "median": "median_possession",
        "std": "sd_possession",
        "min": "minimum_possession",
        "max": "maximum_possession"
    }
)


print("\nTeam possession statistics:")
print(team_averages.to_string(index=False))


# Check the final dataset
print("\nNumber of teams:")
print(len(team_averages))

print("\nMissing values:")
print(team_averages.isnull().sum())


# Check the possession range
print("\nPossession range:")
print("Minimum:", team_match_possession["possession"].min())
print("Maximum:", team_match_possession["possession"].max())

# Find knockout-stage matches
knockout_stage = matches[matches["Game Week"].isna()]

# Get the teams that played in the knockout stage
knockout_teams = pd.concat([
    knockout_stage["home_team_name"],
    knockout_stage["away_team_name"]
]).drop_duplicates()

print("\nNumber of qualified teams:")
print(len(knockout_teams))

print("\nQualified teams:")
print(knockout_teams.tolist())

# Add qualification status
team_averages["qualification"] = "Eliminated"

for team in knockout_teams:
    team_averages.loc[
        team_averages["team"] == team,
        "qualification"
    ] = "Qualified"


print("\nQualification counts:")
print(team_averages["qualification"].value_counts())

print("\nTeam averages with qualification:")
print(team_averages.to_string(index=False))

print("\nFinal checks:")

print("Number of teams:", len(team_averages))

print("\nQualification:")
print(team_averages["qualification"].value_counts())

print("\nMissing values:")
print(team_averages.isnull().sum())

processed_folder = project_folder / "data" / "processed"

processed_folder.mkdir(exist_ok=True)

output_file = processed_folder / "task1_team_possession.csv"

team_averages.to_csv(output_file, index=False)

print("\nProcessed dataset saved to:")
print(output_file)