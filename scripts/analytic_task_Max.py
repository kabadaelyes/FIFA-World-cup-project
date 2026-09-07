import pandas as pd
import numpy as np
import statsmodels.stats.weightstats as stm
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats 


# Find the project folder and load the dataset
project_folder = Path(__file__).resolve().parent.parent
file_path = project_folder / "data" / "raw" / "matches.csv"
matches = pd.read_csv(file_path)


# Keep only group-stage matches
group_stage = matches[matches["Game Week"].isin([1, 2, 3])].copy()

print("Total matches:", len(matches))
print("Group-stage matches:", len(group_stage))


# Get goals and shots for each team in group stages
home_goals = group_stage[["home_team_name", "home_team_goal_count"]].copy()
home_goals.columns = ["Team", "Goals"]

away_goals = group_stage[["away_team_name", "away_team_goal_count"]].copy()
away_goals.columns = ["Team", "Goals"]

home_shots = group_stage[["home_team_name", "home_team_shots"]].copy()
home_shots.columns = ["Team", "Shots"]

away_shots = group_stage[["away_team_name", "away_team_shots"]].copy()
away_shots.columns = ["Team", "Shots"]


# Combine home and away data
team_goals = pd.concat([home_goals, away_goals], ignore_index=True)
team_shots = pd.concat([home_shots, away_shots], ignore_index=True)


# Calculate total goals and total shots for each team
team_stats = pd.merge(team_shots.groupby("Team")["Shots"].sum().reset_index(),
                      team_goals.groupby("Team")["Goals"].sum().reset_index(), on="Team")


# Calculate each team's overall group-stage shots per goal
team_stats["Shots/Goal"] = (team_stats["Shots"] / team_stats["Goals"]).round(4)


# Check that every team has three matches
matches_per_team = pd.concat([home_goals["Team"], away_goals["Team"]]).value_counts()

print("Number of teams:", len(team_stats))
print("Teams with 3 matches:", (matches_per_team == 3).sum())


# Find teams that reached the knockout stage
knockout_stage = matches[matches["Game Week"].isna()]
knockout_teams = pd.concat([knockout_stage["home_team_name"],knockout_stage["away_team_name"]]).drop_duplicates()


# Add qualification status
team_stats["qualification"] = "Eliminated"
team_stats.loc[team_stats["Team"].isin(knockout_teams),"qualification"] = "Qualified"

print("\nQualification:")
print(team_stats["qualification"].value_counts())


# Calculate each team's overall group-stage shots per goal
team_stats["Shots/Goal"] = (team_stats["Shots"] / team_stats["Goals"].replace(0, np.nan)).round(4)


# Calculate mean, median and standard deviation of shots/goal for each qualification group
group_statistics = (team_stats.groupby("qualification")["Shots/Goal"].agg(["mean", "median", "std"]))
group_statistics.columns = ["Average_shots_per_goal","Median_shots_per_goal","Std_shots_per_goal"]


# Add the group statistics to each team
team_stats = team_stats.join(group_statistics,on="qualification")


# Save the processed population dataset
processed_folder = (project_folder / "data" / "processed" / "max_attack")
processed_folder.mkdir(parents=True, exist_ok=True)
output_file = processed_folder / "team_shots_goal_rate.csv"
team_stats.to_csv(output_file, index=False)