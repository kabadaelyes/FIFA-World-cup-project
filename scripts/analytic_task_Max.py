import pandas as pd
import scipy.stats as st
import math
from pathlib import Path
import statsmodels.stats.weightstats as stm


# Find the project folder and load the dataset
project_folder = Path(__file__).resolve().parent.parent
file_path = project_folder / "data" / "raw" / "matches.csv"
matches = pd.read_csv(file_path)


# Keep only group-stage matches
group_stage = matches[matches["Game Week"].isin([1, 2, 3])].copy()
print("Total matches:", len(matches))
print("Group-stage matches:", len(group_stage))


# Get shots and goals for each team
home_shots = group_stage["home_team_name", "home_team_shots"].copy()
home_shots.columns = ["Team", "Shots"]
away_shots = group_stage["away_team_name", "away_team_shots"].copy()
away_shots.columns = ["Team", "Shots"]
home_goals = group_stage["home_team_name", "home_team_goal_count"].copy()
home_goals.columns = ["Team", "Goals"]
away_goals = group_stage["away_team_name", "away_team_goal_count"].copy()
away_goals.columns = ["Team", "Goals"]


# Calculate shots/goal rate of each team
home_shotspergoal = [group_stage["home_team_name"].copy(), round(home_shots / home_goals, 2)]
home_shotspergoal.columns = ["Team", "Shots/Goal"]
away_shotspergoal = [group_stage["home_team_name"].copy(), round(away_shots / away_goals, 2)]
away_shotspergoal.columns = ["Team", "Shots/Goal"]


# Combine home and away data
team_shots = pd.concat([home_shots, away_shots]) # ignore_index=True
team_goals = pd.concat([home_goals, away_goals])
team_shotspergoal = pd.concat([home_shotspergoal, away_shotspergoal])
team_chance = pd.concat([team_shots, team_goals, team_shotspergoal])

# Calculate each team's group-stage scoring opportunity
team_averages = (team_chance.groupby("team")["possession"].agg(["mean", "median", "std"]).reset_index())

team_averages.columns = ["team","average_scoring_opportunity","median_opportunity","std_opportunity"]


# Check that every team has three matches
matches_per_team = team_possession["team"].value_counts()
print("Number of teams:", len(team_averages))
print("Teams with 3 matches:", (matches_per_team == 3).sum())


# Find teams that reached the knockout stage
knockout_stage = matches[matches["Game Week"].isna()]
knockout_teams = pd.concat([knockout_stage["home_team_name"], knockout_stage["away_team_name"]]).drop_duplicates()


# Add qualification status
team_averages["qualification"] = "Eliminated"
team_averages.loc[team_averages["team"].isin(knockout_teams),"qualification"] = "Qualified"
print("\nQualification:")
print(team_averages["qualification"].value_counts())


# Save the processed population dataset
processed_folder = project_folder / "data" / "processed"
processed_folder.mkdir(exist_ok=True)
output_file = processed_folder / "task_Max_team_scoring_opportunity.csv"
team_averages.to_csv(output_file, index=False)
