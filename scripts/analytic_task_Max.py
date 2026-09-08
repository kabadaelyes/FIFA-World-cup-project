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


# Take a stratified sample of 20 teams with only valid shots/goal
valid_teams = team_stats.dropna(subset=["Shots/Goal"])
qualified = valid_teams[valid_teams["qualification"] == "Qualified"].sample(n=10, random_state=42)
eliminated = valid_teams[valid_teams["qualification"] == "Eliminated"].sample(n=10, random_state=42)
sample = pd.concat([qualified, eliminated], ignore_index=True)
print("\nSample size:", len(sample))
print(sample["qualification"].value_counts())


# Get shots/goal values for each group
qualified_shots_goal_rate = sample.loc[sample["qualification"] == "Qualified","Shots/Goal"]
eliminated_shots_goal_rate  = sample.loc[sample["qualification"] == "Eliminated","Shots/Goal"]


# Descriptive statistics
print("\nDescriptive statistics:")

print("\nQualified:")
print("Average:", qualified_shots_goal_rate.mean())
print("Median:", qualified_shots_goal_rate.median())
print("STD:", qualified_shots_goal_rate.std())

print("\nEliminated:")
print("Average:", eliminated_shots_goal_rate.mean())
print("Median:", eliminated_shots_goal_rate.median())
print("STD:", eliminated_shots_goal_rate.std())


# Create descriptive statistics table
descriptive_statistics = pd.DataFrame({"qualification": ["Qualified", "Eliminated"],
    "Average": [qualified_shots_goal_rate.mean(), eliminated_shots_goal_rate.mean()],
    "Median": [qualified_shots_goal_rate.median(), eliminated_shots_goal_rate.median()],
    "Std": [qualified_shots_goal_rate.std(), eliminated_shots_goal_rate.std()]})

# Round to the 4th decimal value
descriptive_statistics[["Average", "Median", "Std"]] = (descriptive_statistics[["Average", "Median", "Std"]].round(4))


# Difference between the two sample means
difference = (qualified_shots_goal_rate.mean() - eliminated_shots_goal_rate.mean())


# Pooled standard deviation
pooled_sd = (((len(qualified_shots_goal_rate) - 1)
             * qualified_shots_goal_rate.var()
             + (len(eliminated_shots_goal_rate) - 1)
             * eliminated_shots_goal_rate.var())
            / 
            (len(qualified_shots_goal_rate)
            + len(eliminated_shots_goal_rate) - 2)) ** 0.5


# Standard error of the difference
standard_error = pooled_sd * (1 / len(qualified_shots_goal_rate) + 1 / len(eliminated_shots_goal_rate)) ** 0.5


# 95% confidence interval
lower, upper = stm._tconfint_generic(difference, standard_error, 
                                    dof=(len(qualified_shots_goal_rate) + len(eliminated_shots_goal_rate) - 2),
                                    alpha=0.05,
                                    alternative="two-sided")

print("95% Confidence Interval:\n")
print("Difference:", difference)
print("Lower:", lower)
print("Upper:", upper)


# Two-sample t-test
# Hypothesis: Qualified teams have a lower average Shots/Goal (lower Shots/Goal = better efficiency)
t_stat, one_tailed_p = stats.ttest_ind(qualified_shots_goal_rate, eliminated_shots_goal_rate, equal_var=True, alternative="less")

print("\nTwo-sample t-test:")
print("t-statistic:", t_stat)
print("One-tailed p-value:", one_tailed_p)


# Check the distribution of shots/goal
plt.boxplot([qualified_shots_goal_rate, eliminated_shots_goal_rate],tick_labels=["Qualified", "Eliminated"])
plt.ylabel("Shots per Goal")
plt.title("Shots per Goal by Qualification Status")


# Create a result paper
processed_folder = (project_folder / "data" / "processed" / "max_attack")
processed_folder.mkdir(parents=True, exist_ok=True)
conclusion_file = processed_folder / "conclusion.md"

with open(conclusion_file, "w") as file:
    file.write("Statistical Analysis Conclusion\n\n")

    file.write("Method: A one-tailed two-sample t-test was conducted using a significance level of 0.05.\n\n")

    file.write("Null hypothesis: Qualified teams do not have a lower average shots per goal than eliminated teams.\n")
    file.write("(Lower average shots per goal = better attack)\n\n")
    file.write(f"**One-tailed p-value:** {one_tailed_p:.4f}\n\n")
    if one_tailed_p < 0.05:
        file.write("Decision: Reject the null hypothesis.\n")
        file.write("Conclusion: Qualified teams indeed have a lower average shots per goal than eliminated teams.")
    else:
        file.write("Decision: Accept the null hypothesis.\n")
        file.write("Conclusion: Qualified teams indeed don't have a lower average shots per goal than eliminated teams.")


# Present the conclusion
def_value = 0.05

print("\nStatistical Decision:")
if one_tailed_p < def_value:
    print("Reject the null hypothesis\n")
    print("There is sufficient evidence that qualified teams have a lower average Shots/Goal")
else:
    print("Accept the null hypothesis\n")
    print("There is insufficient evidence that qualified teams have a lower average Shots/Goal")

print("\nConclusion:")
if one_tailed_p < def_value:
    print("Qualified teams are likely to be better at turning shots into goals than eliminated teams")
else:
    print("Qualified teams aren't likely to be better at turning shots into goals than eliminated teams")


# Save the processed population dataset
processed_folder = (project_folder / "data" / "processed" / "max_attack")
processed_folder.mkdir(parents=True, exist_ok=True)
output_file = processed_folder / "stratified_sample_data.csv"
sample.to_csv(output_file, index=False)
output_file1 = processed_folder / "descriptive_statistics.csv"
descriptive_statistics.to_csv(output_file1, index=False)
boxplot_file = processed_folder / "shots_per_goal_boxplot.png"
plt.savefig(boxplot_file, dpi=300, bbox_inches="tight")
plt.show()
plt.close()