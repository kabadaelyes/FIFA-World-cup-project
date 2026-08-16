import os
import pandas as pd

# ==========================================================
# HIT140 Foundation of Data Science
# Dataset Validation Script
#
# Validates the FootyStats dataset against the
# official FIFA statistics.
# ==========================================================

# ----------------------------------------------------------
# File paths
# ----------------------------------------------------------

RAW_DATA = "data/raw"
VALIDATION_DATA = "data/validation"

footy_file = os.path.join(RAW_DATA, "teams.csv")
fifa_file = os.path.join(VALIDATION_DATA, "fifa_validation.csv")
report_file = os.path.join(VALIDATION_DATA, "validation_report.csv")

# ----------------------------------------------------------
# Load datasets
# ----------------------------------------------------------

footy = pd.read_csv(footy_file)
fifa = pd.read_csv(fifa_file)

# ----------------------------------------------------------
# Required columns
# ----------------------------------------------------------

required_footy = [
    "team_name",
    "goals_scored",
    "goals_conceded",
    "clean_sheets"
]

required_fifa = [
    "Team",
    "Goals",
    "GoalsConceded",
    "CleanSheets"
]

for column in required_footy:
    if column not in footy.columns:
        raise ValueError(f"Column '{column}' not found in teams.csv")

for column in required_fifa:
    if column not in fifa.columns:
        raise ValueError(f"Column '{column}' not found in fifa_validation.csv")

# ----------------------------------------------------------
# Mapping between FIFA and FootyStats columns
# ----------------------------------------------------------

column_map = {
    "Goals": "goals_scored",
    "GoalsConceded": "goals_conceded",
    "CleanSheets": "clean_sheets"
}

# ----------------------------------------------------------
# Validation
# ----------------------------------------------------------

results = []

total_checks = 0
successful_checks = 0

print("=" * 60)
print(" FIFA vs FOOTYSTATS DATA VALIDATION REPORT")
print("=" * 60)

for _, fifa_team in fifa.iterrows():

    team = str(fifa_team["Team"]).strip()

    # Flexible team matching
    footy_team = footy[
        footy["team_name"].str.contains(
            team,
            case=False,
            na=False
        )
    ]

    if footy_team.empty:
        print(f"\n❌ {team} not found in FootyStats.")
        continue

    footy_team = footy_team.iloc[0]

    print(f"\n{team}")

    for fifa_column, footy_column in column_map.items():

        fifa_value = fifa_team[fifa_column]
        footy_value = footy_team[footy_column]

        total_checks += 1

        if fifa_value == footy_value:

            successful_checks += 1
            status = "PASS"

            print(f"  ✓ {fifa_column:<18} {footy_value}")

        else:

            status = "FAIL"

            print(
                f"  ✗ {fifa_column:<18}"
                f" FIFA={fifa_value}"
                f" FootyStats={footy_value}"
            )

        results.append({
            "Team": team,
            "Variable": fifa_column,
            "FIFA Value": fifa_value,
            "FootyStats Value": footy_value,
            "Status": status
        })

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------

print("\n" + "=" * 60)

if total_checks > 0:

    accuracy = (successful_checks / total_checks) * 100

    print(f"Total Checks      : {total_checks}")
    print(f"Successful Checks : {successful_checks}")
    print(f"Validation Rate   : {accuracy:.2f}%")

    if accuracy == 100:
        print("\n✅ DATA VALIDATION SUCCESSFUL")
        print("The selected FootyStats statistics match")
        print("the official FIFA statistics.")
    else:
        print("\n⚠ Differences detected.")

else:
    print("No matching teams were found.")

print("=" * 60)

# ----------------------------------------------------------
# Save report
# ----------------------------------------------------------

report = pd.DataFrame(results)
report.to_csv(report_file, index=False)

print(f"\nValidation report saved to:\n{report_file}")