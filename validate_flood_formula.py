import pandas as pd

# Load the historical rainfall dataset
df = pd.read_csv("rainfall in india 1901-2015.csv")
# Your actual flood formula logic (from app.py), applied to each row
def calculate_flood_risk(rainfall_24h_equivalent):
    if rainfall_24h_equivalent < 2.5:
        return 5
    elif rainfall_24h_equivalent < 15.5:
        return 20
    elif rainfall_24h_equivalent < 64.4:
        return 40
    elif rainfall_24h_equivalent < 115.5:
        return 65
    elif rainfall_24h_equivalent < 204.4:
        return 85
    else:
        return 98

# This dataset gives monthly totals, not daily - so we estimate an
# "average daily equivalent" for the peak monsoon month (June-Sept) each year
df["monsoon_total"] = df["JUN"] + df["JUL"] + df["AUG"] + df["SEP"]
df["avg_daily_equivalent"] = df["monsoon_total"] / 122  # ~122 days in Jun-Sep

df["predicted_flood_risk"] = df["avg_daily_equivalent"].apply(calculate_flood_risk)

# Show results: which years/subdivisions your formula flags as high risk
high_risk = df[df["predicted_flood_risk"] >= 65]
print(f"Total records: {len(df)}")
print(f"Flagged as high flood risk (65%+): {len(high_risk)}")
print("\nSample high-risk years/regions your formula flagged:")
print(high_risk[["SUBDIVISION", "YEAR", "predicted_flood_risk"]].head(20))