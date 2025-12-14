import os
import warnings
import fastf1
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
pd.set_option("display.max_rows", None)

os.makedirs("f1_cache", exist_ok=True)
fastf1.Cache.enable_cache("f1_cache")
points_FIA = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

calendar_2025 = [
    "Australian Grand Prix",
    "Chinese Grand Prix",
    "Japanese Grand Prix",
    "Bahrain Grand Prix",
    "Saudi Arabian Grand Prix",
    "Miami Grand Prix",
    "Emilia Romagna Grand Prix",
    "Monaco Grand Prix",
    "Spanish Grand Prix",
    "Canadian Grand Prix",
    "Austrian Grand Prix",
    "British Grand Prix",
    "Belgian Grand Prix",
    "Hungarian Grand Prix",
    "Dutch Grand Prix",
    "Italian Grand Prix",
    "Azerbaijan Grand Prix",
    "Singapore Grand Prix",
    "United States Grand Prix",
    "Mexico City Grand Prix",
    "São Paulo Grand Prix",
    "Las Vegas Grand Prix",
    "Qatar Grand Prix",
    "Abu Dhabi Grand Prix",
]

drivers_2026 = [
    ("Lando Norris", "McLaren"), ("Oscar Piastri", "McLaren"),
    ("George Russell", "Mercedes"), ("Andrea Kimi Antonelli", "Mercedes"),
    ("Max Verstappen", "Red Bull Racing"), ("Isack Hadjar", "Red Bull Racing"),
    ("Charles Leclerc", "Ferrari"), ("Lewis Hamilton", "Ferrari"),
    ("Alexander Albon", "Williams"), ("Carlos Sainz Jnr", "Williams"),
    ("Liam Lawson", "Racing Bulls"), ("Arvid Lindblad", "Racing Bulls"),
    ("Fernando Alonso", "Aston Martin"), ("Lance Stroll", "Aston Martin"),
    ("Oliver Bearman", "Haas F1 Team"), ("Esteban Ocon", "Haas F1 Team"),
    ("Nico Hulkenberg", "Audi"), ("Gabriel Bortoleto", "Audi"),
    ("Pierre Gasly", "Alpine"), ("Franco Colapinto", "Alpine"),
    ("Valtteri Bottas", "Cadillac"), ("Sergio Perez", "Cadillac")
]

def load_session(year, event_name, session_code):
    try:
        session = fastf1.get_session(year, event_name, session_code)
        session.load()
        return session
    except Exception:
        return None

def load_season(year):
    rows = []

    schedule = fastf1.get_event_schedule(year, include_testing=False)

    for _, event in schedule.iterrows():
        gp_name = event["EventName"]

        race = load_session(year, gp_name, "R")
        quali = load_session(year, gp_name, "Q")

        if (race is None) or (quali is None):
            continue
        if race.results is None or quali.results is None:
            continue

        res = race.results.copy()
        
        if "ClassifiedPosition" in res.columns and "Position" not in res.columns:
            res["Position"] = res["ClassifiedPosition"]

        needed_cols = [
            "DriverNumber", "FullName", "TeamName",
            "GridPosition", "Position", "Points", "Status"
        ]
        missing = [c for c in needed_cols if c not in res.columns]
        if missing:
            continue

        df = res[needed_cols].copy()

        df["year"] = year
        df["gp_name"] = gp_name
        df["driver_id"] = df["FullName"]
        df["constructor_id"] = df["TeamName"]

        #quali
        q = quali.results[["DriverNumber", "Position"]].copy()
        q.rename(columns={"Position": "Position_quali"}, inplace=True)
        df = df.merge(q, on="DriverNumber", how="left")

        df["quali_pos"] = pd.to_numeric(df["Position_quali"], errors="coerce")

        #lap
        laps = race.laps

        if laps is not None and not laps.empty:
            med = (
                laps.dropna(subset=["LapTime"])
                .groupby("DriverNumber")["LapTime"]
                .median()
                .dt.total_seconds()
            )
            df["median_lap_s"] = df["DriverNumber"].map(med)

            pits = laps[laps["PitInTime"].notna()].groupby("DriverNumber").size()
            df["pit_stops"] = df["DriverNumber"].map(pits).fillna(0)
        else:
            df["median_lap_s"] = np.nan
            df["pit_stops"] = 0

        #basic numeric features
        df["race_points"] = df["Points"].fillna(0)
        df["grid"] = pd.to_numeric(df["GridPosition"], errors="coerce").fillna(20)

        rows.append(df)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


historical_list = []
for yr in [2023, 2024, 2025]:
    season_df = load_season(yr)
    if not season_df.empty:
        historical_list.append(season_df)

if not historical_list:
    raise RuntimeError(
        "no hist data loaded"
    )

historical = pd.concat(historical_list, ignore_index=True)

historical = historical.sort_values(["year", "gp_name"])

historical["driver_prev_races"] = historical.groupby("driver_id").cumcount()
historical["driver_ppg"] = (
    historical.groupby("driver_id")["race_points"].cumsum()
    - historical["race_points"]
) / historical["driver_prev_races"].replace(0, np.nan)

historical["team_prev_races"] = historical.groupby("constructor_id").cumcount()
historical["team_ppg"] = (
    historical.groupby("constructor_id")["race_points"].cumsum()
    - historical["race_points"]
) / historical["team_prev_races"].replace(0, np.nan)

#filling nans
historical["driver_ppg"].fillna(0, inplace=True)
historical["team_ppg"].fillna(0, inplace=True)
historical["median_lap_s"].fillna(historical["median_lap_s"].median(), inplace=True)
historical["quali_pos"].fillna(historical["quali_pos"].median(), inplace=True)

numeric = [
    "grid", "quali_pos", "median_lap_s", "pit_stops",
    "driver_prev_races", "driver_ppg",
    "team_prev_races", "team_ppg"
]
categoric = ["gp_name", "TeamName"]

preprocess = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric),
        ("categoric", OneHotEncoder(handle_unknown="ignore"), categoric)
    ]
)

#random forest tuned to reduce overfitting
model = Pipeline(
    steps=[
        ("prep", preprocess),
        ("rf", RandomForestRegressor(
            n_estimators=600,
            min_samples_leaf=3,
            max_depth=14,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

model.fit(historical[numeric + categoric], historical["race_points"])

calendar = calendar_2025

season_2026 = pd.DataFrame(
    [
        {
            "year": 2026,
            "gp_name": gp,
            "driver_id": d,
            "TeamName": t,
            "constructor_id": t
        }
        for gp in calendar for d, t in drivers_2026
    ]
)

driver_priors = (
    historical
    .sort_values(["year", "gp_name"])
    .groupby("driver_id", as_index=False)[["driver_id", "driver_prev_races", "driver_ppg"]]
    .tail(1)
    .reset_index(drop=True)
)

season_2026 = season_2026.merge(
    driver_priors,
    on="driver_id",
    how="left"
)

team_priors = (
    historical
    .sort_values(["year", "gp_name"])
    .groupby("constructor_id", as_index=False)[["constructor_id", "team_prev_races", "team_ppg"]]
    .tail(1)
    .reset_index(drop=True)
)

season_2026 = season_2026.merge(
    team_priors,
    on="constructor_id",
    how="left"
)

#priors for rookies
season_2026["driver_prev_races"].fillna(0, inplace=True)
season_2026["driver_ppg"].fillna(0, inplace=True)
season_2026["team_prev_races"].fillna(0, inplace=True)
season_2026["team_ppg"].fillna(0, inplace=True)

#Arvid Lindbald adjustment
season_2026.loc[season_2026["driver_id"] == "Arvid Lindblad", "driver_ppg"] *= 0.7

#Race weekend assumptions for all races
season_2026["grid"] = 10
season_2026["quali_pos"] = 10
season_2026["median_lap_s"] = historical["median_lap_s"].median()
season_2026["pit_stops"] = 2

#Randomizer
rng = np.random.default_rng(42)  

#base prediction per driver per race
season_2026["pred_points_base"] = model.predict(season_2026[numeric + categoric])


def score_race(df):
    df = df.copy()

    race_std = df["pred_points_base"].std()
    #bring the champion from 425-450 range prior to previos trends, and even point spread across all 22 drivers
    noise_scale = 0.7 * race_std if race_std > 0 else 0.15
    noise = rng.normal(loc=0.0, scale=noise_scale, size=len(df))

    df["pred_points"] = df["pred_points_base"] + noise

    df = df.sort_values("pred_points", ascending=False).reset_index(drop=True)
    df["pos"] = np.arange(1, len(df) + 1)
    df["pts"] = df["pos"].map(points_FIA).fillna(0)
    return df

season_2026 = (
    season_2026
    .groupby("gp_name", group_keys=False)
    .apply(score_race)
    .reset_index(drop=True)
)

drivers = (
    season_2026.groupby("driver_id")["pts"]
    .sum()
    .sort_values(ascending=False)
)

constructors = (
    season_2026.groupby("constructor_id")["pts"]
    .sum()
    .sort_values(ascending=False)
)

print()
print("=== 2026 DRIVERS CHAMPIONSHIP ===")
print(drivers)
print()
print("=== 2026 CONSTRUCTORS CHAMPIONSHIP ===")
print(constructors)

sns.set(style="whitegrid")

#driver bar chart
plt.figure(figsize=(10, 8))
drivers_plot = drivers.sort_values(ascending=True) 
plt.barh(drivers_plot.index, drivers_plot.values, color="tab:blue")
plt.xlabel("Points")
plt.title("2026 Drivers' Championship (Predicted)")
plt.tight_layout()
plt.show()
#constructor bar chart
plt.figure(figsize=(8, 6))
constructors_plot = constructors.sort_values(ascending=True)
plt.barh(constructors_plot.index, constructors_plot.values, color="tab:green")
plt.xlabel("Points")
plt.title("2026 Constructors' Championship (Predicted)")
plt.tight_layout()
plt.show()

champion = drivers.index[0]
champion_races = (
    season_2026[season_2026["driver_id"] == champion]
    .copy()
)

champion_races["gp_order"] = champion_races["gp_name"].apply(
    lambda x: calendar.index(x) if x in calendar else 999
)
champion_races = champion_races.sort_values("gp_order")

champion_races["cumulative_pts"] = champion_races["pts"].cumsum()

plt.figure(figsize=(10, 5))
plt.plot(
    champion_races["gp_order"] + 1,
    champion_races["cumulative_pts"],
    marker="o",
    color="tab:red"
)
plt.xticks(
    ticks=np.arange(1, len(calendar) + 1),
    labels=[c.replace(" Grand Prix", "") for c in calendar],
    rotation=60,
    ha="right",
    fontsize=8
)
plt.ylabel("Cumulative Points")
plt.xlabel("Round")
plt.title(f"{champion} – Cumulative Points Over 2026 Season (Predicted)")
plt.tight_layout()
plt.show()