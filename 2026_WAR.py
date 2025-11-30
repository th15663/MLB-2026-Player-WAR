import pandas as pd
from pybaseball import batting_stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import numpy as np

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------
def load_multi_year_stats(start_year=2015, end_year=2025):
    all_stats = []
    for year in range(start_year, end_year + 1):
        print(f"Loading {year} batting data...")
        df = batting_stats(year)
        df["year"] = year
        all_stats.append(df)
    return pd.concat(all_stats, ignore_index=True)


# ------------------------------------------------
# FIND PLAYER IDENTIFIER
# ------------------------------------------------
def get_player_id_column(df):
    for col in ["playerID", "playerid", "IDfg", "ID"]:
        if col in df.columns:
            return col
    raise ValueError("No player ID column found.")


# ------------------------------------------------
# CLEAN COLUMNS
# ------------------------------------------------
def clean_stats(df):
    pid = get_player_id_column(df)

    cols = [
        "Name", pid, "year",
        "PA", "AB", "H", "2B", "3B", "HR",
        "BB", "SO", "OBP", "SLG", "OPS",
        "ISO", "BABIP", "SB", "CS",
        "WAR"
    ]

    cols = [c for c in cols if c in df.columns]
    df = df[cols].copy()
    df = df.dropna(subset=["WAR"])
    return df, pid


# ------------------------------------------------
# ADD MULTI-YEAR FEATURES
# ------------------------------------------------
def add_multi_year_features(df, pid):
    df = df.sort_values([pid, "year"])

    base_features = ["WAR", "OBP", "SLG", "OPS", "HR", "BB", "SO", "PA"]

    for col in base_features:
        df[f"{col}_prev1"] = df.groupby(pid)[col].shift(1)
        df[f"{col}_prev2"] = df.groupby(pid)[col].shift(2)
        df[f"{col}_avg2"] = (
            df.groupby(pid)[col]
            .rolling(2)
            .mean()
            .reset_index(0, drop=True)
        )
        df[f"{col}_delta"] = df[col] - df[f"{col}_prev1"]

    df = df.dropna()
    return df


# ------------------------------------------------
# MAIN
# ------------------------------------------------
if __name__ == "__main__":

    print("Loading 2015–2025 batting data...")
    df = load_multi_year_stats(2015, 2025)

    print("Cleaning data...")
    df, pid = clean_stats(df)

    print(f"Using player ID column: {pid}")

    print("Adding multi-year features...")
    df = add_multi_year_features(df, pid)

    print(f"Total training rows after adding features: {len(df)}")

    # --------------------------
    # Prepare data
    # --------------------------
    y = df["WAR"]
    X = df.drop(["WAR", "Name", pid], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # --------------------------
    # XGBoost model
    # --------------------------
    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42
    )

    print("Training model...")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"\nFinal RMSE: {rmse:.4f}")

    importances = sorted(
        list(zip(X.columns, model.feature_importances_)),
        key=lambda x: x[1],
        reverse=True
    )

    print("\nTop 15 important features:")
    for name, score in importances[:15]:
        print(f"{name}: {score:.4f}")

    # =============================
    #   PREDICT 2026 WAR
    # =============================

    print("\nGenerating 2026 WAR predictions...")

    df_2025 = df[df['year'] == 2025].copy()

    feature_cols = X.columns  # ensures perfect alignment with training data

    df_2025['Predicted_WAR_2026'] = model.predict(df_2025[feature_cols])

    final_2026 = df_2025[[pid, 'Name', 'Predicted_WAR_2026']] \
        .sort_values('Predicted_WAR_2026', ascending=False)

    print("\nTop 20 Projected WAR for 2026:")
    print(final_2026.head(20))

# Save full table
# final_2026.to_csv("predicted_war_2026.csv", index=False)
# print("\nSaved: predicted_war_2026.csv")

