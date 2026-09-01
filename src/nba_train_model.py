"""
train_model.py
Trains a Random Forest Regressor and an XGBoost Regressor to predict a
player's next-game points, compares both against the naive baseline
(just predicting the player's own rolling 10-game average), and saves
the trained models.

REGRESSION, not classification -- this is the key difference from the
Premier League project. We measure MAE/RMSE/R^2, not accuracy/F1.

IMPORTANT: chronological split again, same reasoning as before -- we
hold out the most recent 2 seasons entirely so evaluation reflects
predicting games the model has genuinely never seen, not just randomly
withheld rows possibly sandwiched between training games.

IMPORTANT: XGBoost model is saved with .save_model() (native JSON), NOT
joblib/pickle -- learned this the hard way on the Premier League project,
where a joblib-pickled XGBoost model threw a version-mismatch error
("input stream corrupted") when loaded in a different environment.
Random Forest is fine with joblib; only XGBoost needs this treatment.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "model_ready.csv"
MODELS_DIR = BASE_DIR / "models"

FEATURE_COLS = [
    "avg_points_last5", "avg_points_last10",
    "avg_minutes_last5", "avg_minutes_last10",
    "avg_rebounds_last5", "avg_assists_last5", "avg_fga_last5",
    "std_points_last10", "start_rate_last5", "season_avg_points_to_date",
    "days_rest", "is_back_to_back", "IsHome",
    "opp_points_allowed_last10", "opp_pace_last10",
]
TARGET_COL = "PTS"
TEST_SEASONS = ["2022-23", "2023-24"]


def chronological_split(df: pd.DataFrame):
    train = df[~df["Season"].isin(TEST_SEASONS)].copy()
    test = df[df["Season"].isin(TEST_SEASONS)].copy()
    print(f"Train: {len(train)} player-games ({train['Season'].min()} to {train['Season'].max()})")
    print(f"Test:  {len(test)} player-games ({', '.join(TEST_SEASONS)}) -- model has never seen these")
    return train, test


def evaluate(name, y_test, preds):
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"\n{'=' * 55}\n{name}\n{'=' * 55}")
    print(f"MAE:  {mae:.3f} points   (average size of the error, in points)")
    print(f"RMSE: {rmse:.3f} points  (penalizes big misses harder)")
    print(f"R^2:  {r2:.3f}          (fraction of variance explained, 1.0 = perfect)")
    return mae, rmse, r2


def within_n_points(y_test, preds, n):
    """What fraction of predictions land within n points of the real result?
    A more resume-friendly, intuitive number than MAE alone."""
    return (np.abs(y_test - preds) <= n).mean()


def main():
    df = pd.read_csv(DATA_PATH)
    train_df, test_df = chronological_split(df)

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

    # --- Baseline: predict the player's own rolling 10-game average ---
    baseline_preds = test_df["avg_points_last10"]
    print("Baseline: predict the player's own rolling 10-game average")
    base_mae, base_rmse, base_r2 = evaluate("Baseline (rolling average)", y_test, baseline_preds)
    print(f"Within 3 points: {within_n_points(y_test, baseline_preds, 3):.1%}")
    print(f"Within 5 points: {within_n_points(y_test, baseline_preds, 5):.1%}")
    print("\nAny model we ship needs to beat this MAE to be worth the added complexity.")

    MODELS_DIR.mkdir(exist_ok=True)

    # --- Random Forest ---
    rf = RandomForestRegressor(
        n_estimators=300, max_depth=8, min_samples_leaf=15,
        random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_mae, rf_rmse, rf_r2 = evaluate("Random Forest", y_test, rf_preds)
    print(f"Within 3 points: {within_n_points(y_test, rf_preds, 3):.1%}")
    print(f"Within 5 points: {within_n_points(y_test, rf_preds, 5):.1%}")
    joblib.dump(rf, MODELS_DIR / "random_forest.pkl")

    # --- XGBoost ---
    xgb = XGBRegressor(
        n_estimators=400, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="mae",
    )
    xgb.fit(X_train, y_train)
    xgb_preds = xgb.predict(X_test)
    xgb_mae, xgb_rmse, xgb_r2 = evaluate("XGBoost", y_test, xgb_preds)
    print(f"Within 3 points: {within_n_points(y_test, xgb_preds, 3):.1%}")
    print(f"Within 5 points: {within_n_points(y_test, xgb_preds, 5):.1%}")
    xgb.save_model(MODELS_DIR / "xgboost_model.json")  # native format, not joblib

    joblib.dump(FEATURE_COLS, MODELS_DIR / "feature_cols.pkl")

    # --- Summary ---
    print(f"\n{'=' * 70}\nSUMMARY (test: {', '.join(TEST_SEASONS)})\n{'=' * 70}")
    print(f"{'Model':<28}{'MAE':<10}{'RMSE':<10}{'R^2':<10}")
    print(f"{'Baseline (rolling avg)':<28}{base_mae:<10.3f}{base_rmse:<10.3f}{base_r2:<10.3f}")
    print(f"{'Random Forest':<28}{rf_mae:<10.3f}{rf_rmse:<10.3f}{rf_r2:<10.3f}")
    print(f"{'XGBoost':<28}{xgb_mae:<10.3f}{xgb_rmse:<10.3f}{xgb_r2:<10.3f}")
    print(f"\nSaved models to {MODELS_DIR}/")

    # --- Feature importance ---
    importance = pd.Series(xgb.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nXGBoost feature importance:")
    print(importance.round(3))

    return {"baseline": (base_mae, base_rmse, base_r2), "rf": (rf_mae, rf_rmse, rf_r2),
            "xgb": (xgb_mae, xgb_rmse, xgb_r2), "importance": importance}


if __name__ == "__main__":
    main()
