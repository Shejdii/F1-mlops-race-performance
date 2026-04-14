from __future__ import annotations

from pathlib import Path

import pandas as pd

import config


def load_driver_skill() -> pd.DataFrame:
    return pd.read_csv(config.SKILL_REPORT)


def load_predictions() -> pd.DataFrame:
    return pd.read_csv(config.PREDICTIONS_OUT)


def load_race_year_map() -> pd.DataFrame:
    features = pd.read_parquet(config.ART_FEATURES)

    race_year = (
        features[["raceId", "year"]]
        .drop_duplicates()
        .copy()
    )

    return race_year


def load_champions() -> pd.DataFrame:
    champions_path = Path("data/external/champions.csv")
    return pd.read_csv(champions_path)


def build_driver_season(driver_skill: pd.DataFrame) -> pd.DataFrame:
    preds = load_predictions()
    race_year = load_race_year_map()

    merged = preds.merge(race_year, on="raceId", how="left")
    merged = merged.merge(
        driver_skill[["driverId", "driver_name"]],
        on="driverId",
        how="left",
    )

    missing_year = merged["year"].isna().sum()
    if missing_year > 0:
        raise ValueError(f"Missing year for {missing_year} prediction rows after raceId merge")

    merged["is_elite"] = (merged["residual_adj"] <= merged["residual_adj"].quantile(0.10)).astype(int)
    merged["is_bad"] = (merged["residual_adj"] >= merged["residual_adj"].quantile(0.90)).astype(int)

    grouped = (
        merged.groupby(["year", "driverId", "driver_name"], as_index=False)
        .agg(
            mean_residual=("residual_adj", "mean"),
            std_residual=("residual_adj", "std"),
            elite_rate=("is_elite", "mean"),
            bad_rate=("is_bad", "mean"),
            peak_residual=("residual_adj", "min"),
            races=("raceId", "nunique"),
        )
    )

    grouped["std_residual"] = grouped["std_residual"].fillna(0.0)
    return grouped


def pct_rank(series: pd.Series, ascending: bool) -> pd.Series:
    return series.rank(method="average", pct=True, ascending=ascending) * 100.0


def compute_driver_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["speed_score"] = pct_rank(out["mean_residual"], ascending=True)
    out["peak_score"] = pct_rank(out["peak_residual"], ascending=True)
    out["elite_score"] = pct_rank(out["elite_rate"], ascending=False)
    out["consistency_score"] = pct_rank(out["std_residual"], ascending=True)
    out["error_score"] = pct_rank(out["bad_rate"], ascending=True)

    out["driver_season_score"] = (
        0.36 * out["speed_score"]
        + 0.28 * out["peak_score"]
        + 0.16 * out["elite_score"]
        + 0.12 * out["consistency_score"]
        + 0.08 * out["error_score"]
    )

    return out


def build_season_comparison(df: pd.DataFrame, champions: pd.DataFrame) -> pd.DataFrame:
    results: list[dict] = []

    for year, season_df in df.groupby("year"):
        season_df = season_df.sort_values("driver_season_score", ascending=False).reset_index(drop=True)

        benchmark = season_df.iloc[0]

        champ_row = champions[champions["year"] == year]
        if champ_row.empty:
            continue

        champion_driver_id = champ_row["driverId"].iloc[0]
        champion_df = season_df[season_df["driverId"] == champion_driver_id]
        if champion_df.empty:
            continue

        champion = champion_df.iloc[0]
        champion_rank = int(champion_df.index[0]) + 1

        benchmark_score = float(benchmark["driver_season_score"])
        champion_score = float(champion["driver_season_score"])

        champion_to_ceiling_pct = (
            100.0 * champion_score / benchmark_score if benchmark_score != 0 else 0.0
        )
        gap_pct = 100.0 - champion_to_ceiling_pct

        results.append(
            {
                "year": int(year),
                "benchmark_driver": benchmark["driver_name"],
                "benchmark_score": benchmark_score,
                "champion_driver": champion["driver_name"],
                "champion_score": champion_score,
                "champion_rank": champion_rank,
                "champion_to_ceiling_pct": champion_to_ceiling_pct,
                "gap_pct": gap_pct,
            }
        )

    return pd.DataFrame(results).sort_values("year")


def main() -> int:
    Path(config.ART_REPORTS).mkdir(parents=True, exist_ok=True)

    driver_skill = load_driver_skill()
    driver_season = build_driver_season(driver_skill)
    driver_season = compute_driver_score(driver_season)

    champions = load_champions()
    comparison = build_season_comparison(driver_season, champions)

    out_path = Path(config.ART_REPORTS) / "season_analysis.csv"
    comparison.to_csv(out_path, index=False)

    print("\n=== SEASON ANALYSIS ===")
    if not comparison.empty:
        print(comparison.head(15).to_string(index=False))
    else:
        print("No season comparison rows produced.")

    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())