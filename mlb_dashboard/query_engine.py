from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date

import pandas as pd

from .config import AppConfig

try:
    import duckdb
except ImportError:  # pragma: no cover
    duckdb = None


@dataclass
class QueryFilters:
    split: str = "overall"
    recent_window: str = "season"
    weighted_mode: str = "weighted"
    min_pitch_count: int = 0
    min_bip: int = 0
    likely_starters_only: bool = False


class StatcastQueryEngine:
    def __init__(self, config: AppConfig):
        self.config = config
        self._conn = None

    @property
    def conn(self):
        if duckdb is None:
            raise RuntimeError("duckdb is required for the local explorer. Install requirements.txt first.")
        if self._conn is None:
            self._conn = duckdb.connect(str(self.config.db_path), read_only=True)
        return self._conn

    def load_daily_slate(self, target_date: date) -> list[dict]:
        path = self.config.daily_dir / target_date.isoformat() / "slate.json"
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def load_daily_rosters(self, target_date: date) -> pd.DataFrame:
        path = self.config.daily_dir / target_date.isoformat() / "rosters.parquet"
        if not path.exists():
            return pd.DataFrame(columns=["team", "player_id", "player_name"])
        return pd.read_parquet(path)

    def load_daily_hitter_rolling(self, target_date: date) -> pd.DataFrame:
        path = self.config.daily_dir / target_date.isoformat() / "daily_hitter_rolling.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def load_daily_pitcher_rolling(self, target_date: date) -> pd.DataFrame:
        path = self.config.daily_dir / target_date.isoformat() / "daily_pitcher_rolling.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def load_daily_batter_zone_profiles(self, target_date: date) -> pd.DataFrame:
        path = self.config.daily_dir / target_date.isoformat() / "daily_batter_zone_profiles.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def load_daily_pitcher_zone_profiles(self, target_date: date) -> pd.DataFrame:
        path = self.config.daily_dir / target_date.isoformat() / "daily_pitcher_zone_profiles.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def load_daily_batter_family_zone_profiles(self, target_date: date) -> pd.DataFrame:
        path = self.config.daily_dir / target_date.isoformat() / "daily_batter_family_zone_profiles.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def load_daily_pitcher_family_zone_context(self, target_date: date) -> pd.DataFrame:
        path = self.config.daily_dir / target_date.isoformat() / "daily_pitcher_family_zone_context.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def load_daily_pitcher_movement_arsenal(self, target_date: date) -> pd.DataFrame:
        path = self.config.daily_dir / target_date.isoformat() / "daily_pitcher_movement_arsenal.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def load_daily_hitter_pitcher_exclusions(self, target_date: date) -> pd.DataFrame:
        path = self.config.daily_dir / target_date.isoformat() / "hitter_pitcher_exclusions.parquet"
        if not path.exists():
            return pd.DataFrame(columns=["player_id", "exclude_from_hitter_tables"])
        return pd.read_parquet(path)

    def load_pitcher_game_outcomes_for_projections(
        self,
        pitcher_ids: list[int],
        n_starts: int = 30,
    ) -> pd.DataFrame:
        """Load historical game outcomes for the given pitchers, most recent n_starts per pitcher.

        Returns rows with columns including batters_faced, strikeouts, walks, pitch_count
        (pitch_count may be 0 for starts before it was added to the build pipeline).
        """
        path = self.config.tracking_dir / "pitcher_game_outcomes.parquet"
        if not path.exists():
            return pd.DataFrame()
        frame = pd.read_parquet(path)
        if frame.empty or "pitcher_id" not in frame.columns:
            return pd.DataFrame()
        frame = frame.loc[
            frame["pitcher_id"].isin(pitcher_ids)
            & frame.get("started", pd.Series(True, index=frame.index)).fillna(False)
        ].copy()
        if frame.empty:
            return frame
        if "slate_date" in frame.columns:
            frame["slate_date"] = pd.to_datetime(frame["slate_date"], errors="coerce")
            frame = frame.sort_values(["pitcher_id", "slate_date"], ascending=[True, False])
            frame = frame.groupby("pitcher_id", group_keys=False).head(n_starts)
        return frame.reset_index(drop=True)

    def load_pitcher_start_outcomes_for_projections(
        self,
        target_date: date,
        pitcher_ids: list[int],
        n_starts: int = 30,
    ) -> pd.DataFrame:
        daily_path = self.config.daily_dir / target_date.isoformat() / "strikeouts" / "pitcher_start_outcomes.parquet"
        reusable_path = self.config.reusable_dir / "pitcher_start_outcomes.parquet"
        if daily_path.exists():
            return pd.read_parquet(daily_path)
        if not reusable_path.exists():
            return pd.DataFrame()
        frame = pd.read_parquet(reusable_path)
        if frame.empty or "pitcher_id" not in frame.columns:
            return pd.DataFrame()
        frame["pitcher_id"] = pd.to_numeric(frame["pitcher_id"], errors="coerce")
        frame["game_date"] = pd.to_datetime(frame.get("game_date"), errors="coerce")
        frame = frame.loc[
            frame["pitcher_id"].isin(pitcher_ids)
            & frame["game_date"].dt.date.lt(target_date)
        ].copy()
        if frame.empty:
            return frame
        frame = frame.sort_values(["pitcher_id", "game_date", "game_pk"], ascending=[True, False, False], na_position="last")
        return frame.groupby("pitcher_id", group_keys=False).head(n_starts).reset_index(drop=True)

    def load_daily_top_slate_hitters(self, target_date: date) -> pd.DataFrame:
        path = self.config.daily_dir / target_date.isoformat() / "top_slate_hitters.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def load_daily_top_slate_pitchers(self, target_date: date) -> pd.DataFrame:
        path = self.config.daily_dir / target_date.isoformat() / "top_slate_pitchers.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)

    def load_daily_filtered_top_slate_hitters(self, target_date: date, filters: QueryFilters) -> pd.DataFrame:
        daily_dir = self.config.daily_dir / target_date.isoformat()
        path = daily_dir / "top_boards" / f"{filters.split}__{filters.recent_window}__{filters.weighted_mode}__hitters.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return _filter_combo(self.load_daily_top_slate_hitters(target_date), filters.split, filters.recent_window, filters.weighted_mode)

    def load_daily_filtered_top_slate_pitchers(self, target_date: date, filters: QueryFilters) -> pd.DataFrame:
        daily_dir = self.config.daily_dir / target_date.isoformat()
        path = daily_dir / "top_boards" / f"{filters.split}__{filters.recent_window}__{filters.weighted_mode}__pitchers.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return _filter_combo(self.load_daily_top_slate_pitchers(target_date), filters.split, filters.recent_window, filters.weighted_mode)

    def load_daily_slate_summary(self, target_date: date, filters: QueryFilters) -> pd.DataFrame:
        path = self.config.daily_dir / target_date.isoformat() / "slate_summary.parquet"
        if not path.exists():
            return pd.DataFrame()
        return _filter_combo(pd.read_parquet(path), filters.split, filters.recent_window, filters.weighted_mode)

    def load_daily_artifact_manifest(self, target_date: date) -> pd.DataFrame:
        path = self.config.daily_dir / target_date.isoformat() / "artifact_manifest.parquet"
        if path.exists():
            return pd.read_parquet(path)
        json_path = self.config.daily_dir / target_date.isoformat() / "artifact_manifest.json"
        if json_path.exists():
            return pd.DataFrame([json.loads(json_path.read_text(encoding="utf-8"))])
        return pd.DataFrame()

    def load_daily_game_bundle(self, target_date: date, game_pk: int) -> dict[str, pd.DataFrame]:
        game_dir = self.config.daily_dir / target_date.isoformat() / "games" / str(game_pk)
        bundle: dict[str, pd.DataFrame] = {}
        for name in ["matchup", "rolling", "pitcher_detail", "zones", "pitch_shape", "exports"]:
            path = game_dir / f"{name}.parquet"
            bundle[name] = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        return bundle

    def get_pitcher_cards(self, pitcher_ids: list[int], filters: QueryFilters) -> pd.DataFrame:
        if not pitcher_ids:
            return pd.DataFrame()
        return self.conn.execute(
            """
            SELECT *
            FROM pitcher_metrics
            WHERE pitcher_id IN $pitcher_ids
              AND split_key = $split_key
              AND recent_window = $recent_window
              AND weighted_mode = $weighted_mode
            ORDER BY pitcher_name
            """,
            {
                "pitcher_ids": pitcher_ids,
                "split_key": filters.split,
                "recent_window": filters.recent_window,
                "weighted_mode": filters.weighted_mode,
            },
        ).df()

    def get_pitcher_arsenal(self, pitcher_ids: list[int], filters: QueryFilters) -> pd.DataFrame:
        if not pitcher_ids:
            return pd.DataFrame()
        return self.conn.execute(
            """
            SELECT *
            FROM pitcher_arsenal
            WHERE pitcher_id IN $pitcher_ids
              AND split_key = $split_key
              AND recent_window = $recent_window
              AND weighted_mode = $weighted_mode
            ORDER BY pitcher_name, usage_pct DESC
            """,
            {
                "pitcher_ids": pitcher_ids,
                "split_key": filters.split,
                "recent_window": filters.recent_window,
                "weighted_mode": filters.weighted_mode,
            },
        ).df()

    def get_pitcher_summary_by_hand(self, pitcher_ids: list[int], filters: QueryFilters) -> pd.DataFrame:
        if not pitcher_ids:
            return pd.DataFrame()
        return self.conn.execute(
            """
            SELECT *
            FROM pitcher_summary_by_hand
            WHERE pitcher_id IN $pitcher_ids
              AND split_key = $split_key
              AND recent_window = $recent_window
              AND weighted_mode = $weighted_mode
            ORDER BY pitcher_name, batter_side_key
            """,
            {
                "pitcher_ids": pitcher_ids,
                "split_key": filters.split,
                "recent_window": filters.recent_window,
                "weighted_mode": filters.weighted_mode,
            },
        ).df()

    def get_pitcher_arsenal_by_hand(self, pitcher_ids: list[int], filters: QueryFilters) -> pd.DataFrame:
        if not pitcher_ids:
            return pd.DataFrame()
        return self.conn.execute(
            """
            SELECT *
            FROM pitcher_arsenal_by_hand
            WHERE pitcher_id IN $pitcher_ids
              AND split_key = $split_key
              AND recent_window = $recent_window
              AND weighted_mode = $weighted_mode
            ORDER BY pitcher_name, batter_side_key, usage_pct DESC
            """,
            {
                "pitcher_ids": pitcher_ids,
                "split_key": filters.split,
                "recent_window": filters.recent_window,
                "weighted_mode": filters.weighted_mode,
            },
        ).df()

    def get_pitcher_usage_by_count(self, pitcher_ids: list[int], filters: QueryFilters) -> pd.DataFrame:
        if not pitcher_ids:
            return pd.DataFrame()
        return self.conn.execute(
            """
            SELECT *
            FROM pitcher_usage_by_count
            WHERE pitcher_id IN $pitcher_ids
              AND split_key = $split_key
              AND recent_window = $recent_window
              AND weighted_mode = $weighted_mode
            ORDER BY pitcher_name, batter_side_key, count_bucket, usage_pct DESC
            """,
            {
                "pitcher_ids": pitcher_ids,
                "split_key": filters.split,
                "recent_window": filters.recent_window,
                "weighted_mode": filters.weighted_mode,
            },
        ).df()

    def get_team_hitter_pool(
        self,
        team: str,
        opposing_pitcher_hand: str | None,
        filters: QueryFilters,
        roster_player_ids: list[int] | None = None,
    ) -> pd.DataFrame:
        split_key = filters.split
        if split_key == "overall" and opposing_pitcher_hand == "R":
            split_key = "vs_rhp"
        elif split_key == "overall" and opposing_pitcher_hand == "L":
            split_key = "vs_lhp"
        params = {
            "split_key": split_key,
            "recent_window": filters.recent_window,
            "weighted_mode": filters.weighted_mode,
        }
        if roster_player_ids:
            data = self.conn.execute(
                """
                SELECT *
                FROM hitter_metrics
                WHERE batter IN $roster_player_ids
                  AND split_key = $split_key
                  AND recent_window = $recent_window
                  AND weighted_mode = $weighted_mode
                ORDER BY likely_starter_score DESC NULLS LAST, xwoba DESC NULLS LAST
                """,
                {
                    **params,
                    "roster_player_ids": roster_player_ids,
                },
            ).df()
        else:
            data = self.conn.execute(
                """
                SELECT *
                FROM hitter_metrics
                WHERE team = $team
                  AND split_key = $split_key
                  AND recent_window = $recent_window
                  AND weighted_mode = $weighted_mode
                ORDER BY likely_starter_score DESC NULLS LAST, xwoba DESC NULLS LAST
                """,
                {
                    **params,
                    "team": team,
                },
            ).df()
        if filters.likely_starters_only and "likely_starter_score" in data:
            data = data.loc[data["likely_starter_score"].fillna(0) > 0]
        return data

    def run_explorer_query(
        self,
        entity_type: str,
        search_text: str = "",
        split_key: str = "overall",
        recent_window: str = "season",
        weighted_mode: str = "weighted",
        limit: int = 100,
    ) -> pd.DataFrame:
        table = "hitter_metrics" if entity_type == "hitters" else "pitcher_metrics"
        name_col = "hitter_name" if entity_type == "hitters" else "pitcher_name"
        search_pattern = f"%{search_text.lower()}%"
        return self.conn.execute(
            f"""
            SELECT *
            FROM {table}
            WHERE split_key = $split_key
              AND recent_window = $recent_window
              AND weighted_mode = $weighted_mode
              AND LOWER({name_col}) LIKE $search_pattern
            ORDER BY xwoba DESC NULLS LAST
            LIMIT $limit
            """,
            {
                "split_key": split_key,
                "recent_window": recent_window,
                "weighted_mode": weighted_mode,
                "search_pattern": search_pattern,
                "limit": limit,
            },
        ).df()


def load_remote_parquet(base_path: str, filename: str, columns: list[str] | None = None) -> pd.DataFrame:
    path = f"{base_path.rstrip('/')}/{filename}"
    requested = [col for col in (columns or []) if col]
    if requested and duckdb is not None:
        try:
            conn = duckdb.connect()
            try:
                select_cols = ", ".join(f'"{col}"' for col in requested)
                return conn.execute(f"SELECT {select_cols} FROM read_parquet('{path}')").df()
            finally:
                conn.close()
        except Exception:
            pass
    try:
        return pd.read_parquet(path, columns=requested or None)
    except Exception:
        if requested:
            return pd.read_parquet(path)
        raise


def _filter_combo(frame: pd.DataFrame, split: str, recent_window: str, weighted_mode: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    output = frame.copy()
    for column, value in {
        "split_key": split,
        "recent_window": recent_window,
        "weighted_mode": weighted_mode,
    }.items():
        if column in output.columns:
            output = output.loc[output[column].astype(str).eq(str(value))]
    return output.reset_index(drop=True)


def load_remote_parquet_bundle(
    specs: dict[str, tuple[str, str, list[str] | None]],
    *,
    max_workers: int = 8,
) -> dict[str, pd.DataFrame]:
    if not specs:
        return {}
    if len(specs) == 1:
        key, (base_path, filename, columns) = next(iter(specs.items()))
        return {key: load_remote_parquet(base_path, filename, columns=columns)}

    workers = max(1, min(max_workers, len(specs)))
    results: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(load_remote_parquet, base_path, filename, columns): key
            for key, (base_path, filename, columns) in specs.items()
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results


def load_remote_game_bundle(base_url: str, target_date: date, game_pk: int) -> dict[str, pd.DataFrame]:
    game_base = f"{base_url.rstrip('/')}/daily/{target_date.isoformat()}/games/{game_pk}"
    bundle: dict[str, pd.DataFrame] = {}
    for name in ["matchup", "rolling", "pitcher_detail", "zones", "pitch_shape", "exports"]:
        try:
            bundle[name] = load_remote_parquet(game_base, f"{name}.parquet")
        except Exception:
            bundle[name] = pd.DataFrame()
    return bundle
