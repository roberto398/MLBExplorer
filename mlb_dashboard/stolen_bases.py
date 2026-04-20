from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from .config import AppConfig
from .dashboard_views import apply_projected_lineup
from .local_store import read_latest_prop_odds_snapshot
from .odds_service import american_to_implied_prob, build_props_board_payload_from_rows


SB_MARKET = "batter_stolen_bases"
SUMMARY_PRIOR_OPPORTUNITIES = 50.0
SUCCESS_PRIOR_ATTEMPTS = 8.0


@dataclass(frozen=True)
class StolenBaseArtifacts:
    player_profiles: pd.DataFrame
    pitcher_allowed: pd.DataFrame
    catcher_allowed: pd.DataFrame
    team_allowed: pd.DataFrame
    board: pd.DataFrame
    odds: pd.DataFrame
    recent_log: pd.DataFrame


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _num(series: object) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce")
    return pd.Series(dtype="float64")


def _text(series: object, index: pd.Index) -> pd.Series:
    if isinstance(series, pd.Series):
        return series.fillna("").astype(str)
    return pd.Series("", index=index, dtype="object")


def _clip_score(value: pd.Series | float) -> pd.Series | float:
    if isinstance(value, pd.Series):
        return value.clip(lower=0.0, upper=100.0)
    return max(0.0, min(100.0, float(value)))


def _normalize_name(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _offense_team(frame: pd.DataFrame) -> pd.Series:
    top = _text(frame.get("inning_topbot"), frame.index).str.casefold().eq("top")
    return frame.get("away_team", pd.Series(pd.NA, index=frame.index)).where(top, frame.get("home_team", pd.Series(pd.NA, index=frame.index)))


def _defense_team(frame: pd.DataFrame) -> pd.Series:
    top = _text(frame.get("inning_topbot"), frame.index).str.casefold().eq("top")
    return frame.get("home_team", pd.Series(pd.NA, index=frame.index)).where(top, frame.get("away_team", pd.Series(pd.NA, index=frame.index)))


def _runner_for_event(row: pd.Series, steal_base: str | None) -> object:
    if steal_base == "2B":
        return row.get("on_1b")
    if steal_base == "3B":
        return row.get("on_2b")
    if steal_base == "H":
        return row.get("on_3b")
    for column in ("on_1b", "on_2b", "on_3b"):
        value = row.get(column)
        if pd.notna(value):
            return value
    return pd.NA


def _steal_base_from_description(description: str) -> str | None:
    text = description.casefold()
    if re.search(r"(?:steals|caught stealing)\s+(?:\(\d+\)\s+)?2(?:nd)?\s+base", text):
        return "2B"
    if re.search(r"(?:steals|caught stealing)\s+(?:\(\d+\)\s+)?3(?:rd)?\s+base", text):
        return "3B"
    if re.search(r"(?:steals|caught stealing).*(?:home|score)", text):
        return "H"
    return None


def _name_lookup(events: pd.DataFrame, rosters: pd.DataFrame | None = None) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    if not events.empty and "batter" in events.columns:
        name_col = "batter_name" if "batter_name" in events.columns else "player_name"
        team = _offense_team(events)
        lookup = events[["batter", name_col]].copy()
        lookup["team"] = team
        lookup = lookup.rename(columns={"batter": "player_id", name_col: "player_name"})
        rows.append(lookup)
    if rosters is not None and not rosters.empty and {"player_id", "player_name"}.issubset(rosters.columns):
        roster_cols = ["player_id", "player_name"] + (["team"] if "team" in rosters.columns else [])
        rows.append(rosters[roster_cols].copy())
    if not rows:
        return _empty(["player_id", "player_name", "team"])
    combined = pd.concat(rows, ignore_index=True, sort=False)
    combined["player_id"] = _num(combined["player_id"])
    combined = combined.loc[combined["player_id"].notna()].copy()
    combined["player_id"] = combined["player_id"].astype("int64")
    combined["player_name"] = combined["player_name"].replace("", pd.NA)
    combined = combined.dropna(subset=["player_name"])
    combined = combined.sort_values(["player_id", "team"], na_position="last").drop_duplicates("player_id", keep="last")
    return combined.reset_index(drop=True)


def parse_stolen_base_events(events: pd.DataFrame, rosters: pd.DataFrame | None = None) -> pd.DataFrame:
    columns = [
        "game_date",
        "game_pk",
        "runner_id",
        "runner_name",
        "team",
        "opponent",
        "pitcher_id",
        "pitcher_name",
        "catcher_id",
        "event_type",
        "steal_base",
        "is_stolen_base",
        "is_caught_stealing",
        "description",
        "at_bat_number",
        "pitch_number",
    ]
    if events.empty or "des" not in events.columns:
        return _empty(columns)
    descriptions = _text(events.get("des"), events.index)
    mask = descriptions.str.contains(r"\bsteals\b|\bcaught stealing\b", case=False, regex=True, na=False)
    if not mask.any():
        return _empty(columns)
    work = events.loc[mask].copy()
    work["description"] = _text(work.get("des"), work.index)
    work["event_type"] = work["description"].str.contains("caught stealing", case=False, na=False).map({True: "cs", False: "sb"})
    work["steal_base"] = work["description"].map(_steal_base_from_description)
    work["runner_id"] = work.apply(lambda row: _runner_for_event(row, row.get("steal_base")), axis=1)
    work["runner_id"] = _num(work["runner_id"])
    work = work.loc[work["runner_id"].notna()].copy()
    if work.empty:
        return _empty(columns)
    work["runner_id"] = work["runner_id"].astype("int64")
    work["team"] = _offense_team(work)
    work["opponent"] = _defense_team(work)
    work["pitcher_id"] = _num(work.get("pitcher"))
    work["catcher_id"] = _num(work.get("fielder_2"))
    work["pitcher_name"] = work.get("player_name", pd.Series(pd.NA, index=work.index))
    work["is_stolen_base"] = work["event_type"].eq("sb").astype(int)
    work["is_caught_stealing"] = work["event_type"].eq("cs").astype(int)
    names = _name_lookup(events, rosters)
    if not names.empty:
        runner_names = names[["player_id", "player_name"]].rename(columns={"player_name": "runner_name"})
        work = work.merge(runner_names, left_on="runner_id", right_on="player_id", how="left")
    else:
        work["runner_name"] = pd.NA
    dedupe_keys = ["game_pk", "at_bat_number", "pitch_number", "runner_id", "event_type", "steal_base"]
    work = work.drop_duplicates(subset=[column for column in dedupe_keys if column in work.columns], keep="first")
    for column in columns:
        if column not in work.columns:
            work[column] = pd.NA
    return work[columns].reset_index(drop=True)


def _build_opportunities(events: pd.DataFrame) -> pd.DataFrame:
    columns = ["game_date", "game_pk", "runner_id", "base", "team", "opponent", "pitcher_id", "catcher_id"]
    if events.empty:
        return _empty(columns)
    frames: list[pd.DataFrame] = []
    base_specs = [
        ("on_1b", "on_2b", "1B"),
        ("on_2b", "on_3b", "2B"),
        ("on_3b", None, "3B"),
    ]
    for runner_col, block_col, base in base_specs:
        if runner_col not in events.columns:
            continue
        mask = _num(events.get(runner_col)).notna()
        if block_col and block_col in events.columns:
            mask &= _num(events.get(block_col)).isna()
        subset = events.loc[mask].copy()
        if subset.empty:
            continue
        frame = pd.DataFrame(
            {
                "game_date": subset.get("game_date"),
                "game_pk": subset.get("game_pk"),
                "runner_id": _num(subset.get(runner_col)),
                "base": base,
                "team": _offense_team(subset),
                "opponent": _defense_team(subset),
                "pitcher_id": _num(subset.get("pitcher")),
                "catcher_id": _num(subset.get("fielder_2")),
                "at_bat_number": subset.get("at_bat_number"),
                "pitch_number": subset.get("pitch_number"),
            }
        )
        frames.append(frame)
    if not frames:
        return _empty(columns)
    opportunities = pd.concat(frames, ignore_index=True, sort=False)
    opportunities = opportunities.loc[opportunities["runner_id"].notna()].copy()
    opportunities["runner_id"] = opportunities["runner_id"].astype("int64")
    opportunities = opportunities.drop_duplicates(["game_pk", "at_bat_number", "pitch_number", "runner_id", "base"], keep="first")
    return opportunities.reset_index(drop=True)


def _league_rates(events: pd.DataFrame, opportunities: pd.DataFrame) -> dict[str, float]:
    attempts = float(len(events))
    steals = float(events["is_stolen_base"].sum()) if "is_stolen_base" in events.columns else 0.0
    opps = float(len(opportunities))
    return {
        "attempt_rate": attempts / opps if opps else 0.006,
        "success_rate": steals / attempts if attempts else 0.72,
    }


def _profile_from_group(
    opportunities: pd.DataFrame,
    steal_events: pd.DataFrame,
    group_col: str,
    *,
    id_col: str,
) -> pd.DataFrame:
    columns = [
        id_col,
        "opportunities",
        "attempts",
        "stolen_bases",
        "caught_stealing",
        "attempt_rate",
        "success_rate",
    ]
    if opportunities.empty and steal_events.empty:
        return _empty(columns)
    opp = (
        opportunities.groupby(group_col, dropna=True)
        .size()
        .rename("opportunities")
        .reset_index()
        .rename(columns={group_col: id_col})
    )
    att = (
        steal_events.groupby(group_col, dropna=True)
        .agg(
            attempts=("event_type", "size"),
            stolen_bases=("is_stolen_base", "sum"),
            caught_stealing=("is_caught_stealing", "sum"),
        )
        .reset_index()
        .rename(columns={group_col: id_col})
    )
    profile = opp.merge(att, on=id_col, how="outer")
    for column in ("opportunities", "attempts", "stolen_bases", "caught_stealing"):
        profile[column] = _num(profile.get(column)).fillna(0.0)
    profile["attempt_rate"] = profile["attempts"].div(profile["opportunities"].where(profile["opportunities"].gt(0)))
    profile["success_rate"] = profile["stolen_bases"].div(profile["attempts"].where(profile["attempts"].gt(0)))
    return profile[columns].reset_index(drop=True)


def build_stolen_base_profiles(events: pd.DataFrame, rosters: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    steal_events = parse_stolen_base_events(events, rosters)
    opportunities = _build_opportunities(events)
    names = _name_lookup(events, rosters)
    player = _profile_from_group(opportunities, steal_events, "runner_id", id_col="runner_id")
    if not player.empty:
        player = player.merge(names, left_on="runner_id", right_on="player_id", how="left").drop(columns=["player_id"], errors="ignore")
        player = player.rename(columns={"player_name": "runner_name"})
    pitcher = _profile_from_group(opportunities, steal_events, "pitcher_id", id_col="pitcher_id")
    catcher = _profile_from_group(opportunities, steal_events, "catcher_id", id_col="catcher_id")
    team = _profile_from_group(opportunities, steal_events, "opponent", id_col="team")
    return player, pitcher, catcher, team, steal_events


def _shrink_rate(rate: object, sample: object, league: float, prior: float) -> float:
    sample_value = 0.0 if pd.isna(sample) else float(sample)
    rate_value = league if pd.isna(rate) else float(rate)
    return ((rate_value * sample_value) + (league * prior)) / (sample_value + prior) if sample_value + prior > 0 else league


def _reach_base_score(row: pd.Series) -> float:
    xwoba = row.get("xwoba")
    if pd.isna(xwoba):
        return 50.0
    return float(_clip_score(((float(xwoba) - 0.285) / (0.405 - 0.285)) * 100.0))


def _format_american(value: object) -> str:
    if pd.isna(value):
        return "-"
    try:
        return f"{int(round(float(value))):+d}"
    except (TypeError, ValueError):
        return "-"


def _daily_odds_rows(config: AppConfig, target_date: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = read_latest_prop_odds_snapshot(config, target_date, markets=(SB_MARKET,))
    if raw.empty:
        return _empty(["player", "team", "game", "best_price", "best_books", "line", "implied_probability"]), raw
    payload = build_props_board_payload_from_rows(raw)
    board = payload.board.copy()
    if board.empty:
        return _empty(["player", "team", "game", "best_price", "best_books", "line", "implied_probability"]), raw
    over_mask = board.get("side", pd.Series("", index=board.index)).fillna("").astype(str).str.contains("over|yes|stolen", case=False, regex=True)
    if over_mask.any():
        board = board.loc[over_mask].copy()
    board["implied_probability"] = board["best_price"].map(american_to_implied_prob)
    board["player_key"] = board["player"].map(_normalize_name)
    board["team_key"] = board.get("team", pd.Series("", index=board.index)).fillna("").astype(str)
    return board, raw


def _find_hitter_pool(hitter_metrics: pd.DataFrame, rosters: pd.DataFrame, team: str) -> pd.DataFrame:
    if hitter_metrics.empty:
        return pd.DataFrame()
    pool = hitter_metrics.loc[
        hitter_metrics.get("split_key", "").eq("overall")
        & hitter_metrics.get("recent_window", "").eq("season")
        & hitter_metrics.get("weighted_mode", "").eq("weighted")
    ].copy()
    if pool.empty:
        return pd.DataFrame()
    roster = rosters.loc[rosters["team"].eq(team), ["player_id", "player_name"]].drop_duplicates("player_id") if not rosters.empty else pd.DataFrame()
    if not roster.empty:
        pool = pool.loc[pool["batter"].isin(roster["player_id"])].copy()
        pool = pool.merge(roster, left_on="batter", right_on="player_id", how="left")
        pool["hitter_name"] = pool.get("hitter_name").fillna(pool["player_name"])
        pool = pool.drop(columns=["player_id", "player_name"], errors="ignore")
    else:
        pool = pool.loc[pool.get("team", pd.Series("", index=pool.index)).eq(team)].copy()
    pool["team"] = team
    return pool


def build_daily_stolen_base_board(
    config: AppConfig,
    target_date: date,
    schedule: list[dict],
    rosters: pd.DataFrame,
    hitter_metrics: pd.DataFrame,
    player_profiles: pd.DataFrame,
    pitcher_allowed: pd.DataFrame,
    catcher_allowed: pd.DataFrame,
    team_allowed: pd.DataFrame,
    recent_log: pd.DataFrame,
    rotowire_lineups: dict[str, dict[str, object]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    board_columns = [
        "slate_date",
        "game_pk",
        "game",
        "team",
        "opponent",
        "player",
        "batter",
        "lineup_source",
        "lineup_order",
        "projected_lineup",
        "opposing_pitcher_id",
        "opposing_pitcher_name",
        "catcher_id",
        "sb_score",
        "fair_probability",
        "best_price",
        "best_books",
        "line",
        "implied_probability",
        "edge",
        "reach_base_score",
        "runner_attempt_score",
        "opponent_allowed_score",
        "success_quality_score",
        "runner_attempt_rate",
        "runner_success_rate",
        "pitcher_allowed_attempt_rate",
        "catcher_allowed_attempt_rate",
        "team_allowed_attempt_rate",
        "recent_attempts",
        "recent_steals",
    ]
    if not schedule or hitter_metrics.empty:
        odds_board, _ = _daily_odds_rows(config, target_date)
        return _empty(board_columns), odds_board

    odds_board, _ = _daily_odds_rows(config, target_date)
    odds_lookup = odds_board.copy()
    if not odds_lookup.empty:
        odds_lookup = odds_lookup.sort_values("best_price", ascending=False, na_position="last")
    league = _league_rates(recent_log, _build_opportunities(pd.DataFrame()))
    if not player_profiles.empty:
        total_attempts = float(_num(player_profiles.get("attempts")).sum())
        total_opps = float(_num(player_profiles.get("opportunities")).sum())
        total_sb = float(_num(player_profiles.get("stolen_bases")).sum())
        league = {
            "attempt_rate": total_attempts / total_opps if total_opps else 0.006,
            "success_rate": total_sb / total_attempts if total_attempts else 0.72,
        }
    player_map = player_profiles.set_index("runner_id") if not player_profiles.empty and "runner_id" in player_profiles.columns else pd.DataFrame()
    pitcher_map = pitcher_allowed.set_index("pitcher_id") if not pitcher_allowed.empty and "pitcher_id" in pitcher_allowed.columns else pd.DataFrame()
    catcher_map = catcher_allowed.set_index("catcher_id") if not catcher_allowed.empty and "catcher_id" in catcher_allowed.columns else pd.DataFrame()
    team_map = team_allowed.set_index("team") if not team_allowed.empty and "team" in team_allowed.columns else pd.DataFrame()

    rows: list[dict] = []
    for game in schedule:
        game_label = f"{game.get('away_team')} @ {game.get('home_team')}"
        sides = [
            (game.get("away_team"), game.get("home_team"), game.get("home_probable_pitcher_id"), game.get("home_probable_pitcher_name")),
            (game.get("home_team"), game.get("away_team"), game.get("away_probable_pitcher_id"), game.get("away_probable_pitcher_name")),
        ]
        for team, opponent, opposing_pitcher_id, opposing_pitcher_name in sides:
            hitters = _find_hitter_pool(hitter_metrics, rosters, str(team))
            hitters = apply_projected_lineup(hitters, str(team), rotowire_lineups)
            if hitters.empty:
                continue
            for _, hitter in hitters.iterrows():
                batter = hitter.get("batter")
                if pd.isna(batter):
                    continue
                batter_id = int(batter)
                p = player_map.loc[batter_id] if not player_map.empty and batter_id in player_map.index else pd.Series(dtype="object")
                pitcher_id = int(opposing_pitcher_id) if opposing_pitcher_id and not pd.isna(opposing_pitcher_id) else None
                pitcher = pitcher_map.loc[pitcher_id] if pitcher_id is not None and not pitcher_map.empty and pitcher_id in pitcher_map.index else pd.Series(dtype="object")
                team_allowed_row = team_map.loc[str(opponent)] if not team_map.empty and str(opponent) in team_map.index else pd.Series(dtype="object")
                catcher = pd.Series(dtype="object")
                runner_attempt = _shrink_rate(p.get("attempt_rate"), p.get("opportunities"), league["attempt_rate"], SUMMARY_PRIOR_OPPORTUNITIES)
                runner_success = _shrink_rate(p.get("success_rate"), p.get("attempts"), league["success_rate"], SUCCESS_PRIOR_ATTEMPTS)
                pitcher_allowed_rate = _shrink_rate(
                    pitcher.get("attempt_rate"),
                    pitcher.get("opportunities"),
                    league["attempt_rate"],
                    SUMMARY_PRIOR_OPPORTUNITIES,
                )
                catcher_allowed_rate = _shrink_rate(
                    catcher.get("attempt_rate"),
                    catcher.get("opportunities"),
                    league["attempt_rate"],
                    SUMMARY_PRIOR_OPPORTUNITIES,
                )
                team_allowed_rate = _shrink_rate(
                    team_allowed_row.get("attempt_rate"),
                    team_allowed_row.get("opportunities"),
                    league["attempt_rate"],
                    SUMMARY_PRIOR_OPPORTUNITIES,
                )
                opponent_allowed_rate = pitcher_allowed_rate * 0.60 + team_allowed_rate * 0.40
                reach_score = _reach_base_score(hitter)
                runner_attempt_score = float(_clip_score((runner_attempt / max(league["attempt_rate"] * 3.0, 0.015)) * 100.0))
                opponent_allowed_score = float(_clip_score((opponent_allowed_rate / max(league["attempt_rate"] * 2.5, 0.012)) * 100.0))
                success_score = float(_clip_score(((runner_success - 0.52) / (0.88 - 0.52)) * 100.0))
                sb_score = float(_clip_score(reach_score * 0.30 + runner_attempt_score * 0.35 + opponent_allowed_score * 0.25 + success_score * 0.10))
                fair_probability = max(0.01, min(0.45, 0.015 + (sb_score / 100.0) * 0.30))
                player_name = hitter.get("hitter_name") or p.get("runner_name") or str(batter_id)
                odds_row = pd.Series(dtype="object")
                player_key = _normalize_name(player_name)
                if not odds_lookup.empty:
                    player_odds = odds_lookup.loc[odds_lookup["player_key"].eq(player_key)].copy()
                    if not player_odds.empty:
                        team_odds = player_odds.loc[player_odds["team_key"].eq(str(team))]
                        odds_row = (team_odds if not team_odds.empty else player_odds).iloc[0]
                implied = odds_row.get("implied_probability")
                rows.append(
                    {
                        "slate_date": target_date,
                        "game_pk": game.get("game_pk"),
                        "game": game_label,
                        "team": team,
                        "opponent": opponent,
                        "player": player_name,
                        "batter": batter_id,
                        "lineup_source": hitter.get("lineup_source"),
                        "lineup_order": hitter.get("lineup_order"),
                        "projected_lineup": bool(pd.notna(hitter.get("lineup_order"))),
                        "opposing_pitcher_id": pitcher_id,
                        "opposing_pitcher_name": opposing_pitcher_name,
                        "catcher_id": pd.NA,
                        "sb_score": sb_score,
                        "fair_probability": fair_probability,
                        "best_price": odds_row.get("best_price", pd.NA),
                        "best_books": odds_row.get("best_books", pd.NA),
                        "line": odds_row.get("line", pd.NA),
                        "implied_probability": implied,
                        "edge": fair_probability - float(implied) if pd.notna(implied) else pd.NA,
                        "reach_base_score": reach_score,
                        "runner_attempt_score": runner_attempt_score,
                        "opponent_allowed_score": opponent_allowed_score,
                        "success_quality_score": success_score,
                        "runner_attempt_rate": runner_attempt,
                        "runner_success_rate": runner_success,
                        "pitcher_allowed_attempt_rate": pitcher_allowed_rate,
                        "catcher_allowed_attempt_rate": catcher_allowed_rate if not catcher.empty else pd.NA,
                        "team_allowed_attempt_rate": team_allowed_rate,
                        "recent_attempts": p.get("attempts", 0),
                        "recent_steals": p.get("stolen_bases", 0),
                    }
                )
    board = pd.DataFrame(rows)
    if board.empty:
        return _empty(board_columns), odds_board
    board["odds_display"] = board["best_price"].map(_format_american)
    board = board.sort_values(["sb_score", "fair_probability"], ascending=[False, False], na_position="last").reset_index(drop=True)
    for column in board_columns:
        if column not in board.columns:
            board[column] = pd.NA
    extra = [column for column in board.columns if column not in board_columns]
    return board[board_columns + extra], odds_board


def build_stolen_base_artifacts(
    config: AppConfig,
    target_date: date,
    schedule: list[dict],
    rosters: pd.DataFrame,
    hitter_metrics: pd.DataFrame,
    raw_statcast: pd.DataFrame,
    rotowire_lineups: dict[str, dict[str, object]] | None = None,
) -> StolenBaseArtifacts:
    if raw_statcast.empty:
        player, pitcher, catcher, team, recent = (
            _empty(["runner_id"]),
            _empty(["pitcher_id"]),
            _empty(["catcher_id"]),
            _empty(["team"]),
            _empty(["runner_id"]),
        )
    else:
        history = raw_statcast.loc[pd.to_datetime(raw_statcast.get("game_date"), errors="coerce").dt.date < target_date].copy()
        player, pitcher, catcher, team, recent = build_stolen_base_profiles(history, rosters)
    board, odds = build_daily_stolen_base_board(
        config,
        target_date,
        schedule,
        rosters,
        hitter_metrics,
        player,
        pitcher,
        catcher,
        team,
        recent,
        rotowire_lineups,
    )
    return StolenBaseArtifacts(
        player_profiles=player,
        pitcher_allowed=pitcher,
        catcher_allowed=catcher,
        team_allowed=team,
        board=board,
        odds=odds,
        recent_log=recent,
    )


def write_stolen_base_artifacts(config: AppConfig, target_date: date, artifacts: StolenBaseArtifacts) -> None:
    config.reusable_dir.mkdir(parents=True, exist_ok=True)
    artifacts.player_profiles.to_parquet(config.reusable_dir / "stolen_base_player_profiles.parquet", index=False)
    artifacts.pitcher_allowed.to_parquet(config.reusable_dir / "stolen_base_pitcher_allowed.parquet", index=False)
    artifacts.catcher_allowed.to_parquet(config.reusable_dir / "stolen_base_catcher_allowed.parquet", index=False)
    artifacts.team_allowed.to_parquet(config.reusable_dir / "stolen_base_team_allowed.parquet", index=False)
    target_dir = config.daily_dir / target_date.isoformat() / "stolen_bases"
    target_dir.mkdir(parents=True, exist_ok=True)
    artifacts.board.to_parquet(target_dir / "board.parquet", index=False)
    artifacts.odds.to_parquet(target_dir / "odds.parquet", index=False)
    artifacts.recent_log.to_parquet(target_dir / "recent_log.parquet", index=False)


def load_local_stolen_base_artifacts(config: AppConfig, target_date: date) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_dir = config.daily_dir / target_date.isoformat() / "stolen_bases"
    board = pd.read_parquet(target_dir / "board.parquet") if (target_dir / "board.parquet").exists() else pd.DataFrame()
    odds = pd.read_parquet(target_dir / "odds.parquet") if (target_dir / "odds.parquet").exists() else pd.DataFrame()
    recent = pd.read_parquet(target_dir / "recent_log.parquet") if (target_dir / "recent_log.parquet").exists() else pd.DataFrame()
    return board, odds, recent


def load_remote_stolen_base_artifacts(base_url: str, target_date: date) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    day = target_date.isoformat()
    root = base_url.rstrip("/") + f"/daily/{day}/stolen_bases"
    board = pd.read_parquet(f"{root}/board.parquet")
    try:
        odds = pd.read_parquet(f"{root}/odds.parquet")
    except Exception:
        odds = pd.DataFrame()
    try:
        recent = pd.read_parquet(f"{root}/recent_log.parquet")
    except Exception:
        recent = pd.DataFrame()
    return board, odds, recent
