from __future__ import annotations

import os
import re
from datetime import date, timedelta

import pandas as pd
import streamlit as st

from .branding import apply_branding_head, page_icon_path
from .config import AppConfig
from .dashboard_views import latest_built_date
from .stolen_bases import load_local_stolen_base_artifacts, load_remote_stolen_base_artifacts
from .team_logos import team_logo_data_uri
from .ui_components import render_metric_grid


def _require_admin_password() -> bool:
    required = st.secrets.get("ADMIN_PASSWORD")
    if not required:
        st.error("Stolen Bases is locked. Add ADMIN_PASSWORD to Streamlit secrets to enable this page.")
        return False
    entry = st.text_input("Stolen Bases password", type="password")
    if not entry:
        st.info("Enter the Stolen Bases password to view this page.")
        return False
    if str(entry) != str(required):
        st.error("Incorrect password.")
        return False
    return True


def _hosted_base_url() -> str:
    return os.getenv("MLB_HOSTED_BASE_URL", "").rstrip("/")


def _default_date(config: AppConfig) -> date:
    latest = latest_built_date(config.daily_dir)
    return latest or date.today()


@st.cache_data(show_spinner=False)
def _load_remote_cached(base_url: str, target_date: date) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_remote_stolen_base_artifacts(base_url, target_date)


@st.cache_data(show_spinner=False)
def _load_local_cached(target_date: date) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_local_stolen_base_artifacts(AppConfig(), target_date)


def _load_artifacts(config: AppConfig, target_date: date) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, date]:
    board, odds, recent = _load_local_cached(target_date)
    if not board.empty:
        return board, odds, recent, "local", target_date
    base_url = _hosted_base_url()
    if not base_url:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "missing", target_date
    last_error: Exception | None = None
    for offset in range(8):
        candidate = target_date - timedelta(days=offset)
        try:
            board, odds, recent = _load_remote_cached(base_url, candidate)
            if not board.empty:
                return board, odds, recent, "published", candidate
        except Exception as exc:
            last_error = exc
    if last_error:
        st.caption(f"Stolen Bases artifact probe: {last_error}")
    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "missing", target_date


def _pct(value: object) -> str:
    if pd.isna(value):
        return "-"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "-"


def _score(value: object) -> str:
    if pd.isna(value):
        return "-"
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "-"


def _format_american(value: object) -> str:
    if pd.isna(value):
        return "-"
    try:
        return f"{int(round(float(value))):+d}"
    except (TypeError, ValueError):
        return "-"


def _board_display(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    display["team_logo"] = [team_logo_data_uri(team) or str(team) for team in display.get("team", pd.Series("", index=display.index))]
    display["Opp"] = [team_logo_data_uri(team) or str(team) for team in display.get("opponent", pd.Series("", index=display.index))]
    if "best_price" in display.columns:
        display["odds_display"] = display["best_price"].map(_format_american)
    columns = [
        "player",
        "team_logo",
        "game",
        "Opp",
        "sb_score",
        "fair_probability",
        "odds_display",
        "edge",
        "runner_attempt_rate",
        "runner_success_rate",
        "pitcher_allowed_attempt_rate",
        "catcher_allowed_attempt_rate",
        "team_allowed_attempt_rate",
        "best_books",
        "lineup_source",
    ]
    for column in columns:
        if column not in display.columns:
            display[column] = pd.NA
    return display[columns + [column for column in ("batter", "opposing_pitcher_name") if column in display.columns]]


def _filter_board(frame: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.title("Stolen Bases Filters")
    work = frame.copy()
    teams = sorted(work["team"].dropna().astype(str).unique().tolist()) if "team" in work.columns else []
    books = sorted(
        {
            item.strip()
            for value in work.get("best_books", pd.Series(dtype="object")).dropna().astype(str)
            for item in value.split(",")
            if item.strip()
        }
    )
    selected_teams = st.sidebar.multiselect("Teams", teams)
    min_score = st.sidebar.slider("Minimum SB Score", 0, 100, 0, 1)
    odds_only = st.sidebar.checkbox("Odds only", value=False)
    selected_books = st.sidebar.multiselect("Sportsbooks", books)
    projected_only = st.sidebar.checkbox("Projected lineup only", value=False)
    if selected_teams:
        work = work.loc[work["team"].isin(selected_teams)]
    if "sb_score" in work.columns:
        work = work.loc[pd.to_numeric(work["sb_score"], errors="coerce").fillna(0).ge(min_score)]
    if odds_only and "best_price" in work.columns:
        work = work.loc[pd.to_numeric(work["best_price"], errors="coerce").notna()]
    if selected_books and "best_books" in work.columns:
        pattern = "|".join(re.escape(book) for book in selected_books)
        work = work.loc[work["best_books"].fillna("").astype(str).str.contains(pattern, case=False, regex=True)]
    if projected_only and "projected_lineup" in work.columns:
        work = work.loc[work["projected_lineup"].fillna(False).astype(bool)]
    return work


def _render_detail(row: pd.Series, board: pd.DataFrame, odds: pd.DataFrame, recent: pd.DataFrame) -> None:
    st.subheader("Player Detail")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SB Score", _score(row.get("sb_score")))
    c2.metric("Fair Probability", _pct(row.get("fair_probability")))
    c3.metric("Best Odds", _format_american(row.get("best_price")))
    c4.metric("Edge", _pct(row.get("edge")))

    profile = pd.DataFrame(
        [
            {
                "Player": row.get("player"),
                "Team": team_logo_data_uri(str(row.get("team"))) or row.get("team"),
                "Opponent": team_logo_data_uri(str(row.get("opponent"))) or row.get("opponent"),
                "Opposing Pitcher": row.get("opposing_pitcher_name"),
                "Runner Attempt Rate": _pct(row.get("runner_attempt_rate")),
                "Runner Success Rate": _pct(row.get("runner_success_rate")),
                "Pitcher Allowed Rate": _pct(row.get("pitcher_allowed_attempt_rate")),
                "Team Allowed Rate": _pct(row.get("team_allowed_attempt_rate")),
                "Recent Attempts": row.get("recent_attempts"),
                "Recent Steals": row.get("recent_steals"),
            }
        ]
    )
    render_metric_grid(profile, key="sb-player-profile", height=120, use_lightweight=True)

    batter = row.get("batter")
    if pd.notna(batter) and not recent.empty and "runner_id" in recent.columns:
        log = recent.loc[pd.to_numeric(recent["runner_id"], errors="coerce").eq(float(batter))].copy()
        if not log.empty:
            log = log.sort_values("game_date", ascending=False).head(20)
            display_log = log[["game_date", "team", "opponent", "event_type", "steal_base", "pitcher_name", "description"]].copy()
            st.markdown("##### Recent Steal Log")
            st.dataframe(display_log, hide_index=True, use_container_width=True, height=220)

    if not odds.empty and "player_name" in odds.columns:
        player_name = str(row.get("player", "")).casefold()
        player_odds = odds.loc[odds["player_name"].fillna("").astype(str).str.casefold().eq(player_name)].copy()
        if not player_odds.empty:
            columns = [column for column in ["sportsbook", "selection_label", "line", "odds_american", "commence_time"] if column in player_odds.columns]
            st.markdown("##### Odds By Book")
            st.dataframe(player_odds[columns], hide_index=True, use_container_width=True, height=220)


def main() -> None:
    st.set_page_config(page_title="Kasper Stolen Bases", page_icon=page_icon_path(), layout="wide")
    apply_branding_head()
    st.title("Stolen Bases")
    st.caption("Admin betting board for steal targets and batter stolen-base odds.")
    if not _require_admin_password():
        return

    config = AppConfig()
    target_date = st.sidebar.date_input("Slate date", value=_default_date(config))
    board, odds, recent, source, resolved_date = _load_artifacts(config, target_date)
    if board.empty:
        st.warning("Stolen Bases artifacts are not published for this slate yet. Rebuild and publish artifacts.")
        return
    if resolved_date != target_date:
        st.warning(f"No Stolen Bases artifact found for {target_date.isoformat()}. Showing {resolved_date.isoformat()} from {source}.")
    else:
        st.caption(f"Source: {source} | Slate: {resolved_date.isoformat()}")

    filtered = _filter_board(board)
    if filtered.empty:
        st.info("No stolen-base rows match these filters.")
        return
    st.markdown("### Board")
    render_metric_grid(
        _board_display(filtered),
        key="stolen-bases-board",
        height=520,
        use_lightweight=True,
        higher_is_better={
            "sb_score",
            "fair_probability",
            "edge",
            "runner_attempt_rate",
            "runner_success_rate",
            "pitcher_allowed_attempt_rate",
            "catcher_allowed_attempt_rate",
            "team_allowed_attempt_rate",
        },
    )

    options = filtered["player"].fillna("").astype(str).tolist()
    selected = st.selectbox("Select player", options, index=0)
    selected_row = filtered.loc[filtered["player"].fillna("").astype(str).eq(selected)].iloc[0]
    _render_detail(selected_row, board, odds, recent)


if __name__ == "__main__":
    main()
