from __future__ import annotations

import os
from datetime import date, timedelta
from html import escape
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pandas as pd
import streamlit as st
import streamlit.components.v1 as st_components

from .branding import apply_branding_head, page_icon_path, render_kasper_header
from .dashboard_views import (
    ARSENAL_COLUMNS,
    BATTER_SIDE_LABELS,
    BATTER_ZONE_COLUMNS,
    BEST_MATCHUP_COLUMNS,
    COUNT_BUCKET_ORDER,
    COUNT_USAGE_COLUMNS,
    FAMILY_ZONE_CONTEXT_COLUMNS,
    HITTER_PRESETS,
    HITTER_ROLLING_COLUMNS,
    MOVEMENT_ARSENAL_COLUMNS,
    PITCHER_HIGHER_IS_BETTER,
    PITCHER_LOWER_IS_BETTER,
    PITCHER_ROLLING_COLUMNS,
    PITCHER_SUMMARY_COLUMNS,
    PITCHER_ZONE_COLUMNS,
    TOP_PITCHER_COLUMNS,
    aggregate_batter_zone_map,
    aggregate_pitcher_zone_map,
    add_hitter_matchup_score,
    add_pitcher_rank_score,
    apply_roster_names,
    build_pitcher_matchup_key,
    build_zone_overlay_map,
    build_best_matchups,
    build_slate_summary_matchup_overview,
    build_slate_export_options,
    compute_family_fit_score,
    build_game_export_options,
    filter_excluded_pitchers_from_hitter_pool,
    apply_projected_lineup,
    hitter_columns_for_preset,
    pivot_count_usage,
    sort_arsenal_frame,
)
from .query_engine import load_remote_parquet, load_remote_parquet_bundle
from .rotowire_lineups import fetch_rotowire_lineups, resolve_rotowire_lineups
from .team_logos import add_matchup_logo_columns, team_logo_img_html
from .twitter_exports import TWITTER_EXPORT_DIRNAME, TWITTER_EXPORT_ZIP
from .ui_components import (
    build_pitcher_summary_table,
    render_export_hub,
    render_metric_grid,
    render_slate_export_controls,
    render_selected_game_twitter_export,
    render_sticky_logo_game_nav,
    render_zone_tool,
)


def _base_url() -> str:
    return os.getenv("MLB_HOSTED_BASE_URL", "").rstrip("/")


def _perf_enabled() -> bool:
    return os.getenv("MLB_PERF_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def _record_perf(events: list[tuple[str, float]], label: str, start_time: float) -> None:
    if _perf_enabled():
        events.append((label, perf_counter() - start_time))


def _render_perf(events: list[tuple[str, float]]) -> None:
    if not _perf_enabled() or not events:
        return
    st.caption("Perf: " + " | ".join(f"{label} {duration:.2f}s" for label, duration in events))


def _is_mobile_request() -> bool:
    try:
        context = getattr(st, "context", None)
        headers = getattr(context, "headers", None)
        if headers is None:
            return False
        user_agent = headers.get("User-Agent", "") or headers.get("user-agent", "")
        agent = str(user_agent).lower()
        return any(token in agent for token in ("iphone", "android", "ipad", "mobile", "blackberry", "opera mini", "windows phone"))
    except Exception:
        return False


def _render_hosted_grid(
    frame: pd.DataFrame,
    key: str,
    mobile_safe: bool,
    height: int = 320,
    lower_is_better: set[str] | None = None,
    higher_is_better: set[str] | None = None,
    hidden_columns: set[str] | None = None,
    color_hitter_confidence: bool = False,
) -> pd.DataFrame:
    return render_metric_grid(
        frame,
        key=key,
        height=height,
        lower_is_better=lower_is_better,
        higher_is_better=higher_is_better,
        use_lightweight=True,
        hidden_columns=hidden_columns,
        color_hitter_confidence=color_hitter_confidence,
    )


def _present_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def _sorted_unique(values: list[str]) -> list[str]:
    return sorted({value for value in values if value})


HITTER_REQUIRED_COLUMNS = _sorted_unique(
    [
        *[col for preset in HITTER_PRESETS.values() for col in preset],
        *BEST_MATCHUP_COLUMNS,
        "batter",
        "batter_id",
        "game",
        "game_pk",
        "hitter_name",
        "split_key",
        "recent_window",
        "weighted_mode",
        "team",
        "pitch_count",
        "bip",
        "likely_starter_score",
        "zone_fit_score",
        "hr_form",
        "khr_score",
        "hr_form_pct",
    ]
)

PITCHER_REQUIRED_COLUMNS = _sorted_unique(
    [
        *TOP_PITCHER_COLUMNS,
        *PITCHER_SUMMARY_COLUMNS,
        "pitcher_id",
        "game_pk",
        "split_key",
        "recent_window",
        "weighted_mode",
        "team",
        "game",
    ]
)

PITCHER_SUMMARY_REQUIRED_COLUMNS = _sorted_unique(
    [
        *PITCHER_SUMMARY_COLUMNS,
        "pitcher_id",
        "game_pk",
        "team",
        "batter_side_key",
    ]
)

ARSENAL_REQUIRED_COLUMNS = _sorted_unique(
    [
        *ARSENAL_COLUMNS,
        "pitcher_id",
        "game_pk",
        "team",
        "batter_side_key",
    ]
)

COUNT_USAGE_REQUIRED_COLUMNS = _sorted_unique(
    [
        *COUNT_USAGE_COLUMNS,
        "pitcher_id",
        "game_pk",
        "team",
        "batter_side_key",
    ]
)

BATTER_ZONE_REQUIRED_COLUMNS = _sorted_unique(
    [
        *BATTER_ZONE_COLUMNS,
        "batter_id",
        "pitcher_hand_key",
    ]
)

PITCHER_ZONE_REQUIRED_COLUMNS = _sorted_unique(
    [
        *PITCHER_ZONE_COLUMNS,
        "pitcher_id",
        "batter_side_key",
    ]
)

BATTER_FAMILY_REQUIRED_COLUMNS = _sorted_unique(
    [
        "batter_id",
        "pitch_family",
        "zone_bucket",
        "weighted_sample_size",
        "prior_weight_share",
        "usage_rate_overall",
        "damage_rate",
        "xwoba",
    ]
)

PITCHER_FAMILY_REQUIRED_COLUMNS = _sorted_unique(
    [
        *FAMILY_ZONE_CONTEXT_COLUMNS,
        "pitcher_id",
    ]
)

MOVEMENT_REQUIRED_COLUMNS = _sorted_unique(
    [
        *MOVEMENT_ARSENAL_COLUMNS,
        "pitcher_id",
    ]
)

GAME_SECTION_OPTIONS = ["Matchup", "Rolling", "Pitcher Zones", "Hitter Zones", "Exports"]

HITTER_ROLLING_REQUIRED_COLUMNS = _sorted_unique(
    [
        *HITTER_ROLLING_COLUMNS,
        "player_name",
        "rolling_window",
    ]
)

PITCHER_ROLLING_REQUIRED_COLUMNS = _sorted_unique(
    [
        *PITCHER_ROLLING_COLUMNS,
        "player_name",
        "rolling_window",
    ]
)

def _hitter_table_columns(frame: pd.DataFrame, columns: list[str]) -> tuple[list[str], set[str]]:
    present = _present_columns(frame, columns)
    hidden: set[str] = set()
    if "hitter_name" in present:
        for sample_column in ("pitch_count", "bip"):
            if sample_column in frame.columns and sample_column not in present:
                present.append(sample_column)
                hidden.add(sample_column)
    if "hr_form" in present and "hr_form_pct" in frame.columns and "hr_form_pct" not in present:
        present.append("hr_form_pct")
        hidden.add("hr_form_pct")
    return present, hidden


def _render_hitter_confidence_legend() -> None:
    chips = [
        ("High", "#166534"),
        ("Medium", "#1f2937"),
        ("Thin", "#b45309"),
        ("Very Thin", "#b91c1c"),
    ]
    chip_html = "".join(
        (
            "<span style='display:inline-flex;align-items:center;gap:8px;"
            "padding:4px 10px;border-radius:999px;border:1px solid rgba(15,23,42,0.18);"
            "background:#f8fafc;margin-right:8px;font-size:13px;font-weight:600;'>"
            f"<span style='width:10px;height:10px;border-radius:999px;background:{color};display:inline-block;'></span>"
            f"{label}</span>"
        )
        for label, color in chips
    )
    st.markdown(
        "<div style='font-size:13px;color:#334155;margin-bottom:8px;'>"
        "<strong>Player name sample size</strong> "
        "<span style='color:#64748b'>(legend applies to player name text color)</span>"
        "</div>"
        f"<div style='margin-bottom:12px;'>{chip_html}</div>",
        unsafe_allow_html=True,
    )


def _empty_like(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.head(0).copy() if not frame.empty else frame.copy()


def _build_family_fit_board(
    hitters: pd.DataFrame,
    batter_family_zone_profiles: pd.DataFrame,
    pitcher_family_zone_context: pd.DataFrame,
    pitcher_id: int | None,
) -> pd.DataFrame:
    if hitters.empty or pitcher_id is None or batter_family_zone_profiles.empty or pitcher_family_zone_context.empty:
        return pd.DataFrame(columns=["hitter_name", "team", "family_fit_score", "matchup_score", "ceiling_score", "xwoba"])
    rows: list[dict] = []
    for _, row in hitters.iterrows():
        batter_id = pd.to_numeric(pd.Series([row.get("batter")]), errors="coerce").iloc[0]
        if pd.isna(batter_id):
            continue
        fit_score = compute_family_fit_score(
            batter_family_zone_profiles,
            pitcher_family_zone_context,
            int(batter_id),
            int(pitcher_id),
        )
        rows.append(
            {
                "hitter_name": row.get("hitter_name"),
                "team": row.get("team"),
                "family_fit_score": None if fit_score is None else float(fit_score) * 100.0,
                "matchup_score": row.get("matchup_score"),
                "ceiling_score": row.get("ceiling_score"),
                "xwoba": row.get("xwoba"),
            }
        )
    return pd.DataFrame(rows).sort_values(["family_fit_score", "matchup_score"], ascending=[False, False], na_position="last")


def _render_pitch_shape_context(
    game_pk: int,
    team_label: str,
    pitcher_row: pd.DataFrame,
    movement_arsenal: pd.DataFrame,
    family_context: pd.DataFrame,
    opposing_hitters: pd.DataFrame,
    batter_family_zone_profiles: pd.DataFrame,
    pitcher_family_zone_context: pd.DataFrame,
    mobile_safe: bool,
) -> None:
    if pitcher_row.empty:
        st.info("No pitch-shape context available.")
        return
    pitcher_id = int(pitcher_row["pitcher_id"].iloc[0])
    movement_detail = movement_arsenal.loc[movement_arsenal["pitcher_id"] == pitcher_id].copy() if not movement_arsenal.empty else pd.DataFrame()
    family_detail = family_context.loc[family_context["pitcher_id"] == pitcher_id].copy() if not family_context.empty else pd.DataFrame()
    family_board = _build_family_fit_board(opposing_hitters, batter_family_zone_profiles, pitcher_family_zone_context, pitcher_id)

    st.markdown(f"##### {team_label} Pitch Shape")
    if movement_detail.empty:
        st.info("No weighted movement arsenal data available.")
    else:
        _render_hosted_grid(
            movement_detail[[column for column in MOVEMENT_ARSENAL_COLUMNS if column in movement_detail.columns]],
            key=f"movement-arsenal-hosted-{game_pk}-{team_label}",
            mobile_safe=mobile_safe,
            height=230,
            higher_is_better={"usage_rate", "avg_velocity", "avg_spin_rate", "avg_extension", "avg_pfx_x", "avg_pfx_z"},
        )
    if family_detail.empty:
        st.info("No weighted family-zone profile available.")
    else:
        _render_hosted_grid(
            family_detail[[column for column in FAMILY_ZONE_CONTEXT_COLUMNS if column in family_detail.columns]],
            key=f"family-context-hosted-{game_pk}-{team_label}",
            mobile_safe=mobile_safe,
            height=210,
            lower_is_better={"prior_weight_share", "damage_allowed_rate", "xwoba_allowed"},
            higher_is_better={"usage_rate_overall", "whiff_rate", "called_strike_rate"},
        )
    if family_board.empty:
        st.info("No opposing-hitter family fit context available.")
    else:
        _render_hosted_grid(
            family_board.head(6),
            key=f"family-fit-board-hosted-{game_pk}-{team_label}",
            mobile_safe=mobile_safe,
            height=210,
            higher_is_better={"family_fit_score", "matchup_score", "ceiling_score", "xwoba"},
        )


def _frame_by_pitcher_id(frame: pd.DataFrame) -> dict[object, pd.DataFrame]:
    if frame.empty or "pitcher_id" not in frame.columns:
        return {}
    return {pitcher_id: group.copy() for pitcher_id, group in frame.groupby("pitcher_id", sort=False)}


@st.cache_data(show_spinner=False)
def _load_slate_artifacts(base_url: str, target_date: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    day = target_date.isoformat()
    daily_base = f"{base_url}/daily/{day}"
    loaded = load_remote_parquet_bundle(
        {
            "slate": (daily_base, "slate.parquet", None),
            "rosters": (daily_base, "rosters.parquet", None),
        }
    )
    return loaded["slate"], loaded["rosters"]


@st.cache_data(show_spinner=False)
def _load_top_board_artifacts(base_url: str, target_date: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    day = target_date.isoformat()
    daily_base = f"{base_url}/daily/{day}"
    loaded = load_remote_parquet_bundle(
        {
            "top_hitters": (daily_base, "top_slate_hitters.parquet", None),
            "top_pitchers": (daily_base, "top_slate_pitchers.parquet", None),
        }
    )
    return loaded["top_hitters"], loaded["top_pitchers"]


@st.cache_data(show_spinner=False)
def _load_filtered_top_board_artifacts(
    base_url: str,
    target_date: date,
    split: str,
    recent_window: str,
    weighted_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    day = target_date.isoformat()
    daily_base = f"{base_url}/daily/{day}"
    board_base = f"{daily_base}/top_boards"
    hitter_file = f"{split}__{recent_window}__{weighted_mode}__hitters.parquet"
    pitcher_file = f"{split}__{recent_window}__{weighted_mode}__pitchers.parquet"
    try:
        loaded = load_remote_parquet_bundle(
            {
                "top_hitters": (board_base, hitter_file, None),
                "top_pitchers": (board_base, pitcher_file, None),
            }
        )
        return loaded["top_hitters"], loaded["top_pitchers"]
    except Exception:
        top_hitters, top_pitchers = _load_top_board_artifacts(base_url, target_date)
        return (
            _filter_top_board(top_hitters, split, recent_window, weighted_mode),
            _filter_top_board(top_pitchers, split, recent_window, weighted_mode),
        )


@st.cache_data(show_spinner=False)
def _load_slate_summary_artifact(
    base_url: str,
    target_date: date,
    split: str,
    recent_window: str,
    weighted_mode: str,
) -> pd.DataFrame:
    day = target_date.isoformat()
    daily_base = f"{base_url}/daily/{day}"
    try:
        frame = load_remote_parquet(daily_base, "slate_summary.parquet")
    except Exception:
        return pd.DataFrame()
    return _filter_top_board(frame, split, recent_window, weighted_mode)


@st.cache_data(show_spinner=False)
def _load_artifact_manifest(base_url: str, target_date: date) -> pd.DataFrame:
    day = target_date.isoformat()
    daily_base = f"{base_url}/daily/{day}"
    try:
        return load_remote_parquet(daily_base, "artifact_manifest.parquet")
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _load_core_artifacts(
    base_url: str,
    target_date: date,
) -> tuple[pd.DataFrame, ...]:
    day = target_date.isoformat()
    daily_base = f"{base_url}/daily/{day}"
    loaded = load_remote_parquet_bundle(
        {
            "slate": (daily_base, "slate.parquet", None),
            "rosters": (daily_base, "rosters.parquet", None),
            "hitters": (daily_base, "daily_hitter_metrics.parquet", HITTER_REQUIRED_COLUMNS),
            "pitchers": (daily_base, "daily_pitcher_metrics.parquet", PITCHER_REQUIRED_COLUMNS),
            "pitcher_summary_by_hand": (daily_base, "daily_pitcher_summary_by_hand.parquet", PITCHER_SUMMARY_REQUIRED_COLUMNS),
            "arsenal": (daily_base, "daily_pitcher_arsenal.parquet", ARSENAL_REQUIRED_COLUMNS),
            "arsenal_by_hand": (daily_base, "daily_pitcher_arsenal_by_hand.parquet", ARSENAL_REQUIRED_COLUMNS),
            "usage_by_count": (daily_base, "daily_pitcher_usage_by_count.parquet", COUNT_USAGE_REQUIRED_COLUMNS),
            "batter_zone_profiles": (daily_base, "daily_batter_zone_profiles.parquet", BATTER_ZONE_REQUIRED_COLUMNS),
            "pitcher_zone_profiles": (daily_base, "daily_pitcher_zone_profiles.parquet", PITCHER_ZONE_REQUIRED_COLUMNS),
        }
    )
    try:
        hitter_pitcher_exclusions = load_remote_parquet(daily_base, "hitter_pitcher_exclusions.parquet")
    except Exception:
        hitter_pitcher_exclusions = pd.DataFrame(columns=["player_id", "exclude_from_hitter_tables"])
    return (
        loaded["slate"],
        loaded["rosters"],
        loaded["hitters"],
        loaded["pitchers"],
        loaded["pitcher_summary_by_hand"],
        loaded["arsenal"],
        loaded["arsenal_by_hand"],
        loaded["usage_by_count"],
        loaded["batter_zone_profiles"],
        loaded["pitcher_zone_profiles"],
        hitter_pitcher_exclusions,
    )


@st.cache_data(show_spinner=False)
def _load_rolling_artifacts(base_url: str, target_date: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    day = target_date.isoformat()
    daily_base = f"{base_url}/daily/{day}"
    loaded = load_remote_parquet_bundle(
        {
            "hitter_rolling": (daily_base, "daily_hitter_rolling.parquet", HITTER_ROLLING_REQUIRED_COLUMNS),
            "pitcher_rolling": (daily_base, "daily_pitcher_rolling.parquet", PITCHER_ROLLING_REQUIRED_COLUMNS),
        }
    )
    return loaded["hitter_rolling"], loaded["pitcher_rolling"]


@st.cache_data(show_spinner=False)
def _load_pitch_shape_artifacts(base_url: str, target_date: date) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    day = target_date.isoformat()
    daily_base = f"{base_url}/daily/{day}"
    loaded = load_remote_parquet_bundle(
        {
            "batter_family_zone_profiles": (daily_base, "daily_batter_family_zone_profiles.parquet", BATTER_FAMILY_REQUIRED_COLUMNS),
            "pitcher_family_zone_context": (daily_base, "daily_pitcher_family_zone_context.parquet", PITCHER_FAMILY_REQUIRED_COLUMNS),
            "pitcher_movement_arsenal": (daily_base, "daily_pitcher_movement_arsenal.parquet", MOVEMENT_REQUIRED_COLUMNS),
        }
    )
    return loaded["batter_family_zone_profiles"], loaded["pitcher_family_zone_context"], loaded["pitcher_movement_arsenal"]


def _sidebar() -> tuple[date, str, str, str, int, int, bool, str]:
    st.sidebar.title("Hosted Filters")
    target_date = st.sidebar.date_input("Slate date", value=date.today())
    split = st.sidebar.selectbox("Split", ["overall", "vs_rhp", "vs_lhp", "home", "away"])
    recent_window = st.sidebar.selectbox("Recent window", ["season", "last_45_days", "last_14_days"])
    weighted_mode = st.sidebar.radio("Weighting", ["weighted", "unweighted"], horizontal=True)
    min_pitch_count = st.sidebar.slider("Min pitches", 0, 3000, 0, 25)
    min_bip = st.sidebar.slider("Min BIP", 0, 500, 0, 5)
    likely_only = st.sidebar.checkbox("Likely starters only", value=False)
    preset_names = list(HITTER_PRESETS.keys())
    hitter_preset = st.sidebar.selectbox("Hitter view", preset_names, index=preset_names.index("All stats"))
    return target_date, split, recent_window, weighted_mode, min_pitch_count, min_bip, likely_only, hitter_preset


def _load_artifacts_with_fallback(
    base_url: str,
    target_date: date,
    lookback_days: int = 7,
) -> tuple[date, tuple[pd.DataFrame, ...]]:
    last_error: Exception | None = None
    for offset in range(lookback_days + 1):
        candidate = target_date - timedelta(days=offset)
        try:
            return candidate, _load_core_artifacts(base_url, candidate)
        except Exception as exc:  # pragma: no cover
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError("No published hosted artifacts were found.")


def _load_slate_with_fallback(
    base_url: str,
    target_date: date,
    lookback_days: int = 7,
) -> tuple[date, tuple[pd.DataFrame, pd.DataFrame]]:
    last_error: Exception | None = None
    for offset in range(lookback_days + 1):
        candidate = target_date - timedelta(days=offset)
        try:
            return candidate, _load_slate_artifacts(base_url, candidate)
        except Exception as exc:  # pragma: no cover
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError("No published hosted slate artifacts were found.")


def _game_selection(slate: list[dict]) -> tuple[str, list[dict], str]:
    return render_sticky_logo_game_nav(
        slate,
        key_prefix="game-selector-hosted",
        sections=GAME_SECTION_OPTIONS,
    )


def _render_slate_status_strip(
    target_date: date,
    resolved_date: date,
    split: str,
    recent_window: str,
    weighted_mode: str,
    hitter_preset: str,
) -> None:
    items = [
        ("Slate", target_date.isoformat()),
        ("Published", resolved_date.isoformat()),
        ("Split", split),
        ("Window", recent_window),
        ("Mode", weighted_mode),
        ("View", hitter_preset),
    ]
    chips = "".join(
        f"<span class='kasper-status-chip'><strong>{escape(label)}:</strong> {escape(str(value))}</span>"
        for label, value in items
    )
    st.markdown(
        f"""
        <style>
        .kasper-status-strip {{
            display:flex;
            flex-wrap:wrap;
            gap:8px;
            margin: 4px 0 14px 0;
        }}
        .kasper-status-chip {{
            display:inline-flex;
            align-items:center;
            border:1px solid rgba(31,41,55,0.14);
            border-radius:8px;
            background:#f8fafc;
            color:#374151;
            font-size:0.82rem;
            gap:4px;
            padding:5px 8px;
            white-space:nowrap;
        }}
        .kasper-status-chip strong {{
            color:#111827;
        }}
        </style>
        <div class="kasper-status-strip">{chips}</div>
        """,
        unsafe_allow_html=True,
    )


def _render_selected_game_mini_header(game: dict) -> None:
    away_team = str(game.get("away_team", "") or "")
    home_team = str(game.get("home_team", "") or "")
    away_pitcher = escape(str(game.get("away_probable_pitcher_name") or "Away starter TBD"))
    home_pitcher = escape(str(game.get("home_probable_pitcher_name") or "Home starter TBD"))
    status = escape(str(game.get("game_status") or "Scheduled"))
    game_pk = escape(str(game.get("game_pk") or ""))
    st.markdown(
        f"""
        <style>
        .selected-game-mini-header {{
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:16px;
            border:1px solid rgba(31,41,55,0.14);
            border-radius:8px;
            background:#f8fafc;
            margin: 8px 0 14px 0;
            padding:12px 14px;
        }}
        .selected-game-logos {{
            display:flex;
            align-items:center;
            gap:10px;
            font-size:1.05rem;
            font-weight:800;
            min-width:120px;
        }}
        .selected-game-meta {{
            display:flex;
            flex-wrap:wrap;
            gap:8px 14px;
            justify-content:flex-end;
            color:#4b5563;
            font-size:0.9rem;
        }}
        .selected-game-meta strong {{
            color:#111827;
        }}
        @media (max-width: 760px) {{
            .selected-game-mini-header {{
                align-items:flex-start;
                flex-direction:column;
            }}
            .selected-game-meta {{
                justify-content:flex-start;
            }}
        }}
        </style>
        <div class="selected-game-mini-header">
            <div class="selected-game-logos">
                {team_logo_img_html(away_team, size=34)}
                <span>@</span>
                {team_logo_img_html(home_team, size=34)}
            </div>
            <div class="selected-game-meta">
                <span><strong>{escape(away_team)} starter:</strong> {away_pitcher}</span>
                <span><strong>{escape(home_team)} starter:</strong> {home_pitcher}</span>
                <span><strong>Status:</strong> {status}</span>
                <span><strong>Game PK:</strong> {game_pk}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _filter_top_board(frame: pd.DataFrame, split: str, recent_window: str, weighted_mode: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    output = frame.copy()
    for column, value in {
        "split_key": split,
        "recent_window": recent_window,
        "weighted_mode": weighted_mode,
    }.items():
        if column in output.columns:
            output = output.loc[output[column].eq(value)].copy()
    return output


def _matchup_display_columns(frame: pd.DataFrame) -> list[str]:
    logo_columns = ["away_logo", "matchup_at", "home_logo"]
    if all(column in frame.columns for column in logo_columns) and frame[["away_logo", "home_logo"]].notna().any().any():
        return logo_columns
    return ["game"] if "game" in frame.columns else []


@st.cache_data(show_spinner=False, ttl=300)
def _load_remote_twitter_export_zip(base_url: str, target_date_value: str) -> bytes:
    export_url = f"{base_url.rstrip('/')}/daily/{target_date_value}/{TWITTER_EXPORT_DIRNAME}/{TWITTER_EXPORT_ZIP}"
    with urlopen(export_url, timeout=30) as response:
        return response.read()


def _render_full_slate_twitter_export_button(base_url: str, resolved_date: date) -> None:
    st.markdown("#### Full Slate Twitter Export")
    try:
        zip_bytes = _load_remote_twitter_export_zip(base_url, resolved_date.isoformat())
    except (HTTPError, URLError, TimeoutError, OSError):
        st.info("Full-slate export artifact is not published for this slate yet. Rebuild and publish artifacts.")
        return
    st.download_button(
        label="Download Full Slate Twitter Export",
        data=zip_bytes,
        file_name=f"kasper_full_slate_game_cards_{resolved_date.isoformat()}.zip",
        mime="application/zip",
        use_container_width=True,
        key=f"full-slate-twitter-export-{resolved_date.isoformat()}",
    )


def _render_top_board_sections(
    top_hitters: pd.DataFrame,
    top_pitchers: pd.DataFrame,
    hitter_preset: str,
    mobile_safe: bool,
    base_url: str,
    resolved_date: date,
) -> None:
    st.header("Top Slate Hitters")
    if top_hitters.empty:
        st.info("No full-slate hitter board artifact is available for this slate.")
    else:
        st.caption("HR Form is display-only: recent EV90 and launch-angle shape versus the hitter's own season baseline.")
        preset_columns = hitter_columns_for_preset(hitter_preset)
        ranked_hitters = top_hitters.sort_values(["matchup_score", "xwoba"], ascending=[False, False], na_position="last")
        display_hitters = add_matchup_logo_columns(ranked_hitters)
        export_options = build_slate_export_options(
            ranked_hitters,
            preset_columns,
            top_pitchers.sort_values(["pitcher_score", "xwoba"], ascending=[False, True], na_position="last") if not top_pitchers.empty else top_pitchers,
        )
        render_slate_export_controls(
            "top-matchups-export-hosted",
            "Top Slate Export",
            export_options,
            None,
        )
        _render_full_slate_twitter_export_button(base_url, resolved_date)
        top_hitter_columns, top_hitter_hidden = _hitter_table_columns(
            display_hitters,
            _matchup_display_columns(display_hitters) + [
                column for column in preset_columns if column in display_hitters.columns and column != "game"
            ],
        )
        _logo_cols = {"away_logo", "matchup_at", "home_logo"}
        top_hitter_columns = [c for c in top_hitter_columns if c not in _logo_cols]
        if "hitter_name" in top_hitter_columns:
            top_hitter_columns = ["hitter_name"] + [c for c in top_hitter_columns if c != "hitter_name"]
        _render_hitter_confidence_legend()
        _render_hosted_grid(
            display_hitters[top_hitter_columns].head(10),
            key="top-slate-hitters-hosted",
            mobile_safe=mobile_safe,
            height=320,
            hidden_columns=top_hitter_hidden,
            color_hitter_confidence=True,
        )

    st.header("Top Slate Pitchers")
    if top_pitchers.empty:
        st.info("No full-slate pitcher board artifact is available for this slate.")
    else:
        ranked_pitchers = top_pitchers.sort_values(["pitcher_score", "xwoba"], ascending=[False, True], na_position="last")
        display_pitchers = add_matchup_logo_columns(ranked_pitchers)
        pitcher_columns = [
            column
            for column in [*_matchup_display_columns(display_pitchers), *TOP_PITCHER_COLUMNS]
            if column in display_pitchers.columns and column != "game"
        ]
        if pitcher_columns:
            _render_hosted_grid(
                display_pitchers[pitcher_columns].head(10),
                key="top-slate-pitchers-hosted",
                mobile_safe=mobile_safe,
                height=320,
                lower_is_better=PITCHER_LOWER_IS_BETTER,
                higher_is_better=PITCHER_HIGHER_IS_BETTER,
            )
        else:
            st.info("No pitcher table columns available for this slate.")


def _render_artifact_health(manifest: pd.DataFrame, mobile_safe: bool) -> None:
    if manifest.empty:
        return
    with st.expander("Artifact Health", expanded=False):
        display_columns = [
            column
            for column in [
                "target_date",
                "built_at_utc",
                "metrics_version",
                "game_count",
                "hitter_snapshot_rows",
                "pitcher_snapshot_rows",
                "live_rows",
                "timing_prepare_assets_seconds",
                "timing_daily_artifacts_seconds",
                "timing_per_game_artifacts_seconds",
            ]
            if column in manifest.columns
        ]
        _render_hosted_grid(
            manifest[display_columns] if display_columns else manifest,
            key="artifact-health-hosted",
            mobile_safe=mobile_safe,
            height=130,
        )


def _filter_hitters(
    hitters: pd.DataFrame,
    rosters: pd.DataFrame,
    hitter_pitcher_exclusions: pd.DataFrame,
    rotowire_lineups: dict[str, dict[str, object]],
    team: str,
    opposing_hand: str | None,
    split: str,
    recent_window: str,
    weighted_mode: str,
    min_pitch_count: int,
    min_bip: int,
    likely_only: bool,
) -> pd.DataFrame:
    split_key = split
    if split == "overall" and opposing_hand == "R":
        split_key = "vs_rhp"
    elif split == "overall" and opposing_hand == "L":
        split_key = "vs_lhp"
    roster_lookup = rosters.loc[rosters["team"] == team, ["player_id"]].dropna().drop_duplicates()
    roster_player_ids = pd.to_numeric(roster_lookup["player_id"], errors="coerce").dropna().astype(int)
    frame = hitters.loc[
        (hitters["split_key"] == split_key)
        & (hitters["recent_window"] == recent_window)
        & (hitters["weighted_mode"] == weighted_mode)
    ].copy()
    if not roster_player_ids.empty and "batter" in frame.columns:
        frame = frame.loc[frame["batter"].isin(roster_player_ids)]
    frame["team"] = team
    if likely_only and "likely_starter_score" in frame:
        frame = frame.loc[frame["likely_starter_score"].fillna(0) > 0]
    frame = filter_excluded_pitchers_from_hitter_pool(apply_roster_names(frame, rosters, team), hitter_pitcher_exclusions)
    return apply_projected_lineup(frame, team, rotowire_lineups)


def _filter_pitcher_frame(
    frame: pd.DataFrame,
    split: str,
    recent_window: str,
    weighted_mode: str,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    filtered = frame.copy()
    if "split_key" in filtered.columns:
        filtered = filtered.loc[filtered["split_key"] == split]
    if "recent_window" in filtered.columns:
        filtered = filtered.loc[filtered["recent_window"] == recent_window]
    if "weighted_mode" in filtered.columns:
        filtered = filtered.loc[filtered["weighted_mode"] == weighted_mode]
    return filtered


def _render_pitcher_tab(
    game_pk: int,
    team_label: str,
    pitcher_summary_by_hand: pd.DataFrame,
    pitcher_arsenal: pd.DataFrame,
    pitcher_by_hand: pd.DataFrame,
    pitcher_count_usage: pd.DataFrame,
    pitcher_row: pd.DataFrame,
    movement_arsenal: pd.DataFrame,
    family_context: pd.DataFrame,
    opposing_hitters: pd.DataFrame,
    batter_family_zone_profiles: pd.DataFrame,
    pitcher_family_zone_context: pd.DataFrame,
    mobile_safe: bool,
) -> list[dict]:
    export_sections: list[dict] = []
    tab_summary, tab_arsenal, tab_count, tab_shape = st.tabs(["Summary", "Arsenal", "Count Usage", "Pitch Shape"])

    with tab_summary:
        summary_table = build_pitcher_summary_table(pitcher_summary_by_hand)
        _render_hosted_grid(
            summary_table,
            key=f"summary-{game_pk}-{team_label}",
            mobile_safe=mobile_safe,
            height=132,
            lower_is_better=PITCHER_LOWER_IS_BETTER,
            higher_is_better=PITCHER_HIGHER_IS_BETTER,
        )
        if "batter_side_key" in pitcher_summary_by_hand.columns:
            for side_key in BATTER_SIDE_LABELS:
                summary_frame = pitcher_summary_by_hand.loc[pitcher_summary_by_hand["batter_side_key"] == side_key]
                if not summary_frame.empty:
                    summary_columns = _present_columns(summary_frame, PITCHER_SUMMARY_COLUMNS)
                    export_sections.append(
                        {
                            "title": f"{team_label} Summary {BATTER_SIDE_LABELS[side_key]}",
                            "frame": summary_frame[summary_columns],
                            "lower_is_better": PITCHER_LOWER_IS_BETTER,
                            "higher_is_better": PITCHER_HIGHER_IS_BETTER,
                        }
                    )

    with tab_arsenal:
        arsenal_tabs = st.tabs([BATTER_SIDE_LABELS[key] for key in BATTER_SIDE_LABELS])
        for side_key, side_tab in zip(BATTER_SIDE_LABELS, arsenal_tabs):
            with side_tab:
                if side_key == "all":
                    side_frame = sort_arsenal_frame(pitcher_arsenal)
                elif "batter_side_key" in pitcher_by_hand.columns:
                    side_frame = sort_arsenal_frame(pitcher_by_hand.loc[pitcher_by_hand["batter_side_key"] == side_key])
                else:
                    side_frame = _empty_like(pitcher_by_hand)
                if side_frame.empty:
                    st.info("No arsenal data available.")
                else:
                    arsenal_columns = _present_columns(side_frame, ARSENAL_COLUMNS)
                    arsenal_grid = _render_hosted_grid(
                        side_frame[arsenal_columns],
                        key=f"arsenal-{game_pk}-{team_label}-{side_key}",
                        mobile_safe=mobile_safe,
                        height=250,
                        lower_is_better={"hard_hit_pct", "xwoba_con"},
                        higher_is_better={"usage_pct", "swstr_pct", "avg_release_speed", "avg_spin_rate"},
                    )
                    export_sections.append(
                        {
                            "title": f"{team_label} Arsenal {BATTER_SIDE_LABELS[side_key]}",
                            "frame": arsenal_grid[_present_columns(arsenal_grid, ARSENAL_COLUMNS)],
                            "lower_is_better": {"hard_hit_pct", "xwoba_con"},
                            "higher_is_better": {"usage_pct", "swstr_pct", "avg_release_speed", "avg_spin_rate"},
                        }
                    )

    with tab_count:
        sub_tabs = st.tabs([BATTER_SIDE_LABELS[key] for key in BATTER_SIDE_LABELS])
        for side_key, side_tab in zip(BATTER_SIDE_LABELS, sub_tabs):
            with side_tab:
                if "batter_side_key" in pitcher_count_usage.columns:
                    side_count = pitcher_count_usage.loc[pitcher_count_usage["batter_side_key"] == side_key]
                else:
                    side_count = _empty_like(pitcher_count_usage)
                if side_key == "all":
                    side_arsenal = pitcher_arsenal[["pitch_name", "usage_pct"]] if not pitcher_arsenal.empty else pd.DataFrame(columns=["pitch_name", "usage_pct"])
                elif "batter_side_key" in pitcher_by_hand.columns:
                    side_arsenal = pitcher_by_hand.loc[pitcher_by_hand["batter_side_key"] == side_key, ["pitch_name", "usage_pct"]]
                else:
                    side_arsenal = pd.DataFrame(columns=["pitch_name", "usage_pct"])
                count_frame = pivot_count_usage(side_count, side_arsenal)
                if count_frame.empty:
                    st.info("No count-state usage data available.")
                else:
                    count_columns = _present_columns(count_frame, COUNT_USAGE_COLUMNS)
                    count_grid = _render_hosted_grid(
                        count_frame[count_columns],
                        key=f"count-{game_pk}-{team_label}-{side_key}",
                        mobile_safe=mobile_safe,
                        height=250,
                        higher_is_better=set(COUNT_BUCKET_ORDER),
                    )
                    export_sections.append(
                        {
                            "title": f"{team_label} Count Usage {BATTER_SIDE_LABELS[side_key]}",
                            "frame": count_grid[_present_columns(count_grid, COUNT_USAGE_COLUMNS)],
                            "higher_is_better": set(COUNT_BUCKET_ORDER),
                        }
                    )

    with tab_shape:
        _render_pitch_shape_context(
            game_pk,
            team_label,
            pitcher_row,
            movement_arsenal,
            family_context,
            opposing_hitters,
            batter_family_zone_profiles,
            pitcher_family_zone_context,
            mobile_safe,
        )

    return export_sections


def _build_pitcher_export_sections(
    team_label: str,
    pitcher_summary_by_hand: pd.DataFrame,
    pitcher_arsenal: pd.DataFrame,
    pitcher_by_hand: pd.DataFrame,
    pitcher_count_usage: pd.DataFrame,
) -> list[dict]:
    export_sections: list[dict] = []
    for side_key, side_label in BATTER_SIDE_LABELS.items():
        if "batter_side_key" in pitcher_summary_by_hand.columns:
            summary_frame = pitcher_summary_by_hand.loc[pitcher_summary_by_hand["batter_side_key"] == side_key]
            if not summary_frame.empty:
                summary_columns = _present_columns(summary_frame, PITCHER_SUMMARY_COLUMNS)
                export_sections.append(
                    {
                        "title": f"{team_label} Summary {side_label}",
                        "frame": summary_frame[summary_columns],
                        "lower_is_better": PITCHER_LOWER_IS_BETTER,
                        "higher_is_better": PITCHER_HIGHER_IS_BETTER,
                    }
                )

        if side_key == "all":
            side_arsenal = sort_arsenal_frame(pitcher_arsenal)
        elif "batter_side_key" in pitcher_by_hand.columns:
            side_arsenal = sort_arsenal_frame(pitcher_by_hand.loc[pitcher_by_hand["batter_side_key"] == side_key])
        else:
            side_arsenal = _empty_like(pitcher_by_hand)
        if not side_arsenal.empty:
            arsenal_columns = _present_columns(side_arsenal, ARSENAL_COLUMNS)
            export_sections.append(
                {
                    "title": f"{team_label} Arsenal {side_label}",
                    "frame": side_arsenal[arsenal_columns],
                    "lower_is_better": {"hard_hit_pct", "xwoba_con"},
                    "higher_is_better": {"usage_pct", "swstr_pct", "avg_release_speed", "avg_spin_rate"},
                }
            )

        if "batter_side_key" in pitcher_count_usage.columns:
            side_count = pitcher_count_usage.loc[pitcher_count_usage["batter_side_key"] == side_key]
        else:
            side_count = _empty_like(pitcher_count_usage)
        if side_key == "all":
            side_usage = pitcher_arsenal[["pitch_name", "usage_pct"]] if not pitcher_arsenal.empty else pd.DataFrame(columns=["pitch_name", "usage_pct"])
        elif "batter_side_key" in pitcher_by_hand.columns:
            side_usage = pitcher_by_hand.loc[pitcher_by_hand["batter_side_key"] == side_key, ["pitch_name", "usage_pct"]]
        else:
            side_usage = pd.DataFrame(columns=["pitch_name", "usage_pct"])
        count_frame = pivot_count_usage(side_count, side_usage)
        if not count_frame.empty:
            count_columns = _present_columns(count_frame, COUNT_USAGE_COLUMNS)
            export_sections.append(
                {
                    "title": f"{team_label} Count Usage {side_label}",
                    "frame": count_frame[count_columns],
                    "higher_is_better": set(COUNT_BUCKET_ORDER),
                }
            )
    return export_sections


def render_matchups_page() -> None:
    st.set_page_config(page_title="Kasper", page_icon=page_icon_path(), layout="wide")
    apply_branding_head()
    render_kasper_header()
    perf_events: list[tuple[str, float]] = []
    base_url = _base_url()
    if not base_url:
        st.error("Set MLB_HOSTED_BASE_URL to your Hugging Face dataset file base URL before running this app.")
        return

    target_date, split, recent_window, weighted_mode, min_pitch_count, min_bip, likely_only, hitter_preset = _sidebar()
    mobile_safe = True
    try:
        load_start = perf_counter()
        resolved_date, slate_artifacts = _load_slate_with_fallback(base_url, target_date)
        slate, rosters = slate_artifacts
        _record_perf(perf_events, "slate load", load_start)
    except Exception as exc:  # pragma: no cover
        st.error(f"Unable to load hosted slate for {target_date.isoformat()}: {exc}")
        return
    if resolved_date != target_date:
        st.warning(
            f"No published artifacts were found for {target_date.isoformat()}. "
            f"Showing the latest available published slate from {resolved_date.isoformat()} instead."
        )
    _render_slate_status_strip(target_date, resolved_date, split, recent_window, weighted_mode, hitter_preset)
    all_games = slate.to_dict(orient="records")
    top_hitters = pd.DataFrame()
    top_pitchers = pd.DataFrame()
    try:
        top_board_start = perf_counter()
        top_hitters, top_pitchers = _load_filtered_top_board_artifacts(base_url, resolved_date, split, recent_window, weighted_mode)
        _render_top_board_sections(top_hitters, top_pitchers, hitter_preset, mobile_safe, base_url, resolved_date)
        _render_artifact_health(_load_artifact_manifest(base_url, resolved_date), mobile_safe)
        _record_perf(perf_events, "top boards", top_board_start)
    except Exception:
        st.warning("Full-slate top board artifacts were not found. Rebuild and publish artifacts to restore full-slate top tables.")

    if "url_params_initialized" not in st.session_state:
        st.session_state["url_params_initialized"] = True
        _game_pk_from_url = st.query_params.get("game")
        _section_from_url = st.query_params.get("section")
        if _game_pk_from_url:
            st.session_state["game-selector-hosted-selected-game-pk"] = _game_pk_from_url
        if _game_pk_from_url and _section_from_url:
            st.session_state[f"section-{_game_pk_from_url}"] = _section_from_url

    _render_hosted_selected_game_area(
        base_url,
        target_date,
        resolved_date,
        split,
        recent_window,
        weighted_mode,
        min_pitch_count,
        min_bip,
        likely_only,
        hitter_preset,
        all_games,
        top_hitters,
        perf_events,
        mobile_safe,
    )


_SKELETON_HTML = (
    "<style>"
    "@keyframes kasper-shimmer{0%{background-position:-400px 0}100%{background-position:400px 0}}"
    ".kasper-skel{background:linear-gradient(90deg,#e8f0f4 25%,#d0dce2 50%,#e8f0f4 75%);"
    "background-size:800px 100%;animation:kasper-shimmer 1.4s infinite linear;"
    "border-radius:8px;height:18px;margin:4px 0;}"
    ".kasper-skel-row{display:flex;gap:8px;margin-bottom:10px;}"
    ".kasper-skel-lg{height:80px;border-radius:12px;}"
    "</style>"
    "<div class='kasper-skel-row'>"
    "<div class='kasper-skel' style='width:35%'></div>"
    "<div class='kasper-skel' style='width:25%'></div>"
    "</div>"
    "<div class='kasper-skel kasper-skel-lg'></div>"
    "<div class='kasper-skel' style='width:60%;margin-top:8px'></div>"
)


@st.fragment
def _render_hosted_selected_game_area(
    base_url: str,
    target_date: date,
    resolved_date: date,
    split: str,
    recent_window: str,
    weighted_mode: str,
    min_pitch_count: int,
    min_bip: int,
    likely_only: bool,
    hitter_preset: str,
    all_games: list[dict],
    top_hitters: pd.DataFrame,
    perf_events: list[tuple[str, float]],
    mobile_safe: bool,
) -> None:
    st.divider()
    selected_label, selected_games, selected_section = _game_selection(all_games)
    if selected_games:
        st.query_params["game"] = str(selected_games[0]["game_pk"])
        st.query_params["section"] = selected_section
    else:
        st.query_params.pop("game", None)
        st.query_params.pop("section", None)
    st.caption(f"{'Slate summary' if selected_label == 'Slate Summary' else 'Selected game'} | {len(all_games)} games on slate")
    if selected_games:
        st_components.html(
            """
            <style>
              *{box-sizing:border-box;margin:0}
              body{background:transparent;font-family:"Segoe UI",system-ui,sans-serif}
              #share-btn{appearance:none;background:#f8fafc;border:1px solid rgba(31,41,55,0.18);
                border-radius:999px;color:#374151;cursor:pointer;font-size:11px;font-weight:600;
                padding:4px 10px;line-height:1.4}
              #share-btn:active{background:#eef3f8}
            </style>
            <button id="share-btn">\U0001f517 Copy Link</button>
            <script>
              document.getElementById("share-btn").addEventListener("click",function(){
                var btn=this,url=window.parent.location.href;
                if(navigator.clipboard){
                  navigator.clipboard.writeText(url).then(function(){
                    btn.textContent="\u2713 Copied!";
                    setTimeout(function(){btn.textContent="\U0001f517 Copy Link"},2000);
                  });
                }else{
                  var ta=document.createElement("textarea");
                  ta.value=url;document.body.appendChild(ta);ta.select();
                  document.execCommand("copy");document.body.removeChild(ta);
                  btn.textContent="\u2713 Copied!";
                  setTimeout(function(){btn.textContent="\U0001f517 Copy Link"},2000);
                }
              });
            </script>
            """,
            height=34,
        )
    if selected_label == "Slate Summary":
        st.header("Slate Summary")
        slate_summary = pd.DataFrame(all_games)
        slate_summary["game"] = slate_summary["away_team"].astype(str) + " @ " + slate_summary["home_team"].astype(str)
        slate_summary = add_matchup_logo_columns(slate_summary)
        matchup_columns = _matchup_display_columns(slate_summary) or ["away_team", "home_team"]
        summary_columns = [
            column
            for column in [
                *matchup_columns,
                "away_probable_pitcher_name",
                "home_probable_pitcher_name",
                "game_status",
                "game_pk",
            ]
            if column in slate_summary.columns
        ]
        _render_hosted_grid(
            slate_summary[summary_columns] if summary_columns else slate_summary,
            key="slate-summary-hosted",
            mobile_safe=mobile_safe,
            height=420,
        )
        summary_matchups = _load_slate_summary_artifact(base_url, resolved_date, split, recent_window, weighted_mode)
        if summary_matchups.empty:
            summary_matchups = build_slate_summary_matchup_overview(top_hitters, per_game=3)
        hidden_summary_columns = {"split_key", "recent_window", "weighted_mode"}
        visible_summary_columns = [column for column in summary_matchups.columns if column not in hidden_summary_columns]
        if not summary_matchups.empty:
            st.markdown("#### Top 3 Matchups By Game")
            _render_hosted_grid(
                summary_matchups[visible_summary_columns],
                key="slate-summary-top-matchups-hosted",
                mobile_safe=mobile_safe,
                height=420,
            )
        else:
            st.info("No top matchup board rows are available for this slate summary.")
        _render_perf(perf_events)
        return
    skel_ph = st.empty()
    skel_ph.markdown(_SKELETON_HTML, unsafe_allow_html=True)
    try:
        load_start = perf_counter()
        resolved_date, artifacts = _load_artifacts_with_fallback(base_url, target_date)
        (
            slate,
            rosters,
            hitters,
            pitchers,
            pitcher_summary_by_hand,
            arsenal,
            arsenal_by_hand,
            usage_by_count,
            batter_zone_profiles,
            pitcher_zone_profiles,
            hitter_pitcher_exclusions,
        ) = artifacts
        _record_perf(perf_events, "core load", load_start)
    except Exception as exc:  # pragma: no cover
        st.error(f"Unable to load hosted artifacts for {target_date.isoformat()}: {exc}")
        return
    active_sections = {selected_section}
    batter_join_col = "batter_id" if "batter_id" in batter_zone_profiles.columns else "batter"
    pitcher_join_col = "pitcher_id" if "pitcher_id" in pitcher_zone_profiles.columns else "pitcher"
    valid_teams = tuple(sorted({game["away_team"] for game in all_games} | {game["home_team"] for game in all_games}))
    try:
        rotowire_start = perf_counter()
        rotowire_lineups = resolve_rotowire_lineups(fetch_rotowire_lineups(resolved_date, valid_teams), rosters)
        _record_perf(perf_events, "rotowire", rotowire_start)
    except Exception:
        rotowire_lineups = {}

    hitter_rolling = pd.DataFrame()
    pitcher_rolling = pd.DataFrame()
    batter_family_zone_profiles = pd.DataFrame()
    pitcher_family_zone_context = pd.DataFrame()
    pitcher_movement_arsenal = pd.DataFrame()
    if "Rolling" in active_sections:
        rolling_start = perf_counter()
        hitter_rolling, pitcher_rolling = _load_rolling_artifacts(base_url, resolved_date)
        _record_perf(perf_events, "rolling load", rolling_start)
    if {"Matchup", "Exports"} & active_sections:
        shape_start = perf_counter()
        (
            batter_family_zone_profiles,
            pitcher_family_zone_context,
            pitcher_movement_arsenal,
        ) = _load_pitch_shape_artifacts(base_url, resolved_date)
        _record_perf(perf_events, "pitch-shape load", shape_start)

    batter_zone_named = pd.DataFrame()
    if {"Pitcher Zones", "Hitter Zones"} & active_sections:
        zone_named_start = perf_counter()
        roster_lookup = rosters[["team", "player_id", "player_name"]].drop_duplicates("player_id")
        batter_zone_named = batter_zone_profiles.merge(roster_lookup, left_on=batter_join_col, right_on="player_id", how="left")
        _record_perf(perf_events, "zone merge", zone_named_start)

    hitters_by_game: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    pitchers_by_game: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    best_matchups_by_game: dict[int, pd.DataFrame] = {}
    pitcher_summary_by_hand_map: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    pitcher_arsenal_map: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    pitcher_by_hand_map: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    pitcher_count_map: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    opponent_hitters_by_key: dict[tuple[object, object, object, object, object], pd.DataFrame] = {}
    filtered_pitchers = _filter_pitcher_frame(pitchers, split, recent_window, weighted_mode)
    filtered_summary = _filter_pitcher_frame(pitcher_summary_by_hand, split, recent_window, weighted_mode)
    filtered_arsenal = _filter_pitcher_frame(arsenal, split, recent_window, weighted_mode)
    filtered_arsenal_by_hand = _filter_pitcher_frame(arsenal_by_hand, split, recent_window, weighted_mode)
    filtered_count = _filter_pitcher_frame(usage_by_count, split, recent_window, weighted_mode)
    filtered_family_context = _filter_pitcher_frame(pitcher_family_zone_context, split, recent_window, weighted_mode)
    filtered_movement_arsenal = _filter_pitcher_frame(pitcher_movement_arsenal, split, recent_window, weighted_mode)
    summary_by_pitcher = _frame_by_pitcher_id(filtered_summary)
    arsenal_by_pitcher = _frame_by_pitcher_id(filtered_arsenal)
    arsenal_by_hand_by_pitcher = _frame_by_pitcher_id(filtered_arsenal_by_hand)
    count_by_pitcher = _frame_by_pitcher_id(filtered_count)
    family_context_by_pitcher = _frame_by_pitcher_id(filtered_family_context)
    movement_arsenal_by_pitcher = _frame_by_pitcher_id(filtered_movement_arsenal)

    game_loop_start = perf_counter()
    for game in selected_games:
        away_pitcher = filtered_pitchers.loc[filtered_pitchers["pitcher_id"] == game.get("away_probable_pitcher_id")].copy()
        home_pitcher = filtered_pitchers.loc[filtered_pitchers["pitcher_id"] == game.get("home_probable_pitcher_id")].copy()
        away_hand = home_pitcher["p_throws"].iloc[0] if not home_pitcher.empty else None
        home_hand = away_pitcher["p_throws"].iloc[0] if not away_pitcher.empty else None

        away_hitters = _filter_hitters(hitters, rosters, hitter_pitcher_exclusions, rotowire_lineups, game["away_team"], away_hand, split, recent_window, weighted_mode, min_pitch_count, min_bip, likely_only)
        home_hitters = _filter_hitters(hitters, rosters, hitter_pitcher_exclusions, rotowire_lineups, game["home_team"], home_hand, split, recent_window, weighted_mode, min_pitch_count, min_bip, likely_only)
        hitters_by_game[game["game_pk"]] = (
            add_hitter_matchup_score(
                away_hitters,
                batter_zone_profiles=batter_zone_profiles,
                pitcher_zone_profiles=pitcher_zone_profiles,
                opposing_pitcher_id=game.get("home_probable_pitcher_id"),
                opposing_pitcher_hand=away_hand,
            ),
            add_hitter_matchup_score(
                home_hitters,
                batter_zone_profiles=batter_zone_profiles,
                pitcher_zone_profiles=pitcher_zone_profiles,
                opposing_pitcher_id=game.get("away_probable_pitcher_id"),
                opposing_pitcher_hand=home_hand,
            ),
        )
        away_scored_hitters, home_scored_hitters = hitters_by_game[game["game_pk"]]
        if game.get("away_probable_pitcher_id"):
            opponent_hitters_by_key[
                build_pitcher_matchup_key(game["game_pk"], game.get("away_probable_pitcher_id"), split, recent_window, weighted_mode)
            ] = home_scored_hitters
        if game.get("home_probable_pitcher_id"):
            opponent_hitters_by_key[
                build_pitcher_matchup_key(game["game_pk"], game.get("home_probable_pitcher_id"), split, recent_window, weighted_mode)
            ] = away_scored_hitters
        best_matchups_by_game[game["game_pk"]] = build_best_matchups(*hitters_by_game[game["game_pk"]])
        pitcher_summary_by_hand_map[game["game_pk"]] = (
            summary_by_pitcher.get(game.get("away_probable_pitcher_id"), filtered_summary.head(0)).copy(),
            summary_by_pitcher.get(game.get("home_probable_pitcher_id"), filtered_summary.head(0)).copy(),
        )
        pitcher_arsenal_map[game["game_pk"]] = (
            arsenal_by_pitcher.get(game.get("away_probable_pitcher_id"), filtered_arsenal.head(0)).copy(),
            arsenal_by_pitcher.get(game.get("home_probable_pitcher_id"), filtered_arsenal.head(0)).copy(),
        )
        pitcher_by_hand_map[game["game_pk"]] = (
            arsenal_by_hand_by_pitcher.get(game.get("away_probable_pitcher_id"), filtered_arsenal_by_hand.head(0)).copy(),
            arsenal_by_hand_by_pitcher.get(game.get("home_probable_pitcher_id"), filtered_arsenal_by_hand.head(0)).copy(),
        )
        pitcher_count_map[game["game_pk"]] = (
            count_by_pitcher.get(game.get("away_probable_pitcher_id"), filtered_count.head(0)).copy(),
            count_by_pitcher.get(game.get("home_probable_pitcher_id"), filtered_count.head(0)).copy(),
        )
    slate_pitcher_ids = {
        pitcher_id
        for game in selected_games
        for pitcher_id in [game.get("away_probable_pitcher_id"), game.get("home_probable_pitcher_id")]
        if pitcher_id
    }
    scored_slate_pitchers = (
        add_pitcher_rank_score(
            filtered_pitchers.loc[filtered_pitchers["pitcher_id"].isin(slate_pitcher_ids)].copy(),
            opponent_hitters_by_key=opponent_hitters_by_key or None,
            batter_family_zone_profiles=batter_family_zone_profiles,
            pitcher_family_zone_context=pitcher_family_zone_context,
        )
        if slate_pitcher_ids
        else filtered_pitchers.head(0).copy()
    )
    scored_pitchers_by_id = _frame_by_pitcher_id(scored_slate_pitchers)
    for game in selected_games:
        pitchers_by_game[game["game_pk"]] = (
            scored_pitchers_by_id.get(game.get("away_probable_pitcher_id"), filtered_pitchers.head(0)).copy(),
            scored_pitchers_by_id.get(game.get("home_probable_pitcher_id"), filtered_pitchers.head(0)).copy(),
        )
    _record_perf(perf_events, "game prep", game_loop_start)
    skel_ph.empty()

    _render_perf(perf_events)
    hitter_columns = hitter_columns_for_preset(hitter_preset)

    for idx, game in enumerate(selected_games):
        away_hitters, home_hitters = hitters_by_game.get(game["game_pk"], (pd.DataFrame(), pd.DataFrame()))
        away_pitcher, home_pitcher = pitchers_by_game.get(game["game_pk"], (pd.DataFrame(), pd.DataFrame()))
        away_pitcher_id = game.get("away_probable_pitcher_id")
        home_pitcher_id = game.get("home_probable_pitcher_id")
        away_summary_by_hand = summary_by_pitcher.get(away_pitcher_id, filtered_summary.head(0)).copy()
        home_summary_by_hand = summary_by_pitcher.get(home_pitcher_id, filtered_summary.head(0)).copy()
        away_arsenal = arsenal_by_pitcher.get(away_pitcher_id, filtered_arsenal.head(0)).copy()
        home_arsenal = arsenal_by_pitcher.get(home_pitcher_id, filtered_arsenal.head(0)).copy()
        away_by_hand = arsenal_by_hand_by_pitcher.get(away_pitcher_id, filtered_arsenal_by_hand.head(0)).copy()
        home_by_hand = arsenal_by_hand_by_pitcher.get(home_pitcher_id, filtered_arsenal_by_hand.head(0)).copy()
        away_count = count_by_pitcher.get(away_pitcher_id, filtered_count.head(0)).copy()
        home_count = count_by_pitcher.get(home_pitcher_id, filtered_count.head(0)).copy()
        away_family_context = family_context_by_pitcher.get(away_pitcher_id, filtered_family_context.head(0)).copy()
        home_family_context = family_context_by_pitcher.get(home_pitcher_id, filtered_family_context.head(0)).copy()
        away_movement = movement_arsenal_by_pitcher.get(away_pitcher_id, filtered_movement_arsenal.head(0)).copy()
        home_movement = movement_arsenal_by_pitcher.get(home_pitcher_id, filtered_movement_arsenal.head(0)).copy()
        best_matchups = best_matchups_by_game.get(game["game_pk"], pd.DataFrame()).copy()

        with st.container():
            _render_selected_game_mini_header(game)
            active_section = selected_section
            if active_section == "Matchup":
                st.markdown("#### Best Matchups")
                matchup_columns = _present_columns(best_matchups, BEST_MATCHUP_COLUMNS)
                if matchup_columns:
                    matchup_columns, matchup_hidden = _hitter_table_columns(best_matchups, matchup_columns)
                    _render_hitter_confidence_legend()
                    best_matchups = _render_hosted_grid(
                        best_matchups[matchup_columns],
                        key=f"best-hosted-{game['game_pk']}",
                        mobile_safe=mobile_safe,
                        height=170,
                        hidden_columns=matchup_hidden,
                        color_hitter_confidence=True,
                    )
                else:
                    st.info("No matchup rows available for this game.")

                with st.expander("Pitchers", expanded=True):
                    pitcher_cols = st.columns(2)
                    with pitcher_cols[0]:
                        st.markdown(f"##### {team_logo_img_html(game['away_team'], size=24)} Starter", unsafe_allow_html=True)
                        away_export_sections = _render_pitcher_tab(
                            game["game_pk"],
                            game["away_team"],
                            away_summary_by_hand,
                            away_arsenal,
                            away_by_hand,
                            away_count,
                            away_pitcher,
                            away_movement,
                            away_family_context,
                            home_hitters,
                            batter_family_zone_profiles,
                            filtered_family_context,
                            mobile_safe,
                        )
                    with pitcher_cols[1]:
                        st.markdown(f"##### {team_logo_img_html(game['home_team'], size=24)} Starter", unsafe_allow_html=True)
                        home_export_sections = _render_pitcher_tab(
                            game["game_pk"],
                            game["home_team"],
                            home_summary_by_hand,
                            home_arsenal,
                            home_by_hand,
                            home_count,
                            home_pitcher,
                            home_movement,
                            home_family_context,
                            away_hitters,
                            batter_family_zone_profiles,
                            filtered_family_context,
                            mobile_safe,
                        )

                st.markdown("#### Hitters")
                st.markdown(f"{team_logo_img_html(game['away_team'], size=22)} vs {game.get('home_probable_pitcher_name') or 'opposing starter'}", unsafe_allow_html=True)
                away_hitter_columns, away_hitter_hidden = _hitter_table_columns(away_hitters, hitter_columns)
                away_hitters = _render_hosted_grid(
                    away_hitters[away_hitter_columns],
                    key=f"away-hitters-hosted-{game['game_pk']}",
                    mobile_safe=mobile_safe,
                    height=360,
                    hidden_columns=away_hitter_hidden,
                    color_hitter_confidence=True,
                )
                st.markdown(f"{team_logo_img_html(game['home_team'], size=22)} vs {game.get('away_probable_pitcher_name') or 'opposing starter'}", unsafe_allow_html=True)
                home_hitter_columns, home_hitter_hidden = _hitter_table_columns(home_hitters, hitter_columns)
                home_hitters = _render_hosted_grid(
                    home_hitters[home_hitter_columns],
                    key=f"home-hitters-hosted-{game['game_pk']}",
                    mobile_safe=mobile_safe,
                    height=360,
                    hidden_columns=home_hitter_hidden,
                    color_hitter_confidence=True,
                )

            elif active_section == "Rolling":
                roll_tabs = st.tabs(["Rolling 5", "Rolling 10", "Rolling 15"])
                away_hitter_names = set(away_hitters.get("hitter_name", pd.Series(dtype="object")).dropna().tolist())
                home_hitter_names = set(home_hitters.get("hitter_name", pd.Series(dtype="object")).dropna().tolist())
                away_pitcher_name = away_pitcher["pitcher_name"].iloc[0] if not away_pitcher.empty else None
                home_pitcher_name = home_pitcher["pitcher_name"].iloc[0] if not home_pitcher.empty else None
                for label, tab in zip(["Rolling 5", "Rolling 10", "Rolling 15"], roll_tabs):
                    with tab:
                        cols = st.columns(2)
                        with cols[0]:
                            hitter_frame = hitter_rolling.loc[
                                hitter_rolling["rolling_window"].eq(label)
                                & hitter_rolling["player_name"].isin(sorted(away_hitter_names | home_hitter_names))
                            ]
                            _render_hosted_grid(
                                hitter_frame[[column for column in HITTER_ROLLING_COLUMNS if column in hitter_frame.columns]],
                                key=f"h-roll-hosted-{game['game_pk']}-{label}",
                                mobile_safe=mobile_safe,
                                height=260,
                            )
                        with cols[1]:
                            pitcher_frame = pitcher_rolling.loc[
                                pitcher_rolling["rolling_window"].eq(label)
                                & pitcher_rolling["player_name"].isin([name for name in [away_pitcher_name, home_pitcher_name] if name])
                            ]
                            _render_hosted_grid(
                                pitcher_frame[[column for column in PITCHER_ROLLING_COLUMNS if column in pitcher_frame.columns]],
                                key=f"p-roll-hosted-{game['game_pk']}-{label}",
                                mobile_safe=mobile_safe,
                                height=260,
                                lower_is_better=PITCHER_LOWER_IS_BETTER | {"barrel_bip_pct"},
                                higher_is_better=PITCHER_HIGHER_IS_BETTER | {"avg_release_speed"},
                            )

            elif active_section == "Pitcher Zones":
                zone_tabs = st.tabs([game["away_team"], game["home_team"]])
                for pitcher_row, tab, team_label in [
                    (away_pitcher, zone_tabs[0], game["away_team"]),
                    (home_pitcher, zone_tabs[1], game["home_team"]),
                ]:
                    with tab:
                        if pitcher_row.empty:
                            st.info("No pitcher zone data available.")
                        else:
                            pitcher_id = pitcher_row["pitcher_id"].iloc[0]
                            zone_frame = (
                                pitcher_zone_profiles.loc[pitcher_zone_profiles[pitcher_join_col] == pitcher_id]
                                .sort_values("sample_size", ascending=False, na_position="last")
                                .head(200)
                            )
                            opposing_hitters = home_hitters if team_label == game["away_team"] else away_hitters
                            opponent_names = sorted(opposing_hitters.get("hitter_name", pd.Series(dtype="object")).dropna().unique().tolist())
                            selected_hitter = (
                                st.selectbox("Overlay hitter", opponent_names, key=f"p-zone-hosted-hitter-{game['game_pk']}-{team_label}")
                                if opponent_names
                                else None
                            )
                            hitter_detail = batter_zone_named.loc[batter_zone_named["player_name"] == selected_hitter].copy() if selected_hitter else pd.DataFrame()
                            pitch_types = ["All pitches"] + sorted(
                                set(zone_frame.get("pitch_type", pd.Series(dtype="object")).dropna().tolist())
                                | set(hitter_detail.get("pitch_type", pd.Series(dtype="object")).dropna().tolist())
                            )
                            selected_pitch = st.selectbox("Pitch type", pitch_types, key=f"p-zone-hosted-pitch-{game['game_pk']}-{team_label}")
                            pitcher_map = aggregate_pitcher_zone_map(zone_frame, selected_pitch)
                            hitter_map = aggregate_batter_zone_map(hitter_detail, selected_pitch)
                            render_zone_tool(
                                title=f"{pitcher_row['pitcher_name'].iloc[0]} Usage",
                                subtitle=f"{selected_pitch} | Pitcher zone attack",
                                zone_map=pitcher_map,
                                key=f"p-zone-react-{game['game_pk']}-{team_label}",
                                map_kind="pitcher",
                                overlay_zone_map=hitter_map,
                            )
                            _render_hosted_grid(
                                zone_frame[[column for column in PITCHER_ZONE_COLUMNS if column in zone_frame.columns]],
                                key=f"p-zone-hosted-{game['game_pk']}-{team_label}",
                                mobile_safe=mobile_safe,
                                height=240,
                                higher_is_better={"usage_rate"},
                            )

            elif active_section == "Hitter Zones":
                zone_tabs = st.tabs([game["away_team"], game["home_team"]])
                for team_label, tab in [(game["away_team"], zone_tabs[0]), (game["home_team"], zone_tabs[1])]:
                    with tab:
                        zone_frame = (
                            batter_zone_named.loc[batter_zone_named["team"] == team_label]
                            .sort_values("sample_size", ascending=False, na_position="last")
                            .head(250)
                        )
                        hitter_options = sorted(zone_frame.get("player_name", pd.Series(dtype="object")).dropna().unique().tolist())
                        selected_hitter = (
                            st.selectbox("Hitter", hitter_options, key=f"h-zone-hosted-hitter-{game['game_pk']}-{team_label}")
                            if hitter_options
                            else None
                        )
                        hitter_detail = zone_frame.loc[zone_frame["player_name"] == selected_hitter].copy() if selected_hitter else pd.DataFrame()
                        hitter_detail["damage_rate"] = (
                            pd.to_numeric(hitter_detail.get("hit_rate"), errors="coerce").fillna(0) * 0.6
                            + pd.to_numeric(hitter_detail.get("hr_rate"), errors="coerce").fillna(0) * 0.4
                        )
                        opposing_pitcher = home_pitcher if team_label == game["away_team"] else away_pitcher
                        pitcher_detail = (
                            pitcher_zone_profiles.loc[pitcher_zone_profiles[pitcher_join_col] == opposing_pitcher["pitcher_id"].iloc[0]].copy()
                            if not opposing_pitcher.empty
                            else pd.DataFrame()
                        )
                        pitch_types = ["All pitches"] + sorted(
                            set(hitter_detail.get("pitch_type", pd.Series(dtype="object")).dropna().tolist())
                            | set(pitcher_detail.get("pitch_type", pd.Series(dtype="object")).dropna().tolist())
                        )
                        selected_pitch = st.selectbox("Pitch type", pitch_types, key=f"h-zone-hosted-pitch-{game['game_pk']}-{team_label}")
                        hitter_map = aggregate_batter_zone_map(hitter_detail, selected_pitch)
                        pitcher_map = aggregate_pitcher_zone_map(pitcher_detail, selected_pitch)
                        opposing_name = opposing_pitcher["pitcher_name"].iloc[0] if not opposing_pitcher.empty else "Opposing Pitcher"
                        render_zone_tool(
                            title=f"{selected_hitter or team_label} Damage",
                            subtitle=f"{selected_pitch} | Hitter zone quality",
                            zone_map=hitter_map,
                            key=f"h-zone-react-{game['game_pk']}-{team_label}",
                            map_kind="hitter",
                            overlay_zone_map=pitcher_map,
                        )
                        _render_hosted_grid(
                            hitter_detail[[column for column in BATTER_ZONE_COLUMNS if column in hitter_detail.columns]],
                            key=f"h-zone-hosted-{game['game_pk']}-{team_label}",
                            mobile_safe=mobile_safe,
                            height=240,
                            higher_is_better={"hit_rate", "hr_rate", "damage_rate"},
                        )

            elif active_section == "Exports":
                away_export_sections = _build_pitcher_export_sections(
                    game["away_team"],
                    away_summary_by_hand,
                    away_arsenal,
                    away_by_hand,
                    away_count,
                )
                home_export_sections = _build_pitcher_export_sections(
                    game["home_team"],
                    home_summary_by_hand,
                    home_arsenal,
                    home_by_hand,
                    home_count,
                )
                export_options = build_game_export_options(
                    game_title=f"{game['away_team']} @ {game['home_team']}",
                    away_team=game["away_team"],
                    home_team=game["home_team"],
                    best_matchups=best_matchups,
                    away_sections=away_export_sections,
                    home_sections=home_export_sections,
                    away_hitters=away_hitters,
                    home_hitters=home_hitters,
                )
                render_selected_game_twitter_export(
                    key=f"twitter-card-hosted-{game['game_pk']}",
                    game=game,
                    away_hitters=away_hitters,
                    home_hitters=home_hitters,
                )
                render_export_hub(
                    key=f"export-hosted-{game['game_pk']}",
                    title=f"{game['away_team']} @ {game['home_team']}",
                    export_options=export_options,
                )
        st.divider()


def main() -> None:
    pages = {
        "All User Pages": [
            st.Page(render_matchups_page, title="Kasper Matchups", url_path="kasper-matchups", default=True),
            st.Page("pages/1_Exit_Velo_Log.py", title="Exit Velo Log", url_path="exit-velo-log"),
            st.Page("pages/1_Weather.py", title="Weather", url_path="weather"),
        ],
        "Admin Pages": [
            st.Page("pages/2_Props_Board.py", title="Props Board", url_path="props-board"),
            st.Page("pages/3_Player_Analysis.py", title="Player Analysis", url_path="player-analysis"),
            st.Page("pages/4_Backtesting.py", title="Backtesting", url_path="backtesting"),
            st.Page("pages/5_Strikeouts.py", title="Strikeouts", url_path="strikeouts"),
            st.Page("pages/6_Bio_Mechanics.py", title="Bio Mechanics", url_path="bio-mechanics"),
            st.Page("pages/7_Stolen_Bases.py", title="Stolen Bases", url_path="stolen-bases"),
        ],
    }
    st.navigation(pages).run()


if __name__ == "__main__":
    main()
