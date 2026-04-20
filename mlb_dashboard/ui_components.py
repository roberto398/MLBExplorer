from __future__ import annotations

from io import BytesIO
from math import cos, radians, sin
from pathlib import Path
import warnings
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import streamlit as st
from .components import render_game_selector, render_sticky_game_nav
from .components import render_zone_tool as render_zone_tool_component
from .team_logos import matchup_logo_html, team_logo_data_uri
from .twitter_exports import HAS_PILLOW as HAS_TWITTER_EXPORT_PILLOW
from .twitter_exports import build_twitter_game_card

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PILLOW = True
except ImportError:  # pragma: no cover
    HAS_PILLOW = False

try:
    from st_aggrid import AgGrid, DataReturnMode, GridOptionsBuilder, GridUpdateMode, JsCode
    HAS_AGGRID = True
except ImportError:  # pragma: no cover
    HAS_AGGRID = False

PERCENT_COLUMNS = {
    "swstr_pct",
    "called_strike_pct",
    "csw_pct",
    "putaway_pct",
    "pulled_barrel_pct",
    "barrel_bbe_pct",
    "barrel_bip_pct",
    "sweet_spot_pct",
    "fb_pct",
    "ball_pct",
    "hard_hit_pct",
    "usage_pct",
    "usage_rate",
    "usage_rate_within_family",
    "usage_rate_overall",
    "prior_weight_share",
    "gb_pct",
    "All counts",
    "Early count",
    "Even count",
    "Pitcher ahead",
    "Pitcher behind",
    "Two-strike",
    "Pre two-strike",
    "Full count",
    # Strikeout page display columns
    "K Rate",
    "BB Rate",
    "Lineup K%",
    "Blend K%",
    "Mix Whiff",
    "SwStr%",
    "K Prob",
}
RATE_COLUMNS = {
    "xwoba",
    "xwoba_con",
    "iso",
    "avg_launch_angle",
    "avg_release_speed",
    "avg_spin_rate",
    "avg_velocity",
    "avg_extension",
    "avg_pfx_x",
    "avg_pfx_z",
    "avg_release_pos_x",
    "avg_release_pos_z",
    "avg_spin_axis",
    "gb_fb_ratio",
    "matchup_score",
    "test_score",
    "ceiling_score",
    "khr_score",
    "pitcher_score",
    "strikeout_score",
    "raw_pitcher_score",
    "raw_strikeout_score",
    "pitcher_matchup_adjustment",
    "strikeout_matchup_adjustment",
    "opponent_lineup_quality",
    "opponent_contact_threat",
    "opponent_whiff_tendency",
    "opponent_family_fit_allowed",
    "zone_fit_score",
    "siera",
    "fit_score",
    # Strikeout page display columns
    "Proj K",
    "Proj BB",
    "Proj BF",
    "Avg Pitches",
    "W. Starts",
    "P/PA",
    "Whiff Idx",
    "Mix Fit",
}
LOWER_IS_BETTER = {"swstr_pct", "siera", "opponent_lineup_quality", "opponent_contact_threat", "opponent_family_fit_allowed"}
HIGHER_IS_BETTER = {
    "called_strike_pct",
    "csw_pct",
    "putaway_pct",
    "pulled_barrel_pct",
    "barrel_bbe_pct",
    "barrel_bip_pct",
    "sweet_spot_pct",
    "fb_pct",
    "ball_pct",
    "hard_hit_pct",
    "usage_pct",
    "xwoba",
    "xwoba_con",
    "iso",
    "avg_release_speed",
    "avg_spin_rate",
    "matchup_score",
    "test_score",
    "ceiling_score",
    "khr_score",
    "zone_fit_score",
    "strikeout_score",
    "raw_pitcher_score",
    "raw_strikeout_score",
    "pitcher_matchup_adjustment",
    "strikeout_matchup_adjustment",
    "opponent_whiff_tendency",
    "gb_pct",
    "gb_fb_ratio",
    "sb_score",
    "fair_probability",
    "edge",
    "runner_attempt_rate",
    "runner_success_rate",
    "pitcher_allowed_attempt_rate",
    "catcher_allowed_attempt_rate",
    "team_allowed_attempt_rate",
}
TARGET_COLUMNS = {"avg_launch_angle": (20.0, 27.5, 35.0)}
HITTER_CONFIDENCE_COLOR_COLUMN = "__hitter_confidence_color"
HITTER_CONFIDENCE_LABEL_COLUMN = "__hitter_confidence_label"
HITTER_CONFIDENCE_HELPER_COLUMNS = {HITTER_CONFIDENCE_COLOR_COLUMN, HITTER_CONFIDENCE_LABEL_COLUMN}
HR_FORM_PCT_COLUMN = "hr_form_pct"
HR_FORM_HELPER_COLUMNS = {HR_FORM_PCT_COLUMN}
HITTER_CONFIDENCE_COLORS = {
    "High": "#166534",
    "Medium": "#1f2937",
    "Thin": "#b45309",
    "Very Thin": "#b91c1c",
}
LOGO_COLUMNS = {"away_logo", "home_logo", "team_logo", "Opp"}
SLATE_SUMMARY_SELECTION = "__slate_summary__"
DISPLAY_LABELS = {
    "away_logo": "Away",
    "matchup_at": "@",
    "home_logo": "Home",
    "team_logo": "Team",
    "game": "Game",
    "game_label": "Game",
    "split_label": "Split",
    "hitter_name": "Hitter",
    "pitcher_name": "Pitcher",
    "team": "Team",
    "pitch_count": "Pitches",
    "bip": "BIP",
    "iso": "ISO",
    "xwoba": "xwOBA",
    "xwoba_con": "xwOBAcon",
    "swstr_pct": "SwStr%",
    "called_strike_pct": "CalledStrike%",
    "csw_pct": "CSW%",
    "putaway_pct": "PutAway%",
    "pulled_barrel_pct": "PulledBrl%",
    "barrel_bbe_pct": "Brl/BBE%",
    "barrel_bip_pct": "Brl/BIP%",
    "sweet_spot_pct": "SweetSpot%",
    "fb_pct": "FB%",
    "ball_pct": "Ball%",
    "siera": "SIERA",
    "gb_pct": "GB%",
    "gb_fb_ratio": "GB/FB",
    "hard_hit_pct": "HH%",
    "avg_launch_angle": "LA",
    "avg_release_speed": "Velo",
    "avg_spin_rate": "Spin",
    "avg_velocity": "Velo",
    "avg_extension": "Ext",
    "avg_pfx_x": "HB",
    "avg_pfx_z": "IVB",
    "avg_release_pos_x": "Rel X",
    "avg_release_pos_z": "Rel Z",
    "avg_spin_axis": "Axis",
    "usage_pct": "Usage%",
    "weighted_sample_size": "Weighted Sample",
    "sample_2026": "2026 Sample",
    "sample_prior": "Prior Sample",
    "weight_2026": "2026 Weight",
    "weight_prior": "Prior Weight",
    "prior_weight_share": "Prior Weight%",
    "pitch_name": "Pitch",
    "p_throws": "Throws",
    "likely_starter_score": "Likely",
    "matchup_score": "Matchup",
    "test_score": "Test Score",
    "ceiling_score": "Ceiling",
    "zone_fit_score": "Zone Fit",
    "hr_form": "HR Form",
    "khr_score": "kHR",
    "hr_form_pct": "HR Form%",
    "pitcher_score": "Pitch Score",
    "strikeout_score": "Strikeout Score",
    "raw_pitcher_score": "Raw Pitch Score",
    "raw_strikeout_score": "Raw Strikeout Score",
    "pitcher_matchup_adjustment": "Pitch Adj",
    "strikeout_matchup_adjustment": "K Adj",
    "opponent_lineup_quality": "Opp Quality",
    "opponent_contact_threat": "Opp Contact",
    "opponent_whiff_tendency": "Opp Whiff",
    "opponent_family_fit_allowed": "Opp Fit",
    "lineup_source": "Lineup Source",
    "lineup_hitter_count": "Opp Hitters",
    "player": "Player",
    "sb_score": "SB Score",
    "fair_probability": "Fair Prob",
    "odds_display": "Odds",
    "edge": "Edge",
    "runner_attempt_rate": "Runner Att%",
    "runner_success_rate": "Runner SB%",
    "pitcher_allowed_attempt_rate": "Pitcher Allowed%",
    "catcher_allowed_attempt_rate": "Catcher Allowed%",
    "team_allowed_attempt_rate": "Team Allowed%",
    "best_books": "Book",
    "player_name": "Player",
    "rolling_window": "Window",
    "games_in_window": "Games",
    "sample_size": "Sample",
    "pitch_type": "Pitch Type",
    "zone": "Zone",
    "pitch_family": "Family",
    "zone_bucket": "Zone",
    "whiff_rate": "Whiff%",
    "ball_in_play_rate": "BIP%",
    "hit_rate": "Hit%",
    "hr_rate": "HR%",
    "damage_rate": "Damage%",
    "usage_rate_overall": "Usage%",
    "hit_allowed_rate": "Hit Allowed%",
    "hr_allowed_rate": "HR Allowed%",
    "damage_allowed_rate": "Damage Allowed%",
    "xwoba_allowed": "xwOBA Allowed",
    "fit_score": "Fit",
    "batter_id": "Batter",
    "pitcher_id": "Pitcher ID",
    "usage_rate": "Usage%",
}
INTEGER_COLUMNS = {"pitch_count", "bip", "likely_starter_score", "sample_size", "sample_2026", "sample_prior", "lineup_hitter_count"}

SHORT_COLUMNS = {"team", "p_throws", "bip", "pitch_count", "likely_starter_score", "lineup_hitter_count"}
MEDIUM_COLUMNS = {
    "split_label",
    "xwoba",
    "xwoba_con",
    "swstr_pct",
    "called_strike_pct",
    "csw_pct",
    "putaway_pct",
    "pulled_barrel_pct",
    "barrel_bbe_pct",
    "barrel_bip_pct",
    "sweet_spot_pct",
    "fb_pct",
    "ball_pct",
    "gb_pct",
    "hard_hit_pct",
    "avg_launch_angle",
    "avg_release_speed",
    "avg_spin_rate",
    "avg_velocity",
    "avg_extension",
    "avg_pfx_x",
    "avg_pfx_z",
    "usage_pct",
    "usage_rate",
    "usage_rate_within_family",
    "matchup_score",
    "test_score",
    "ceiling_score",
    "pitcher_score",
    "strikeout_score",
    "raw_pitcher_score",
    "raw_strikeout_score",
    "pitcher_matchup_adjustment",
    "strikeout_matchup_adjustment",
    "opponent_lineup_quality",
    "opponent_contact_threat",
    "opponent_whiff_tendency",
    "opponent_family_fit_allowed",
    "weighted_sample_size",
    "prior_weight_share",
    "whiff_rate",
    "ball_in_play_rate",
    "hit_rate",
    "hr_rate",
    "damage_rate",
    "usage_rate_overall",
    "hit_allowed_rate",
    "hr_allowed_rate",
    "damage_allowed_rate",
    "xwoba_allowed",
}
LONG_COLUMNS = {"hitter_name", "pitcher_name", "game", "pitch_name", "pitch_family", "lineup_source"}

PITCHER_SUMMARY_TABLE_COLUMNS = [
    "split_label",
    "p_throws",
    "pitch_count",
    "bip",
    "xwoba",
    "strikeout_score",
    "called_strike_pct",
    "csw_pct",
    "swstr_pct",
    "putaway_pct",
    "pulled_barrel_pct",
    "barrel_bip_pct",
    "ball_pct",
    "siera",
    "fb_pct",
    "hard_hit_pct",
    "avg_launch_angle",
]

ZONE_RECTANGLES = {
    11: (1, 0, 3, 1),
    12: (1, 4, 3, 1),
    13: (0, 1, 1, 3),
    14: (4, 1, 1, 3),
    1: (1, 1, 1, 1),
    2: (2, 1, 1, 1),
    3: (3, 1, 1, 1),
    4: (1, 2, 1, 1),
    5: (2, 2, 1, 1),
    6: (3, 2, 1, 1),
    7: (1, 3, 1, 1),
    8: (2, 3, 1, 1),
    9: (3, 3, 1, 1),
}


def _format_value(column: str, value: object, export_mode: bool = False) -> str:
    if column in LOGO_COLUMNS:
        return value if isinstance(value, str) else ""
    if pd.isna(value):
        return "-"
    if column in INTEGER_COLUMNS:
        return f"{int(float(value)):,}"
    if column in PERCENT_COLUMNS:
        decimals = 3 if export_mode else 1
        return f"{float(value):.{decimals}%}"
    if column == "avg_spin_rate":
        decimals = 0 if not export_mode else 3
        return f"{float(value):.{decimals}f}"
    if column in RATE_COLUMNS:
        decimals = 3 if export_mode else (3 if "xwoba" in column or column in {"matchup_score", "test_score", "ceiling_score", "zone_fit_score", "khr_score", "iso"} else 1)
        return f"{float(value):.{decimals}f}"
    if isinstance(value, float):
        decimals = 3 if export_mode else 2
        return f"{value:.{decimals}f}"
    return str(value)


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[idx : idx + 2], 16) for idx in (0, 2, 4))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#" + "".join(f"{max(0, min(255, channel)):02x}" for channel in rgb)


def _blend_color(start: str, end: str, ratio: float) -> str:
    start_rgb = _hex_to_rgb(start)
    end_rgb = _hex_to_rgb(end)
    clamped = max(0.0, min(1.0, ratio))
    blended = tuple(round(start_rgb[i] + (end_rgb[i] - start_rgb[i]) * clamped) for i in range(3))
    return _rgb_to_hex(blended)


def _ease_ratio(ratio: float) -> float:
    clamped = max(0.0, min(1.0, ratio))
    centered = (clamped - 0.5) * 2.0
    curved = 0.5 + 0.5 * (abs(centered) ** 0.8) * (1 if centered >= 0 else -1)
    return max(0.0, min(1.0, curved))


def _diverging_heatmap_hex(ratio: float) -> str:
    bad = "#c94b4b"
    mid = "#f2dc62"
    good = "#2f8f4e"
    curved = _ease_ratio(ratio)
    if curved <= 0.5:
        inner_ratio = curved / 0.5
        return _blend_color(bad, mid, inner_ratio)
    inner_ratio = (curved - 0.5) / 0.5
    return _blend_color(mid, good, inner_ratio)


def _zone_heatmap_hex(ratio: float) -> str:
    cold = "#58a7e4"
    mid = "#a8b0ad"
    hot = "#ef8d32"
    curved = _ease_ratio(ratio)
    if curved <= 0.5:
        return _blend_color(cold, mid, curved / 0.5)
    return _blend_color(mid, hot, (curved - 0.5) / 0.5)


def _hr_form_heatmap_hex(pct: object) -> str | None:
    numeric = pd.to_numeric(pd.Series([pct]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    value = max(0.05, min(0.95, float(numeric)))
    deep_cold = "#c94b4b"
    cold = "#df8177"
    neutral = "#a8b0ad"
    warm = "#8ec39e"
    hot = "#2f8f4e"
    if value < 0.40:
        return _blend_color(deep_cold, cold, max(0.0, min(1.0, (value - 0.05) / 0.35)))
    if value <= 0.60:
        return neutral
    return _blend_color(warm, hot, max(0.0, min(1.0, (value - 0.60) / 0.35)))


def _two_sided_target_ratio(value: float, low: float, ideal: float, high: float) -> float:
    if value <= ideal:
        return 1.0 - min(1.0, max(0.0, (ideal - value) / max(ideal - low, 1e-9)))
    return 1.0 - min(1.0, max(0.0, (value - ideal) / max(high - ideal, 1e-9)))


def _background_hex(
    column: str,
    value: object,
    series: pd.Series,
    lower_is_better: set[str] | None = None,
    higher_is_better: set[str] | None = None,
) -> str | None:
    if pd.isna(value):
        return None
    numeric_series = pd.to_numeric(series, errors="coerce").dropna()
    if numeric_series.empty:
        return None

    numeric_value = float(value)
    if column in TARGET_COLUMNS:
        low, ideal, high = TARGET_COLUMNS[column]
        ratio = _two_sided_target_ratio(numeric_value, low, ideal, high)
    else:
        col_min = float(numeric_series.min())
        col_max = float(numeric_series.max())
        if abs(col_max - col_min) < 1e-9:
            ratio = 0.5
        else:
            ratio = (numeric_value - col_min) / (col_max - col_min)
        if lower_is_better and column in lower_is_better:
            ratio = 1.0 - ratio
        elif higher_is_better and column not in higher_is_better:
            ratio = 1.0 - ratio

    return _diverging_heatmap_hex(ratio)


def _display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(columns=DISPLAY_LABELS)


def _split_matchup_labels(series: pd.Series) -> tuple[pd.Series, pd.Series] | None:
    parts = series.fillna("").astype(str).str.split("@", n=1, expand=True)
    if parts.shape[1] < 2:
        return None
    away = parts[0].str.strip()
    home = parts[1].str.strip()
    if not (away.ne("").any() and home.ne("").any()):
        return None
    return away, home


def _visible_grid_columns(frame: pd.DataFrame, hidden_columns: set[str] | None = None) -> list[str]:
    hidden = hidden_columns or set()
    return [
        column
        for column in frame.columns
        if column not in hidden and not str(column).startswith("__")
    ]


def _hitter_confidence_label(pitch_count: object, bip: object) -> str:
    pitches = pd.to_numeric(pd.Series([pitch_count]), errors="coerce").iloc[0]
    batted_balls = pd.to_numeric(pd.Series([bip]), errors="coerce").iloc[0]
    if pd.isna(pitches) or pd.isna(batted_balls):
        return "Medium"
    if float(batted_balls) >= 75 and float(pitches) >= 100:
        return "High"
    if float(batted_balls) >= 35 and float(pitches) >= 100:
        return "Medium"
    if float(batted_balls) >= 15:
        return "Thin"
    return "Very Thin"


def _add_hitter_confidence_payload(frame: pd.DataFrame, color_hitter_confidence: bool) -> pd.DataFrame:
    if not color_hitter_confidence or "hitter_name" not in frame.columns:
        return frame
    enriched = frame.copy()
    if HITTER_CONFIDENCE_LABEL_COLUMN in enriched.columns and HITTER_CONFIDENCE_COLOR_COLUMN in enriched.columns:
        return enriched
    labels = [
        _hitter_confidence_label(row.get("pitch_count"), row.get("bip"))
        for _, row in enriched.iterrows()
    ]
    enriched[HITTER_CONFIDENCE_LABEL_COLUMN] = labels
    enriched[HITTER_CONFIDENCE_COLOR_COLUMN] = [
        HITTER_CONFIDENCE_COLORS.get(label, HITTER_CONFIDENCE_COLORS["Medium"])
        for label in labels
    ]
    return enriched


@st.cache_data(show_spinner=False)
def _build_lightweight_grid_payload(
    frame: pd.DataFrame,
    lower_is_better: tuple[str, ...],
    higher_is_better: tuple[str, ...],
    hidden_columns: tuple[str, ...] = (),
    color_hitter_confidence: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hidden = set(hidden_columns) | HITTER_CONFIDENCE_HELPER_COLUMNS | HR_FORM_HELPER_COLUMNS
    work = _add_hitter_confidence_payload(frame, color_hitter_confidence)
    visible_columns = _visible_grid_columns(work, hidden)
    display_frame = pd.DataFrame(index=work.index)
    styles = pd.DataFrame("", index=work.index, columns=[DISPLAY_LABELS.get(column, column) for column in visible_columns])

    lower_set = set(lower_is_better)
    higher_set = set(higher_is_better)

    for column in visible_columns:
        display_column = DISPLAY_LABELS.get(column, column)
        if column == "team":
            display_frame[display_column] = [
                team_logo_data_uri(value) or ""
                for value in work[column]
            ]
        elif column in {"game", "game_label"} and not {"away_logo", "home_logo"}.issubset(set(visible_columns)):
            matchup_parts = _split_matchup_labels(work[column])
            if matchup_parts is not None:
                away, home = matchup_parts
                display_frame["Away"] = [team_logo_data_uri(value) or "" for value in away]
                display_frame["@"] = "@"
                display_frame["Home"] = [team_logo_data_uri(value) or "" for value in home]
                for logo_display_column in ("Away", "@", "Home"):
                    if logo_display_column not in styles.columns:
                        styles[logo_display_column] = ""
                continue
            display_frame[display_column] = [_format_value(column, value) for value in work[column]]
        else:
            display_frame[display_column] = [_format_value(column, value) for value in work[column]]
        if column == "hr_form" and HR_FORM_PCT_COLUMN in work.columns:
            styles[display_column] = [
                f"background-color: {background}; color: #1f1f1f" if background else ""
                for background in [_hr_form_heatmap_hex(value) for value in work[HR_FORM_PCT_COLUMN]]
            ]
        elif column in PERCENT_COLUMNS or column in RATE_COLUMNS:
            column_styles: list[str] = []
            for value in work[column]:
                background = _background_hex(
                    column,
                    value,
                    work[column],
                    lower_is_better=lower_set,
                    higher_is_better=higher_set,
                )
                column_styles.append(f"background-color: {background}; color: #1f1f1f" if background else "")
            styles[display_column] = column_styles
        elif column == "hitter_name" and color_hitter_confidence and HITTER_CONFIDENCE_COLOR_COLUMN in work.columns:
            styles[display_column] = [
                f"color: {color}; font-weight: 650" if color else ""
                for color in work[HITTER_CONFIDENCE_COLOR_COLUMN]
            ]
    styles = styles.reindex(index=display_frame.index, columns=display_frame.columns, fill_value="")
    return display_frame, styles


def render_custom_metric_table(
    frame: pd.DataFrame,
    *,
    key: str,
    height: int = 320,
    metric_styles: dict[str, dict[str, object]] | None = None,
) -> pd.DataFrame:
    if frame.empty:
        st.info("No data available for this selection.")
        return frame

    display_frame = frame.copy()
    styles = pd.DataFrame("", index=display_frame.index, columns=display_frame.columns)
    metric_styles = metric_styles or {}

    for column, config in metric_styles.items():
        if column not in display_frame.columns:
            continue
        numeric = pd.to_numeric(display_frame[column], errors="coerce")
        style_values: list[str] = []
        mode = str(config.get("mode", "high")).strip().lower()
        low = float(config.get("low", 0.0))
        high = float(config.get("high", 1.0))
        ideal = float(config.get("ideal", (low + high) / 2.0))
        for value in numeric:
            if pd.isna(value):
                style_values.append("")
                continue
            if mode == "target":
                ratio = _two_sided_target_ratio(float(value), low, ideal, high)
                background = _zone_heatmap_hex(ratio)
            else:
                if abs(high - low) < 1e-9:
                    ratio = 0.5
                else:
                    ratio = (float(value) - low) / (high - low)
                ratio = max(0.0, min(1.0, ratio))
                if mode == "low":
                    ratio = 1.0 - ratio
                background = _diverging_heatmap_hex(ratio)
            style_values.append(f"background-color: {background}; color: #1f1f1f")
        styles[column] = style_values

    display_frame = display_frame.rename(columns=DISPLAY_LABELS)
    styles = styles.rename(columns=DISPLAY_LABELS)
    st.dataframe(
        display_frame.style.apply(lambda _: styles, axis=None),
        hide_index=True,
        use_container_width=True,
        height=height,
        column_config={
            column: st.column_config.ImageColumn(column, width="small")
            for column in display_frame.columns
            if column in {DISPLAY_LABELS.get(logo_column, logo_column) for logo_column in LOGO_COLUMNS}
        },
    )
    return frame


def render_exit_velo_summary_grid(
    frame: pd.DataFrame,
    *,
    key: str,
    height: int = 520,
) -> pd.DataFrame:
    if frame.empty:
        st.info("No data available for this selection.")
        return frame
    display_frame = frame.copy()
    metric_order = ["BBE", "Avg EV", "Max EV", "PFB%", "FB%", "HH%", "Brl"]
    divider_css = "border-right: 4px solid #20354d; box-shadow: inset -2px 0 0 #20354d"
    divider_left_css = "border-left: 2px solid rgba(32, 53, 77, 0.55)"
    style_frame = pd.DataFrame("", index=display_frame.index, columns=display_frame.columns)

    def _fixed_band_hex(value: object, low: float, mid: float, high: float) -> str:
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(numeric):
            return ""
        if numeric <= mid:
            ratio = 1.0 if abs(mid - low) < 1e-9 else max(0.0, min(1.0, (float(numeric) - low) / (mid - low)))
            return _diverging_heatmap_hex(0.5 * ratio)
        ratio = 1.0 if abs(high - mid) < 1e-9 else max(0.0, min(1.0, (float(numeric) - mid) / (high - mid)))
        return _diverging_heatmap_hex(0.5 + (0.5 * ratio))

    base_band_map: dict[str, tuple[float, float, float]] = {
        "Avg EV": (85.0, 90.0, 95.0),
        "Max EV": (95.0, 103.0, 110.0),
        "PFB%": (10.0, 25.0, 40.0),
        "FB%": (15.0, 30.0, 45.0),
        "HH%": (20.0, 40.0, 60.0),
    }
    brl_high_map = {"L1": 1.0, "L3": 2.0, "L5": 3.0, "L10": 4.0, "L15": 5.0, "L25": 6.0}
    present_windows: list[str] = []
    logo_display_labels = {DISPLAY_LABELS.get(logo_column, logo_column) for logo_column in LOGO_COLUMNS}
    for column in display_frame.columns:
        if column in ({"Player", "Team"} | LOGO_COLUMNS):
            continue
        parts = str(column).split(" ", 1)
        if len(parts) == 2 and parts[0] not in present_windows:
            present_windows.append(parts[0])

    for column in display_frame.columns:
        if column in ({"Player", "Team"} | LOGO_COLUMNS):
            continue
        if column.endswith("Brl"):
            window = str(column).split(" ", 1)[0]
            high = brl_high_map.get(window, 3.0)
            mid = max(high / 2.0, 0.5)
            colors = [_fixed_band_hex(value, 0.0, mid, high) for value in display_frame[column]]
        else:
            colors = None
            for metric_suffix, band in base_band_map.items():
                if column.endswith(metric_suffix):
                    colors = [_fixed_band_hex(value, band[0], band[1], band[2]) for value in display_frame[column]]
                    break
        if colors is not None:
            style_frame[column] = [
                f"background-color: {color}; color: #1f1f1f" if color else ""
                for color in colors
            ]
    group_starts: list[str] = []
    group_ends: list[str] = []
    for column in style_frame.columns:
        if column in {"Player", "Team"}:
            continue
        parts = str(column).split(" ", 1)
        if len(parts) == 2:
            if parts[1] == metric_order[0]:
                group_starts.append(column)
            if parts[1] == metric_order[-1]:
                group_ends.append(column)

    for index, column in enumerate(group_starts):
        if index == 0:
            continue
        style_frame[column] = style_frame[column].fillna("").astype(str) + f"; {divider_left_css}"

    for column in group_ends:
        style_frame[column] = style_frame[column].fillna("").astype(str) + f"; {divider_css}"

    multi_columns: list[tuple[str, str]] = []
    for column in display_frame.columns:
        if column == "Player":
            multi_columns.append(("Player", ""))
        elif column == "Team":
            multi_columns.append(("Team", ""))
        elif column in LOGO_COLUMNS:
            multi_columns.append((DISPLAY_LABELS.get(column, column), ""))
        else:
            parts = str(column).split(" ", 1)
            multi_columns.append((parts[0], parts[1] if len(parts) == 2 else column))
    display_frame.columns = pd.MultiIndex.from_tuples(multi_columns)
    style_frame.columns = display_frame.columns

    formatters: dict[tuple[str, str], object] = {}
    for column in display_frame.columns:
        if column[0] in ({"Player", "Team"} | logo_display_labels):
            continue
        metric_name = column[1]
        if metric_name in {"BBE", "Brl"}:
            formatters[column] = lambda value: "" if pd.isna(value) else f"{int(value)}"
        else:
            formatters[column] = lambda value: "" if pd.isna(value) else f"{float(value):.2f}"

    styler = (
        display_frame.style
        .apply(lambda _: style_frame, axis=None)
        .format(formatters, na_rep="")
    )
    st.dataframe(
        styler,
        hide_index=True,
        use_container_width=True,
        height=height,
        column_config={
            # Streamlit positional column config counts the hidden index at 0,
            # so visible dataframe columns are one-based here.
            index + 1: st.column_config.ImageColumn(column[0], width="small")
            for index, column in enumerate(display_frame.columns)
            if column[0] in {DISPLAY_LABELS.get(logo_column, logo_column) for logo_column in LOGO_COLUMNS}
        },
    )
    return frame


def _prepare_grid_frame(
    frame: pd.DataFrame,
    lower_is_better: set[str] | None = None,
    higher_is_better: set[str] | None = None,
    color_hitter_confidence: bool = False,
) -> pd.DataFrame:
    prepared = _add_hitter_confidence_payload(frame, color_hitter_confidence)
    for column in frame.columns:
        if column in PERCENT_COLUMNS or column in RATE_COLUMNS:
            prepared[f"__style_{column}"] = [
                _background_hex(
                    column,
                    value,
                    frame[column],
                    lower_is_better=lower_is_better or LOWER_IS_BETTER,
                    higher_is_better=higher_is_better or HIGHER_IS_BETTER,
                ) or ""
                for value in frame[column]
            ]
    if "hr_form" in prepared.columns and HR_FORM_PCT_COLUMN in prepared.columns:
        prepared["__style_hr_form"] = [
            _hr_form_heatmap_hex(value) or ""
            for value in prepared[HR_FORM_PCT_COLUMN]
        ]
    return prepared


def _formatted_length(column: str, value: object) -> int:
    return len(_format_value(column, value))


def _column_width(column: str, series: pd.Series) -> tuple[int, int, int]:
    header_name = DISPLAY_LABELS.get(column, column)
    content_lengths = [_formatted_length(column, value) for value in series.head(50)]
    max_len = max([len(header_name), *content_lengths, 4])
    if column in SHORT_COLUMNS:
        min_width = 70
        max_width = 110
        scale = 8
    elif column in LONG_COLUMNS:
        min_width = 140
        max_width = 260
        scale = 9
    elif column in MEDIUM_COLUMNS or column in PERCENT_COLUMNS or column in RATE_COLUMNS:
        min_width = 92
        max_width = 135
        scale = 8
    else:
        min_width = 100
        max_width = 180
        scale = 8
    width = max(min_width, min(max_width, max_len * scale + 24))
    return width, min_width, max_width


def _react_column_payload(column: str, series: pd.Series) -> dict[str, object]:
    header_name = DISPLAY_LABELS.get(column, column)
    width, _, _ = _column_width(column, series)
    kind = "text"
    if column in LONG_COLUMNS:
        kind = "long"
    elif column in SHORT_COLUMNS:
        kind = "short"
    if column in PERCENT_COLUMNS or column in RATE_COLUMNS or column in INTEGER_COLUMNS:
        kind = "numeric"
    return {
        "key": column,
        "label": header_name,
        "numeric": column in PERCENT_COLUMNS or column in RATE_COLUMNS or column in INTEGER_COLUMNS,
        "heat": column in PERCENT_COLUMNS or column in RATE_COLUMNS,
        "width": width,
        "kind": kind,
    }


def _build_react_table_payload(
    frame: pd.DataFrame,
    lower_is_better: set[str] | None = None,
    higher_is_better: set[str] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    lower_set = lower_is_better or LOWER_IS_BETTER
    higher_set = higher_is_better or HIGHER_IS_BETTER
    columns = [_react_column_payload(column, frame[column]) for column in frame.columns]
    rows: list[dict[str, object]] = []
    for idx, (_, row) in enumerate(frame.iterrows()):
        cells: dict[str, dict[str, object]] = {}
        for column in frame.columns:
            value = row[column]
            background = None
            if column == "hr_form" and HR_FORM_PCT_COLUMN in frame.columns:
                background = _hr_form_heatmap_hex(row.get(HR_FORM_PCT_COLUMN))
            elif column in PERCENT_COLUMNS or column in RATE_COLUMNS:
                background = _background_hex(
                    column,
                    value,
                    frame[column],
                    lower_is_better=lower_set,
                    higher_is_better=higher_set,
                )
            cells[column] = {
                "display": _format_value(column, value),
                "sort": None if pd.isna(value) else value,
                "background": background,
            }
        rows.append({"row_id": str(idx), "cells": cells})
    return columns, rows


def _render_sticky_html_table(
    display_frame: pd.DataFrame,
    styles: pd.DataFrame,
    height: int,
) -> None:
    """Render display_frame as a scrollable HTML table with the first column sticky."""
    import html as _html
    import streamlit.components.v1 as _c

    _LOGO_COLS = {"Away", "Home", "Team"}
    _NON_NUMERIC = {
        "Hitter", "Pitcher", "Player", "Away", "Home", "Team", "@",
        "Pitch", "Throws", "Split", "Pitch Type", "Family", "Zone",
        "Window", "Game", "Lineup Source",
    }

    cols = list(display_frame.columns)

    def _parse_style(s: str):
        bg = fg = None
        if not s:
            return bg, fg
        for part in s.split(";"):
            p = part.strip()
            if p.startswith("background-color:"):
                bg = p[len("background-color:"):].strip()
            elif p.startswith("color:"):
                fg = p[len("color:"):].strip()
        return bg, fg

    hdr = []
    for i, col in enumerate(cols):
        sticky = ' class="sc"' if i == 0 else ""
        align = "center" if col in _LOGO_COLS else ("right" if col not in _NON_NUMERIC else "left")
        hdr.append(f'<th{sticky} style="text-align:{align}">{_html.escape(str(col))}</th>')

    body = []
    for ri in range(len(display_frame)):
        cells = []
        for ci, col in enumerate(cols):
            val = display_frame.iloc[ri][col]
            style_str = styles.iloc[ri][col] if col in styles.columns else ""
            bg, fg = _parse_style(style_str)
            is_logo = col in _LOGO_COLS
            align = "center" if is_logo else ("right" if col not in _NON_NUMERIC else "left")

            if ci == 0:
                color_attr = f";color:{fg}" if fg else ""
                cells.append(
                    f'<td class="sc" style="font-weight:700{color_attr}">'
                    f'{_html.escape(str(val) if val is not None else "")}</td>'
                )
            elif is_logo and val:
                cells.append(
                    f'<td style="text-align:center">'
                    f'<img src="{val}" style="height:20px;width:20px;object-fit:contain;vertical-align:middle"/>'
                    f'</td>'
                )
            elif bg:
                cells.append(
                    f'<td style="text-align:{align}">'
                    f'<span style="display:inline-block;background:{_html.escape(bg)};border-radius:7px;'
                    f'padding:3px 8px;font-weight:700;color:#1f1f1f;min-width:52px;text-align:right">'
                    f'{_html.escape(str(val) if val is not None else "")}</span></td>'
                )
            else:
                cells.append(
                    f'<td style="text-align:{align}">'
                    f'{_html.escape(str(val) if val is not None else "")}</td>'
                )
        body.append(f'<tr>{"".join(cells)}</tr>')

    html_out = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:transparent;font-family:"Segoe UI",system-ui,sans-serif;font-size:11.5px;color:#1f1f1f}}
.wrap{{height:{height}px;overflow:auto;border:1px solid #e6dcc8;border-radius:12px;background:#fbf7ef;
scrollbar-width:thin;scrollbar-color:#c8b99d #f8f2e6}}
.wrap::-webkit-scrollbar{{width:8px;height:8px}}
.wrap::-webkit-scrollbar-thumb{{background:#cfbea2;border-radius:999px;border:2px solid #f7f1e6}}
.wrap::-webkit-scrollbar-track{{background:#f7f1e6}}
table{{border-collapse:collapse;table-layout:auto}}
th{{position:sticky;top:0;z-index:2;background:linear-gradient(180deg,#21385d 0%,#182b48 100%);
color:#f8f7f3;font-size:10px;text-transform:uppercase;letter-spacing:0.07em;
padding:8px 10px;border-bottom:1px solid #30456c;white-space:nowrap;user-select:none}}
td{{padding:7px 10px;border-bottom:1px solid rgba(230,220,200,0.85);white-space:nowrap;
vertical-align:middle;font-variant-numeric:tabular-nums}}
tr:last-child td{{border-bottom:none}}
tr:hover td{{background:rgba(242,239,231,0.9)!important}}
tr:nth-child(even) td{{background:rgba(251,247,239,0.6)}}
.sc{{position:sticky;left:0;z-index:3;background:#faf6ee;min-width:120px}}
th.sc{{z-index:4;background:linear-gradient(180deg,#21385d 0%,#182b48 100%)}}
tr:nth-child(even) .sc{{background:#f3efe7}}
tr:hover .sc{{background:#ede9e0!important}}
</style></head><body>
<div class="wrap">
<table>
<thead><tr>{"".join(hdr)}</tr></thead>
<tbody>{"".join(body)}</tbody>
</table>
</div>
</body></html>"""

    _c.html(html_out, height=height + 4)


def render_metric_grid(
    frame: pd.DataFrame,
    key: str,
    height: int = 320,
    lower_is_better: set[str] | None = None,
    higher_is_better: set[str] | None = None,
    use_lightweight: bool = False,
    use_react: bool = False,
    title: str | None = None,
    subtitle: str | None = None,
    hidden_columns: set[str] | None = None,
    color_hitter_confidence: bool = False,
    post_styler: object = None,
) -> pd.DataFrame:
    if frame.empty:
        st.info("No data available for this selection.")
        return frame
    hidden = set(hidden_columns or set()) | HITTER_CONFIDENCE_HELPER_COLUMNS | HR_FORM_HELPER_COLUMNS
    visible_columns = _visible_grid_columns(frame, hidden)
    if use_lightweight or not HAS_AGGRID:
        display_frame, styles = _build_lightweight_grid_payload(
            frame,
            tuple(sorted(lower_is_better or LOWER_IS_BETTER)),
            tuple(sorted(higher_is_better or HIGHER_IS_BETTER)),
            tuple(sorted(hidden)),
            color_hitter_confidence,
        )
        if post_styler is not None:
            styles = post_styler(display_frame, frame, styles)
        st.dataframe(
            display_frame.style.apply(lambda _: styles, axis=None),
            hide_index=True,
            use_container_width=True,
            height=height,
            column_config={
                column: st.column_config.ImageColumn(
                    column,
                    width="small",
                )
                for column in display_frame.columns
                if column in {DISPLAY_LABELS.get(logo_column, logo_column) for logo_column in LOGO_COLUMNS}
            },
        )
        return frame

    if not HAS_AGGRID:
        st.error("Install `streamlit-aggrid` to render the matchup tables.")
        return frame

    prepared = _prepare_grid_frame(
        frame,
        lower_is_better=lower_is_better,
        higher_is_better=higher_is_better,
        color_hitter_confidence=color_hitter_confidence,
    )
    builder = GridOptionsBuilder.from_dataframe(prepared)
    builder.configure_default_column(sortable=True, resizable=True, filter=False)
    builder.configure_grid_options(
        tooltipShowDelay=0,
    )

    cell_style = JsCode(
        """
        function(params) {
          if (params.colDef.field === "hitter_name") {
            const confidenceColor = params.data ? params.data["__hitter_confidence_color"] : "";
            if (confidenceColor) {
              return {color: confidenceColor, fontWeight: "650"};
            }
          }
          const styleKey = "__style_" + params.colDef.field;
          const bg = params.data ? params.data[styleKey] : "";
          if (bg) {
            return {backgroundColor: bg, color: "#1f1f1f"};
          }
          return {color: "#1f1f1f"};
        }
        """
    )
    percent_formatter = JsCode(
        """
        function(params) {
          if (params.value === null || params.value === undefined || params.value === "") { return "-"; }
          return (Number(params.value) * 100).toFixed(1) + "%";
        }
        """
    )
    rate_formatter = JsCode(
        """
        function(params) {
          if (params.value === null || params.value === undefined || params.value === "") { return "-"; }
          return Number(params.value).toFixed(3);
        }
        """
    )
    one_decimal_formatter = JsCode(
        """
        function(params) {
          if (params.value === null || params.value === undefined || params.value === "") { return "-"; }
          return Number(params.value).toFixed(1);
        }
        """
    )
    integer_formatter = JsCode(
        """
        function(params) {
          if (params.value === null || params.value === undefined || params.value === "") { return "-"; }
          return Math.round(Number(params.value)).toLocaleString();
        }
        """
    )

    for column in visible_columns:
        header_name = DISPLAY_LABELS.get(column, column)
        formatter = None
        width, min_width, max_width = _column_width(column, frame[column])
        if column in PERCENT_COLUMNS:
            formatter = percent_formatter
        elif column in {"xwoba", "xwoba_con", "iso", "matchup_score", "test_score", "ceiling_score", "khr_score", "pitcher_score"}:
            formatter = rate_formatter
        elif column in {"avg_launch_angle", "avg_release_speed", "gb_fb_ratio"}:
            formatter = one_decimal_formatter
        elif column in {"avg_spin_rate", *INTEGER_COLUMNS}:
            formatter = integer_formatter
        builder.configure_column(
            column,
            header_name=header_name,
            cellStyle=cell_style if column in PERCENT_COLUMNS or column in RATE_COLUMNS or column in {"hitter_name", "hr_form"} else None,
            valueFormatter=formatter,
            width=width,
            minWidth=min_width,
            maxWidth=max_width,
        )
        style_col = f"__style_{column}"
        if style_col in prepared.columns:
            builder.configure_column(style_col, hide=True)

    for column in prepared.columns:
        if column not in visible_columns:
            builder.configure_column(column, hide=True)

    grid_options = builder.build()
    response = AgGrid(
        prepared,
        gridOptions=grid_options,
        height=height,
        fit_columns_on_grid_load=False,
        allow_unsafe_jscode=True,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        theme="streamlit",
        key=key,
    )
    returned = pd.DataFrame(response["data"])
    if returned.empty:
        return frame
    return returned.loc[:, [column for column in visible_columns if column in returned.columns]]


def _game_label(game: dict) -> str:
    return f"{game.get('away_team', '')} @ {game.get('home_team', '')}"


def resolve_logo_game_selection(slate: list[dict], selected_key: object | None) -> tuple[str, list[dict]]:
    if not slate:
        return "Slate Summary", []
    if selected_key == SLATE_SUMMARY_SELECTION:
        return "Slate Summary", []

    game_by_key = {str(game.get("game_pk")): game for game in slate if game.get("game_pk") is not None}
    first_key = next(iter(game_by_key), None)
    selected = str(selected_key) if selected_key is not None else first_key
    if selected not in game_by_key:
        selected = first_key
    if selected is None:
        return "Slate Summary", []
    game = game_by_key[selected]
    return _game_label(game), [game]


def render_logo_game_selector(slate: list[dict], *, key_prefix: str) -> tuple[str, list[dict]]:
    if not slate:
        return "Slate Summary", []

    state_key = f"{key_prefix}-selected-game-pk"
    component_key = f"{key_prefix}-component"
    valid_keys = [str(game.get("game_pk")) for game in slate if game.get("game_pk") is not None]
    valid_selection_keys = {SLATE_SUMMARY_SELECTION, *valid_keys}
    component_state_selection = st.session_state.get(component_key)
    if component_state_selection in valid_selection_keys:
        st.session_state[state_key] = str(component_state_selection)
    if state_key not in st.session_state or st.session_state[state_key] not in valid_selection_keys:
        st.session_state[state_key] = valid_keys[0] if valid_keys else SLATE_SUMMARY_SELECTION

    selected = str(st.session_state[state_key])
    cards: list[dict[str, object]] = [
        {
            "selectionKey": SLATE_SUMMARY_SELECTION,
            "isSummary": True,
        }
    ]
    for game in slate:
        game_key = str(game.get("game_pk"))
        away_team = str(game.get("away_team", "") or "")
        home_team = str(game.get("home_team", "") or "")
        cards.append(
            {
                "selectionKey": game_key,
                "awayTeam": away_team,
                "homeTeam": home_team,
                "awayLogo": team_logo_data_uri(away_team),
                "homeLogo": team_logo_data_uri(home_team),
            }
        )

    component_selection = render_game_selector(
        cards,
        selected_key=selected,
        key=component_key,
        height=250,
    )
    if component_selection in valid_selection_keys:
        st.session_state[state_key] = str(component_selection)

    return resolve_logo_game_selection(slate, st.session_state.get(state_key))


def render_sticky_logo_game_nav(
    slate: list[dict],
    *,
    key_prefix: str,
    sections: list[str],
    default_section: str = "Matchup",
) -> tuple[str, list[dict], str]:
    if not slate:
        return "Slate Summary", [], default_section

    state_key = f"{key_prefix}-selected-game-pk"
    component_key = f"{key_prefix}-sticky-component"
    valid_keys = [str(game.get("game_pk")) for game in slate if game.get("game_pk") is not None]
    valid_selection_keys = {SLATE_SUMMARY_SELECTION, *valid_keys}
    valid_sections = set(sections)

    def _apply_component_value(value: object) -> None:
        if not isinstance(value, dict):
            return
        selection = value.get("selectionKey")
        section = value.get("section")
        if selection in valid_selection_keys:
            st.session_state[state_key] = str(selection)
        if selection in valid_keys and section in valid_sections:
            st.session_state[f"section-{selection}"] = str(section)

    _apply_component_value(st.session_state.get(component_key))
    if state_key not in st.session_state or st.session_state[state_key] not in valid_selection_keys:
        st.session_state[state_key] = valid_keys[0] if valid_keys else SLATE_SUMMARY_SELECTION

    selected = str(st.session_state[state_key])
    selected_section = default_section
    if selected in valid_keys:
        section_key = f"section-{selected}"
        if st.session_state.get(section_key) not in valid_sections:
            st.session_state[section_key] = default_section
        selected_section = str(st.session_state[section_key])

    cards: list[dict[str, object]] = [
        {
            "selectionKey": SLATE_SUMMARY_SELECTION,
            "isSummary": True,
        }
    ]
    for game in slate:
        game_key = str(game.get("game_pk"))
        away_team = str(game.get("away_team", "") or "")
        home_team = str(game.get("home_team", "") or "")
        cards.append(
            {
                "selectionKey": game_key,
                "awayTeam": away_team,
                "homeTeam": home_team,
                "awayLogo": team_logo_data_uri(away_team),
                "homeLogo": team_logo_data_uri(home_team),
                "status": str(game.get("status", "") or game.get("game_status", "") or ""),
                "gameDate": str(game.get("game_date", "") or ""),
            }
        )

    component_selection = render_sticky_game_nav(
        cards,
        selected_key=selected,
        sections=sections,
        selected_section=selected_section,
        key=component_key,
        height=270,
    )
    _apply_component_value(component_selection)

    selected_label, selected_games = resolve_logo_game_selection(slate, st.session_state.get(state_key))
    selected = str(st.session_state.get(state_key))
    selected_section = default_section
    if selected in valid_keys:
        selected_section = str(st.session_state.get(f"section-{selected}", default_section))
    return selected_label, selected_games, selected_section


def build_pitcher_summary_table(pitcher_summary_by_hand: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    split_map = [
        ("all", "All"),
        ("vs_rhh", "vs RHH"),
        ("vs_lhh", "vs LHH"),
    ]
    if pitcher_summary_by_hand.empty or "batter_side_key" not in pitcher_summary_by_hand.columns:
        for _, label in split_map:
            row = {column: None for column in PITCHER_SUMMARY_TABLE_COLUMNS if column != "split_label"}
            row["split_label"] = label
            rows.append(row)
        return pd.DataFrame(rows, columns=PITCHER_SUMMARY_TABLE_COLUMNS)
    for side_key, label in split_map:
        side_frame = pitcher_summary_by_hand.loc[pitcher_summary_by_hand["batter_side_key"] == side_key].copy()
        if side_frame.empty:
            row = {column: None for column in PITCHER_SUMMARY_TABLE_COLUMNS if column != "split_label"}
            row["split_label"] = label
            rows.append(row)
            continue
        first_row = side_frame.iloc[0]
        row = {"split_label": label}
        for column in PITCHER_SUMMARY_TABLE_COLUMNS:
            if column == "split_label":
                continue
            row[column] = first_row.get(column)
        rows.append(row)
    return pd.DataFrame(rows, columns=PITCHER_SUMMARY_TABLE_COLUMNS)


def render_matchup_header(game: dict) -> None:
    st.markdown(
        f"""
<div class="matchup-title">
  {matchup_logo_html(str(game.get("away_team", "")), str(game.get("home_team", "")), size=38)}
</div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"{game.get('status', 'Scheduled')} | Game PK: {game['game_pk']}")


def _zone_map_lookup(zone_map: pd.DataFrame) -> dict[int, dict[str, float]]:
    if zone_map.empty:
        return {}
    lookup: dict[int, dict[str, float]] = {}
    for _, row in zone_map.iterrows():
        zone = row.get("zone")
        if pd.isna(zone):
            continue
        lookup[int(float(zone))] = {
            "zone_value": row.get("zone_value"),
            "display_value": row.get("display_value"),
            "sample_size": row.get("sample_size"),
        }
    return lookup


def _draw_home_plate(draw: ImageDraw.ImageDraw, center_x: int, top_y: int, fill: str, outline: str) -> None:
    points = [
        (center_x - 34, top_y),
        (center_x + 34, top_y),
        (center_x + 48, top_y + 24),
        (center_x, top_y + 42),
        (center_x - 48, top_y + 24),
    ]
    draw.polygon(points, fill=fill, outline=outline)


@st.cache_data(show_spinner=False)
def _build_zone_heatmap_image(title: str, subtitle: str, zone_map: pd.DataFrame, value_mode: str = "percent") -> bytes:
    if not HAS_PILLOW:
        raise RuntimeError("Pillow is required for heatmap rendering.")
    width = 680
    height = 760
    image = Image.new("RGB", (width, height), "#15212a")
    draw = ImageDraw.Draw(image)
    title_font = _load_font(28, bold=True)
    subtitle_font = _load_font(16, bold=False)
    legend_font = _load_font(14, bold=True)
    zone_value_font = _load_font(28, bold=True)
    zone_sample_font = _load_font(13, bold=True)
    label_font = _load_font(13, bold=True)

    for y in range(height):
        ratio = y / max(height - 1, 1)
        row_color = _blend_color("#12202b", "#223746", ratio)
        draw.line((0, y, width, y), fill=row_color, width=1)

    draw.rounded_rectangle((12, 12, width - 12, height - 12), radius=28, fill="#223744", outline="#415765", width=2)
    draw.rounded_rectangle((28, 24, width - 28, 106), radius=22, fill="#f6f1e5", outline="#ddcfb6", width=1)
    _text(draw, (48, 38), title, font=title_font, fill="#15324d")
    _text(draw, (48, 72), subtitle, font=subtitle_font, fill="#546574")

    legend_left = width - 180
    legend_top = 132
    legend_right = width - 40
    draw.rounded_rectangle((legend_left, legend_top, legend_right, legend_top + 168), radius=18, fill="#20323d", outline="#47606d", width=1)
    _text(draw, (legend_left + 18, legend_top + 14), "Zone Score", font=legend_font, fill="#edf2f5")
    legend_labels = [("Best", "#ef8d32"), ("Above Avg", "#c39656"), ("Neutral", "#a8b0ad"), ("Below Avg", "#79a7ca"), ("Worst", "#58a7e4")]
    for idx, (label, color) in enumerate(legend_labels):
        y = legend_top + 44 + idx * 24
        draw.rounded_rectangle((legend_left + 18, y, legend_left + 42, y + 16), radius=5, fill=color, outline="#dce4e8", width=1)
        _text(draw, (legend_left + 54, y - 1), label, font=label_font, fill="#eef4f7")

    grid_origin_x = 54
    grid_origin_y = 136
    unit = 90
    gap = 8
    lookup = _zone_map_lookup(zone_map)
    zone_values = [value["zone_value"] for value in lookup.values() if value.get("zone_value") is not None and not pd.isna(value.get("zone_value"))]
    min_val = min(zone_values) if zone_values else 0.0
    max_val = max(zone_values) if zone_values else 1.0

    def zone_ratio(value: float | None) -> float:
        if value is None or pd.isna(value):
            return 0.5
        if abs(max_val - min_val) < 1e-9:
            return 0.5
        return (float(value) - min_val) / (max_val - min_val)

    zone_panel_left = grid_origin_x - 16
    zone_panel_top = grid_origin_y - 16
    zone_panel_right = grid_origin_x + 5 * unit + 4 * gap + 16
    zone_panel_bottom = grid_origin_y + 5 * unit + 4 * gap + 58
    draw.rounded_rectangle(
        (zone_panel_left, zone_panel_top, zone_panel_right, zone_panel_bottom),
        radius=26,
        fill="#1a2c37",
        outline="#4b6672",
        width=2,
    )

    for zone, (grid_x, grid_y, span_x, span_y) in ZONE_RECTANGLES.items():
        x0 = grid_origin_x + grid_x * (unit + gap)
        y0 = grid_origin_y + grid_y * (unit + gap)
        x1 = x0 + span_x * unit + (span_x - 1) * gap
        y1 = y0 + span_y * unit + (span_y - 1) * gap
        zone_info = lookup.get(zone, {})
        color = _zone_heatmap_hex(zone_ratio(zone_info.get("zone_value")))
        draw.rounded_rectangle((x0 + 2, y0 + 4, x1 + 2, y1 + 4), radius=12, fill="#0f1920")
        draw.rounded_rectangle((x0, y0, x1, y1), radius=12, fill=color, outline="#f1f5f6", width=2)
        sample_size = zone_info.get("sample_size")
        display_value = zone_info.get("display_value")
        label = "-"
        if display_value is not None and not pd.isna(display_value):
            if value_mode == "percent":
                label = f"{float(display_value) * 100:.0f}"
            else:
                label = f"{float(display_value) * 100:.0f}"
        label_bbox = draw.textbbox((0, 0), label, font=zone_value_font)
        label_x = x0 + ((x1 - x0) - (label_bbox[2] - label_bbox[0])) / 2
        label_y = y0 + ((y1 - y0) - (label_bbox[3] - label_bbox[1])) / 2 - 12
        _text(draw, (int(label_x), int(label_y)), label, font=zone_value_font, fill="#0f171b")
        if sample_size is not None and not pd.isna(sample_size):
            sample_text = f"{int(float(sample_size))} pitches"
            sample_bbox = draw.textbbox((0, 0), sample_text, font=zone_sample_font)
            sample_x = x0 + ((x1 - x0) - (sample_bbox[2] - sample_bbox[0])) / 2
            sample_y = label_y + 34
            _text(draw, (int(sample_x), int(sample_y)), sample_text, font=zone_sample_font, fill="#22343b")

    strike_left = grid_origin_x + (unit + gap)
    strike_top = grid_origin_y + (unit + gap)
    strike_right = strike_left + 3 * unit + 2 * gap
    strike_bottom = strike_top + 3 * unit + 2 * gap
    draw.rounded_rectangle((strike_left - 6, strike_top - 6, strike_right + 6, strike_bottom + 6), radius=14, outline="#f4f7f8", width=4)
    draw.rounded_rectangle((strike_left - 2, strike_top - 2, strike_right + 2, strike_bottom + 2), radius=10, outline="#8aa6b4", width=1)
    _draw_home_plate(draw, (strike_left + strike_right) // 2, strike_bottom + 34, fill="#f2f5f6", outline="#d7e1e6")

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_zone_heatmap(title: str, subtitle: str, zone_map: pd.DataFrame, value_mode: str = "percent") -> None:
    if zone_map.empty:
        st.info("No zone heatmap data available.")
        return
    st.image(_build_zone_heatmap_image(title, subtitle, zone_map, value_mode=value_mode), use_container_width=False)


def render_zone_tool(
    title: str,
    subtitle: str,
    zone_map: pd.DataFrame,
    key: str,
    value_mode: str = "percent",
    map_kind: str = "zone",
    overlay_zone_map: pd.DataFrame | None = None,
) -> None:
    if zone_map.empty:
        st.info("No zone heatmap data available.")
        return
    zone_cols = ["zone", "sample_size", "zone_value", "display_value"]
    overlay_rows = overlay_zone_map[[c for c in zone_cols if c in overlay_zone_map.columns]] if overlay_zone_map is not None and not overlay_zone_map.empty else None
    render_zone_tool_component(
        zone_rows=zone_map[[c for c in zone_cols if c in zone_map.columns]],
        key=key,
        title=title,
        subtitle=subtitle,
        value_mode=value_mode,
        map_kind=map_kind,
        overlay_zone_rows=overlay_rows,
    )


def render_weather_field(
    venue_name: str,
    lf_distance_ft: object,
    cf_distance_ft: object,
    rf_distance_ft: object,
    wind_speed_mph: object,
    wind_direction_deg: object,
) -> bytes | None:
    if not HAS_PILLOW:
        return None

    width = 300
    height = 248
    image = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    icon_left = 20
    icon_top = 12
    icon_right = width - 12
    icon_bottom = height - 14
    cx = (icon_left + icon_right) // 2
    cy = (icon_top + icon_bottom) // 2

    draw.rounded_rectangle(
        (icon_left, icon_top, icon_right, icon_bottom),
        radius=22,
        fill="#f7faf6",
        outline="#edf2eb",
        width=1,
    )

    # Soft glow behind the field mark.
    draw.ellipse((cx - 82, cy - 84, cx + 82, cy + 72), fill="#eef8eb", outline=None)

    # Simple outfield arc and foul poles like the reference tile.
    arc_box = (cx - 98, cy - 96, cx + 98, cy + 34)
    draw.arc(arc_box, start=208, end=332, fill="#92edb1", width=6)
    draw.line((cx - 84, cy + 4, cx - 24, cy - 58), fill="#c6edd0", width=4)
    draw.line((cx + 84, cy + 4, cx + 24, cy - 58), fill="#c6edd0", width=4)

    # Minimal diamond and bases.
    diamond = [
        (cx, cy + 22),
        (cx - 28, cy - 6),
        (cx, cy - 34),
        (cx + 28, cy - 6),
    ]
    draw.polygon(diamond, outline="#d5dbe4", fill="#fff2b8")
    draw.line([diamond[0], diamond[1], diamond[2], diamond[3], diamond[0]], fill="#d4d9e0", width=3)
    for base_x, base_y in [(cx - 28, cy - 6), (cx, cy - 34), (cx + 28, cy - 6)]:
        draw.rectangle((base_x - 5, base_y - 5, base_x + 5, base_y + 5), fill="#f5f7fb", outline="#c7cfda")
    _draw_home_plate(draw, cx, cy + 28, fill="#f5f7fb", outline="#c7cfda")

    if not pd.isna(wind_speed_mph) and not pd.isna(wind_direction_deg):
        toward_deg = (float(wind_direction_deg) + 180.0) % 360.0
        angle = radians(toward_deg)
        origin_x = cx
        origin_y = cy + 18
        length = 104
        end_x = origin_x + sin(angle) * length
        end_y = origin_y - cos(angle) * length
        draw.line((origin_x, origin_y, end_x, end_y), fill="#4f95e8", width=10)
        head = 22
        left_a = angle + radians(152)
        right_a = angle - radians(152)
        draw.line((end_x, end_y, end_x + sin(left_a) * head, end_y - cos(left_a) * head), fill="#4f95e8", width=8)
        draw.line((end_x, end_y, end_x + sin(right_a) * head, end_y - cos(right_a) * head), fill="#4f95e8", width=8)
    else:
        draw.line((cx - 30, cy + 12, cx + 30, cy - 54), fill="#b8c7d9", width=8)
        draw.line((cx + 30, cy - 54, cx + 12, cy - 52), fill="#b8c7d9", width=6)
        draw.line((cx + 30, cy - 54, cx + 28, cy - 36), fill="#b8c7d9", width=6)

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _table_section_height(title: str, frame: pd.DataFrame) -> int:
    return 32 + 28 + 26 * (len(frame) + 1)


def _export_section_height_estimate(title: str, frame: pd.DataFrame, subtitle: str = "") -> int:
    lowered = title.lower()
    subtitle_height = 20 if subtitle else 0
    if "top slate hitters" in lowered or "top slate pitchers" in lowered:
        return 82 + len(frame) * 46 + 40 + subtitle_height
    if "summary" in lowered:
        return _table_section_height(title, frame) + 24 + subtitle_height
    return 54 + len(frame) * 62 + 36 + subtitle_height


def _summary_card_color(column: str, row: pd.Series, lower_is_better: set[str] | None, higher_is_better: set[str] | None) -> str:
    if column not in PERCENT_COLUMNS and column not in RATE_COLUMNS:
        return "#f6f8fb"
    value = row.get(column)
    if pd.isna(value):
        return "#f6f8fb"
    if column in TARGET_COLUMNS:
        return _background_hex(column, value, pd.Series([value]), lower_is_better=lower_is_better, higher_is_better=higher_is_better) or "#f6f8fb"
    thresholds = {
        "xwoba": (0.24, 0.30, True),
        "swstr_pct": (0.10, 0.14, False),
        "pulled_barrel_pct": (0.03, 0.08, True),
        "barrel_bbe_pct": (0.05, 0.10, True),
        "barrel_bip_pct": (0.04, 0.09, True),
        "fb_pct": (0.20, 0.33, True),
        "hard_hit_pct": (0.34, 0.48, True),
    }
    if column not in thresholds:
        return "#f6f8fb"
    low, high, invert = thresholds[column]
    ratio = (float(value) - low) / max(high - low, 1e-9)
    ratio = max(0.0, min(1.0, ratio))
    if invert:
        ratio = 1.0 - ratio
    return _diverging_heatmap_hex(ratio)


def _infer_summary_stats(frame: pd.DataFrame) -> list[tuple[str, str]]:
    preferred = [
        ("Throws", "p_throws"),
        ("Pitches", "pitch_count"),
        ("BIP", "bip"),
        ("xwOBA", "xwoba"),
        ("SwStr%", "swstr_pct"),
        ("PulledBrl%", "pulled_barrel_pct"),
        ("Brl/BIP%", "barrel_bip_pct"),
        ("FB%", "fb_pct"),
        ("HH%", "hard_hit_pct"),
        ("LA", "avg_launch_angle"),
    ]
    return [(label, column) for label, column in preferred if column in frame.columns]


def _draw_summary_cards(
    draw: ImageDraw.ImageDraw,
    top: int,
    left: int,
    width: int,
    title: str,
    frame: pd.DataFrame,
    font: ImageFont.ImageFont,
    lower_is_better: set[str] | None = None,
    higher_is_better: set[str] | None = None,
) -> int:
    if frame.empty:
        return top
    row = frame.iloc[0]
    stats = _infer_summary_stats(frame)
    draw.text((left, top), title, fill="#102542", font=font)
    y = top + 22
    cols = 5
    card_gap = 10
    card_width = max(120, int((width - (card_gap * (cols - 1))) / cols))
    card_height = 62
    for idx, (label, column_name) in enumerate(stats):
        col_idx = idx % cols
        row_idx = idx // cols
        x0 = left + col_idx * (card_width + card_gap)
        y0 = y + row_idx * (card_height + 10)
        x1 = x0 + card_width
        y1 = y0 + card_height
        fill = _summary_card_color(column_name, row, lower_is_better, higher_is_better)
        draw.rounded_rectangle((x0, y0, x1, y1), radius=12, fill=fill, outline="#d9e2ec", width=1)
        draw.text((x0 + 10, y0 + 10), label, fill="#516273", font=font)
        draw.text((x0 + 10, y0 + 34), _format_value(column_name, row.get(column_name), export_mode=True), fill="#1f1f1f", font=font)
    rows = (len(stats) + cols - 1) // cols
    return y + rows * (card_height + 10)


def _limit_frame(frame: pd.DataFrame, rows: int, columns: list[str] | None = None) -> pd.DataFrame:
    if frame.empty:
        return frame
    limited = frame.copy()
    if columns is not None:
        limited = limited[[column for column in columns if column in limited.columns]]
    return limited.head(rows)


def _section_type(title: str) -> str:
    lowered = title.lower()
    if "best matchups" in lowered:
        return "best"
    if "summary" in lowered:
        return "summary"
    if "arsenal" in lowered:
        return "arsenal"
    if "count usage" in lowered:
        return "count"
    if "hitters" in lowered:
        return "hitters"
    return "table"


def _extract_sections(sections: list[dict], kind: str) -> list[dict]:
    return [section for section in sections if _section_type(section["title"]) == kind]


def _extract_first_matching(sections: list[dict], keyword: str) -> dict | None:
    lowered = keyword.lower()
    for section in sections:
        if lowered in section["title"].lower():
            return section
    return None


def _draw_side_by_side_tables(
    draw: ImageDraw.ImageDraw,
    top: int,
    width: int,
    left_title: str,
    left_frame: pd.DataFrame,
    right_title: str,
    right_frame: pd.DataFrame,
    font: ImageFont.ImageFont,
) -> int:
    gap = 24
    table_width = (width - gap) // 2
    left_bottom = _draw_section(draw, top, 18, table_width, left_title, left_frame, None, None, font)
    right_bottom = _draw_section(draw, top, 18 + table_width + gap, table_width, right_title, right_frame, None, None, font)
    return max(left_bottom, right_bottom)


REPORT_BG = "#0f1222"
REPORT_PANEL = "#151b33"
REPORT_PANEL_ALT = "#1a2440"
REPORT_BORDER = "#cf6bf3"
REPORT_TEXT = "#F7F8FF"
REPORT_MUTED = "#D6DDF7"
REPORT_ACCENT = "#9d91fb"
REPORT_TEXT_DARK = "#111111"
EXPORT_BG_STOPS = ("#0f1222", "#151b33", "#1a2440")
GRAFFITI_BG = REPORT_BG
GRAFFITI_PANEL = REPORT_PANEL
GRAFFITI_PANEL_ALT = REPORT_PANEL_ALT
GRAFFITI_BORDER = REPORT_BORDER
GRAFFITI_TAG = REPORT_TEXT
GRAFFITI_PINK = "#cf6bf3"
GRAFFITI_ORANGE = "#f59e0b"
GRAFFITI_TEAL = "#38bdf8"
GRAFFITI_YELLOW = "#ffd24a"


_FONT_WARNING_EMITTED = False


def _hex_to_rgb(value: str) -> tuple[int, int, int]:
    text = value.lstrip("#")
    return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))


def _blend_rgb(left: tuple[int, int, int], right: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return (
        int(left[0] + (right[0] - left[0]) * t),
        int(left[1] + (right[1] - left[1]) * t),
        int(left[2] + (right[2] - left[2]) * t),
    )


def _paint_vertical_gradient(image: Image.Image, stops: tuple[str, str, str]) -> None:
    top, mid, bottom = (_hex_to_rgb(color) for color in stops)
    draw = ImageDraw.Draw(image)
    height = image.height
    for y in range(height):
        if height <= 1:
            t = 0.0
        else:
            t = y / (height - 1)
        if t <= 0.55:
            local_t = 0 if t == 0 else t / 0.55
            color = _blend_rgb(top, mid, local_t)
        else:
            local_t = (t - 0.55) / 0.45
            color = _blend_rgb(mid, bottom, local_t)
        draw.line([(0, y), (image.width, y)], fill=color)


def _new_export_canvas(width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height), REPORT_BG)
    _paint_vertical_gradient(image, EXPORT_BG_STOPS)
    return image


def _font_candidates(bold: bool) -> list[Path]:
    repo_root = Path(__file__).resolve().parent.parent
    bundled_dir = repo_root / "assets" / "fonts"
    candidates: list[Path] = []

    if bundled_dir.exists():
        patterns = ["*Bold*.ttf", "*Bold*.otf"] if bold else ["*Regular*.ttf", "*Regular*.otf", "*.ttf", "*.otf"]
        for pattern in patterns:
            candidates.extend(sorted(bundled_dir.glob(pattern)))

    streamlit_media_dir = repo_root / ".venv" / "Lib" / "site-packages" / "streamlit" / "static" / "static" / "media"
    if streamlit_media_dir.exists():
        patterns = ["KaTeX_SansSerif-Bold*.ttf", "KaTeX_Main-Bold*.ttf"] if bold else ["KaTeX_SansSerif-Regular*.ttf", "KaTeX_Main-Regular*.ttf"]
        for pattern in patterns:
            candidates.extend(sorted(streamlit_media_dir.glob(pattern)))

    system_candidates = [
        Path(r"C:\Windows\Fonts\arialbd.ttf") if bold else Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\seguisb.ttf") if bold else Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf") if bold else Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf") if bold else Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"),
    ]
    candidates.extend(system_candidates)
    return [path for path in candidates if path.exists()]


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    global _FONT_WARNING_EMITTED
    for path in _font_candidates(bold):
        try:
            return ImageFont.truetype(str(path), size=size)
        except Exception:
            continue
    if not _FONT_WARNING_EMITTED:  # pragma: no cover
        warnings.warn("Poster font fallback hit ImageFont.load_default(); scalable TTF font was not found.", RuntimeWarning)
        _FONT_WARNING_EMITTED = True
    return ImageFont.load_default()


def _section_team(title: str) -> str:
    return title.split(" ", 1)[0] if title else ""


def _panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: str = REPORT_PANEL, radius: int = 18) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=REPORT_BORDER, width=2)


def _graffiti_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: str = GRAFFITI_PANEL,
    radius: int = 28,
) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=GRAFFITI_BORDER, width=3)
    draw.arc((left - 10, top - 12, left + 120, top + 64), start=200, end=350, fill=GRAFFITI_PINK, width=5)
    draw.arc((right - 150, top - 8, right + 8, top + 82), start=190, end=320, fill=GRAFFITI_TEAL, width=5)
    draw.line((left + 20, bottom - 18, left + 90, bottom - 30, left + 150, bottom - 16), fill=GRAFFITI_ORANGE, width=4)
    draw.line((right - 170, bottom - 26, right - 115, bottom - 16, right - 60, bottom - 28), fill=GRAFFITI_PINK, width=4)


def _draw_graffiti_header(
    draw: ImageDraw.ImageDraw,
    width: int,
    title: str,
    subtitle: str,
) -> int:
    header_box = (20, 20, width - 20, 190)
    _graffiti_panel(draw, header_box, fill=GRAFFITI_BG, radius=34)
    draw.rounded_rectangle((40, 38, 220, 76), radius=16, fill=GRAFFITI_TAG)
    _text(draw, (58, 46), "KASPER", _load_font(22, bold=True), "#ffffff")
    draw.rounded_rectangle((238, 46, 430, 64), radius=9, fill=GRAFFITI_PINK)
    draw.rounded_rectangle((248, 66, 478, 82), radius=9, fill=GRAFFITI_TEAL)

    hero_font = _load_font(52, bold=True)
    shadow_fill = GRAFFITI_ORANGE
    outline_fill = GRAFFITI_TAG
    _text(draw, (54, 82), "KASPER", hero_font, shadow_fill, stroke_width=7, stroke_fill=outline_fill)
    _text(draw, (48, 76), "KASPER", hero_font, "#fff8ed", stroke_width=3, stroke_fill=GRAFFITI_TAG)
    _text(draw, (48, 128), title, _load_font(32, bold=True), REPORT_TEXT)
    _text(draw, (48, 156), subtitle, _load_font(22, bold=True), REPORT_MUTED)
    return header_box[3]


def _draw_kasper_export_header(
    draw: ImageDraw.ImageDraw,
    width: int,
    title: str,
    subtitle: str,
) -> int:
    header_box = (20, 20, width - 20, 190)
    _panel(draw, header_box, fill=REPORT_PANEL, radius=30)
    draw.rounded_rectangle((40, 38, 252, 76), radius=16, fill=REPORT_ACCENT)
    _text(draw, (58, 46), "KASPER", _load_font(22, bold=True), "#ffffff")
    _text(draw, (48, 86), "KASPER SCOUTING REPORT", _load_font(44, bold=True), REPORT_TEXT)
    _text(draw, (48, 134), title, _load_font(30, bold=True), REPORT_TEXT)
    _text(draw, (48, 160), subtitle, _load_font(21, bold=True), REPORT_MUTED)
    return header_box[3]


def _text(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    value: str,
    font: ImageFont.ImageFont,
    fill: str = REPORT_TEXT,
    stroke_width: int = 0,
    stroke_fill: str = REPORT_TEXT,
) -> None:
    draw.text(position, value, fill=fill, font=font, stroke_width=stroke_width, stroke_fill=stroke_fill)


def _measure(draw: ImageDraw.ImageDraw, value: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), value, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _report_table_height(frame: pd.DataFrame, row_height: int = 24, header_height: int = 26) -> int:
    return header_height + row_height * max(len(frame), 1) + 20


def _report_summary_height(frame: pd.DataFrame) -> int:
    stats = _infer_summary_stats(frame)
    rows = max(1, (len(stats) + 1) // 2)
    return 64 + rows * 84


def _filter_section_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[[column for column in columns if column in frame.columns]].copy()


def _draw_report_header(
    draw: ImageDraw.ImageDraw,
    width: int,
    title: str,
    subtitle: str,
    away_summary: dict | None,
    home_summary: dict | None,
    title_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
) -> int:
    hero_box = (24, 24, width - 24, 166)
    _panel(draw, hero_box, fill=REPORT_PANEL, radius=24)
    _text(draw, (46, 42), "KASPER SCOUTING REPORT", body_font, REPORT_ACCENT)
    _text(draw, (46, 68), title, title_font, REPORT_TEXT)
    _text(draw, (46, 110), subtitle, body_font, REPORT_TEXT)
    starter_y = 132
    if away_summary is not None and not away_summary["frame"].empty:
        row = away_summary["frame"].iloc[0]
        _text(draw, (46, starter_y), f"Away: {_section_team(away_summary['title'])} | {row.get('pitcher_name', 'Starter')} ({row.get('p_throws', '-')})", body_font, REPORT_TEXT)
    if home_summary is not None and not home_summary["frame"].empty:
        row = home_summary["frame"].iloc[0]
        _text(draw, (width // 2 + 40, starter_y), f"Home: {_section_team(home_summary['title'])} | {row.get('pitcher_name', 'Starter')} ({row.get('p_throws', '-')})", body_font, REPORT_TEXT)
    return hero_box[3] + 18


def _draw_matchup_board(
    draw: ImageDraw.ImageDraw,
    top: int,
    left: int,
    width: int,
    frame: pd.DataFrame,
    title_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
) -> int:
    board_height = 258
    _panel(draw, (left, top, left + width, top + board_height), fill=REPORT_PANEL_ALT)
    _text(draw, (left + 18, top + 16), "Best Matchups", title_font, REPORT_TEXT)
    work = _filter_section_columns(frame.head(3), ["hitter_name", "team", "matchup_score", "iso", "xwoba", "swstr_pct", "pulled_barrel_pct", "sweet_spot_pct", "hard_hit_pct", "avg_launch_angle"])
    card_top = top + 56
    card_height = 58
    for idx, (_, row) in enumerate(work.iterrows(), start=1):
        y0 = card_top + (idx - 1) * (card_height + 10)
        _panel(draw, (left + 14, y0, left + width - 14, y0 + card_height), fill="#ffffff", radius=14)
        _text(draw, (left + 24, y0 + 14), f"{idx}. {row.get('hitter_name', '-')}", body_font, REPORT_TEXT)
        _text(draw, (left + width - 90, y0 + 20), row.get("team", "-"), body_font, REPORT_TEXT)
        chip_specs = [
            ("matchup_score", "Matchup"),
            ("xwoba", "xwOBA"),
            ("swstr_pct", "SwStr%"),
            ("pulled_barrel_pct", "PulledBrl%"),
            ("sweet_spot_pct", "SweetSpot%"),
            ("hard_hit_pct", "HH%"),
            ("avg_launch_angle", "LA"),
        ]
        start_x = left + 250
        chip_width = 110
        chip_gap = 10
        for offset, (column, label) in enumerate(chip_specs):
            chip_x = start_x + offset * (chip_width + chip_gap)
            chip_fill = _background_hex(
                column,
                row.get(column),
                work[column],
                lower_is_better=LOWER_IS_BETTER,
                higher_is_better=HIGHER_IS_BETTER,
            ) or "#e8eef5"
            draw.rounded_rectangle((chip_x, y0 + 10, chip_x + chip_width, y0 + 40), radius=12, fill=chip_fill)
            chip_text = f"{label} {_format_value(column, row.get(column), export_mode=True)}"
            _text(draw, (chip_x + 8, y0 + 18), chip_text, body_font, REPORT_TEXT)
    return top + board_height


def _draw_report_summary_cards(
    draw: ImageDraw.ImageDraw,
    top: int,
    left: int,
    width: int,
    title: str,
    frame: pd.DataFrame,
    title_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
    lower_is_better: set[str] | None = None,
    higher_is_better: set[str] | None = None,
) -> int:
    panel_height = _report_summary_height(frame)
    _panel(draw, (left, top, left + width, top + panel_height), fill=REPORT_PANEL)
    _text(draw, (left + 18, top + 14), title, title_font, REPORT_BORDER)
    if frame.empty:
        _text(draw, (left + 18, top + 50), "No summary available", body_font, REPORT_MUTED)
        return top + panel_height
    row = frame.iloc[0]
    stats = _infer_summary_stats(frame)
    cols = 2
    gap = 12
    card_width = (width - 36 - gap) // cols
    card_height = 68
    start_y = top + 48
    for idx, (label, column_name) in enumerate(stats):
        col_idx = idx % cols
        row_idx = idx // cols
        x0 = left + 18 + col_idx * (card_width + gap)
        y0 = start_y + row_idx * (card_height + 10)
        x1 = x0 + card_width
        y1 = y0 + card_height
        fill = _summary_card_color(column_name, row, lower_is_better, higher_is_better)
        draw.rounded_rectangle((x0, y0, x1, y1), radius=12, fill=fill, outline="#d0d8e0", width=1)
        _text(draw, (x0 + 10, y0 + 10), label, body_font, REPORT_TEXT)
        _text(draw, (x0 + 10, y0 + 36), _format_value(column_name, row.get(column_name), export_mode=True), body_font, REPORT_TEXT)
    return top + panel_height


def _draw_dark_table(
    draw: ImageDraw.ImageDraw,
    top: int,
    left: int,
    width: int,
    title: str,
    frame: pd.DataFrame,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
    lower_is_better: set[str] | None = None,
    higher_is_better: set[str] | None = None,
) -> int:
    subtitle = ""
    if isinstance(title, dict):  # pragma: no cover
        subtitle = str(title.get("subtitle", "")).strip()
        title = str(title.get("title", ""))
    if frame.empty:
        panel_height = 92 if subtitle else 76
        _panel(draw, (left, top, left + width, top + panel_height))
        _text(draw, (left + 16, top + 16), title, font, REPORT_BORDER)
        if subtitle:
            _text(draw, (left + 16, top + 42), subtitle, small_font, REPORT_MUTED)
            _text(draw, (left + 16, top + 62), "No data available", small_font, REPORT_MUTED)
        else:
            _text(draw, (left + 16, top + 42), "No data available", small_font, REPORT_MUTED)
        return top + panel_height

    header_height = 34
    row_height = 32
    padding_x = 12
    padding_y = 8
    title_block_height = 64 if subtitle else 46
    panel_height = title_block_height + _report_table_height(frame, row_height=row_height, header_height=header_height)
    _panel(draw, (left, top, left + width, top + panel_height), fill=REPORT_PANEL)
    _text(draw, (left + 16, top + 14), title, font, REPORT_BORDER)
    if subtitle:
        _text(draw, (left + 16, top + 40), subtitle, small_font, REPORT_MUTED)

    y = top + title_block_height
    display_frame = _display_frame(frame).copy()
    headers = list(display_frame.columns)
    source_by_label = {DISPLAY_LABELS.get(col, col): col for col in frame.columns}
    formatted_rows = [[_format_value(source_by_label.get(column, column), row[column], export_mode=True) for column in headers] for _, row in display_frame.iterrows()]

    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    col_widths: list[int] = []
    for col_idx, header in enumerate(headers):
        max_width = _measure(probe, header, small_font)[0]
        for row in formatted_rows:
            max_width = max(max_width, _measure(probe, row[col_idx], small_font)[0])
        col_widths.append(max_width + padding_x * 2)
    total_width = sum(col_widths)
    if total_width < width - 24:
        extra = (width - 24 - total_width) // max(len(col_widths), 1)
        col_widths = [size + extra for size in col_widths]

    identity_columns = set(_identity_columns_for_section(frame, title))
    x = left + 12
    for idx, header in enumerate(headers):
        draw.rounded_rectangle((x, y, x + col_widths[idx], y + header_height), radius=8, fill="#dce6f2")
        _text(draw, (x + padding_x, y + 7), header, small_font, REPORT_TEXT_DARK)
        x += col_widths[idx]

    for row_idx, row in enumerate(formatted_rows, start=1):
        current_y = y + header_height + (row_idx - 1) * row_height
        x = left + 12
        for col_idx, value in enumerate(row):
            display_label = headers[col_idx]
            source_column = source_by_label.get(display_label, display_label)
            fill = "#ffffff" if row_idx % 2 else "#f7f9fc"
            if source_column == "hr_form" and HR_FORM_PCT_COLUMN in frame.columns:
                fill = _hr_form_heatmap_hex(frame.iloc[row_idx - 1].get(HR_FORM_PCT_COLUMN)) or fill
            elif source_column in frame.columns and (source_column in PERCENT_COLUMNS or source_column in RATE_COLUMNS):
                fill = _background_hex(
                    source_column,
                    frame.iloc[row_idx - 1][source_column],
                    frame[source_column],
                    lower_is_better=lower_is_better or LOWER_IS_BETTER,
                    higher_is_better=higher_is_better or HIGHER_IS_BETTER,
                ) or fill
            draw.rounded_rectangle((x, current_y, x + col_widths[col_idx], current_y + row_height - 2), radius=6, fill=fill)
            text_color = REPORT_TEXT_DARK if source_column in identity_columns else REPORT_TEXT
            _text(draw, (x + padding_x, current_y + padding_y), value, small_font, text_color)
            x += col_widths[col_idx]
    return top + panel_height


def _identity_columns_for_section(frame: pd.DataFrame, title: str) -> list[str]:
    lowered = title.lower()
    prefix_columns = ["game"] if "game" in frame.columns else []
    if "hitters" in lowered or "best matchups" in lowered:
        columns = ["hitter_name", "team"]
    elif "arsenal" in lowered or "count usage" in lowered:
        columns = ["pitch_name"]
    elif "summary" in lowered or "pitcher" in lowered:
        columns = ["pitcher_name", "team", "p_throws"]
    else:
        columns = [frame.columns[0]] if len(frame.columns) else []
    return [column for column in [*prefix_columns, *columns] if column in frame.columns]


def _metric_columns_for_section(frame: pd.DataFrame, title: str) -> list[str]:
    lowered = title.lower()
    if "best matchups" in lowered:
        columns = ["matchup_score", "iso", "xwoba", "swstr_pct", "pulled_barrel_pct", "hard_hit_pct", "fb_pct", "avg_launch_angle"]
    elif "hitters" in lowered:
        columns = ["matchup_score", "iso", "xwoba", "pulled_barrel_pct", "barrel_bip_pct", "hard_hit_pct", "avg_launch_angle"]
    elif "arsenal" in lowered:
        columns = ["usage_pct", "swstr_pct", "hard_hit_pct", "avg_release_speed", "avg_spin_rate", "xwoba_con"]
    elif "count usage" in lowered:
        columns = ["All counts", "Early count", "Even count", "Pitcher ahead", "Pitcher behind", "Two-strike", "Pre two-strike", "Full count"]
    elif "summary" in lowered:
        columns = ["pitch_count", "bip", "iso", "xwoba", "swstr_pct", "pulled_barrel_pct", "barrel_bip_pct", "fb_pct", "hard_hit_pct", "avg_launch_angle"]
    else:
        columns = [column for column in frame.columns if column not in _identity_columns_for_section(frame, title)]
    return [column for column in columns if column in frame.columns]


def _draw_scouting_rows_section(
    draw: ImageDraw.ImageDraw,
    top: int,
    left: int,
    width: int,
    title: str,
    frame: pd.DataFrame,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
    lower_is_better: set[str] | None = None,
    higher_is_better: set[str] | None = None,
) -> int:
    if frame.empty:
        panel_height = 82
        _panel(draw, (left, top, left + width, top + panel_height), fill=REPORT_PANEL)
        _text(draw, (left + 16, top + 14), title, font, REPORT_BORDER)
        _text(draw, (left + 16, top + 48), "No data available", small_font, REPORT_TEXT)
        return top + panel_height

    identity_columns = _identity_columns_for_section(frame, title)
    metric_columns = _metric_columns_for_section(frame, title)
    row_height = 52
    panel_height = 54 + len(frame) * (row_height + 10) + 12
    _panel(draw, (left, top, left + width, top + panel_height), fill=REPORT_PANEL)
    _text(draw, (left + 16, top + 14), title, font, REPORT_BORDER)

    y = top + 52
    identity_width = max(170, min(290, int(width * 0.27)))
    chip_area_left = left + 18 + identity_width + 14
    chip_width = 92
    chip_gap = 8

    for _, row in frame.iterrows():
        row_top = y
        row_bottom = row_top + row_height
        _panel(draw, (left + 12, row_top, left + width - 12, row_bottom), fill="#ffffff", radius=14)

        primary = _format_value(identity_columns[0], row.get(identity_columns[0]), export_mode=True) if identity_columns else "-"
        secondary_parts = [
            _format_value(column, row.get(column), export_mode=True)
            for column in identity_columns[1:]
            if str(_format_value(column, row.get(column), export_mode=True)).strip() != "-"
        ]
        _text(draw, (left + 26, row_top + 10), primary, small_font, REPORT_TEXT_DARK)
        if secondary_parts:
            _text(draw, (left + 26, row_top + 28), " | ".join(secondary_parts), small_font, REPORT_TEXT_DARK)

        for idx, column in enumerate(metric_columns):
            chip_x = chip_area_left + idx * (chip_width + chip_gap)
            if chip_x + chip_width > left + width - 24:
                break
            chip_fill = _background_hex(
                column,
                row.get(column),
                frame[column],
                lower_is_better=lower_is_better or LOWER_IS_BETTER,
                higher_is_better=higher_is_better or HIGHER_IS_BETTER,
            ) or "#e8eef5"
            draw.rounded_rectangle((chip_x, row_top + 10, chip_x + chip_width, row_top + 40), radius=12, fill=chip_fill)
            chip_label = DISPLAY_LABELS.get(column, column)
            chip_text = f"{chip_label} {_format_value(column, row.get(column), export_mode=True)}"
            _text(draw, (chip_x + 8, row_top + 18), chip_text, small_font, REPORT_TEXT)
        y += row_height + 10

    return top + panel_height


def _draw_top_matchups_game_section(
    draw: ImageDraw.ImageDraw,
    top: int,
    left: int,
    width: int,
    section: dict,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> int:
    frame = section["frame"]
    title = section["title"]
    subtitle = str(section.get("subtitle", "")).strip()
    title_font = _load_font(42, bold=True)
    subtitle_font = _load_font(24, bold=True)
    hitter_font = _load_font(42, bold=True)
    team_font = _load_font(26, bold=True)
    header_font = _load_font(18, bold=True)
    value_font = _load_font(34, bold=True)
    row_height = 74
    row_gap = 6
    title_height = 88 if subtitle else 58
    metric_columns = [
        "matchup_score",
        "zone_fit_score",
        "swstr_pct",
        "pulled_barrel_pct",
        "avg_launch_angle",
        "ceiling_score",
    ]
    header_row_height = 28
    panel_height = title_height + header_row_height + 8 + len(frame) * (row_height + row_gap) + 14
    if frame.empty:
        panel_height = 106 if subtitle else 80
    _panel(draw, (left, top, left + width, top + panel_height), fill=REPORT_PANEL)
    _text(draw, (left + 18, top + 14), title, title_font, REPORT_BORDER)
    if subtitle:
        _text(draw, (left + 18, top + 50), subtitle, subtitle_font, REPORT_MUTED)
    if frame.empty:
        _text(draw, (left + 18, top + title_height), "No data available", subtitle_font, REPORT_MUTED)
        return top + panel_height

    y = top + title_height
    identity_width = max(220, min(275, int(width * 0.22)))
    chip_area_left = left + 22 + identity_width + 10
    chip_gap_x = 6
    chip_width = max(136, int((left + width - 22 - chip_area_left - chip_gap_x * (len(metric_columns) - 1) - 18) / len(metric_columns)))
    chip_height = 44
    _text(draw, (left + 24, y + 4), "Hitter", header_font, REPORT_MUTED)
    _text(draw, (left + 24, y + 18), "Team", header_font, REPORT_MUTED)
    for idx, column in enumerate(metric_columns):
        chip_x = chip_area_left + idx * (chip_width + chip_gap_x)
        header_label = DISPLAY_LABELS.get(column, column)
        _text(draw, (chip_x + 4, y + 10), header_label, header_font, REPORT_MUTED)
    y += header_row_height + 8

    for _, row in frame.iterrows():
        row_top = y
        row_bottom = row_top + row_height
        _panel(draw, (left + 12, row_top, left + width - 12, row_bottom), fill="#fbfcfe", radius=14)
        primary = _format_value("hitter_name", row.get("hitter_name"), export_mode=True)
        team_value = _format_value("team", row.get("team"), export_mode=True)
        _text(draw, (left + 24, row_top + 6), primary, hitter_font, REPORT_TEXT_DARK)
        _text(draw, (left + 24, row_top + 40), team_value, team_font, REPORT_TEXT_DARK)

        for idx, column in enumerate(metric_columns):
            if column not in frame.columns:
                continue
            chip_x = chip_area_left + idx * (chip_width + chip_gap_x)
            chip_y = row_top + 15
            chip_fill = _background_hex(
                column,
                row.get(column),
                frame[column],
                lower_is_better=LOWER_IS_BETTER,
                higher_is_better=HIGHER_IS_BETTER,
            ) or "#e8eef5"
            draw.rounded_rectangle((chip_x, chip_y, chip_x + chip_width, chip_y + chip_height), radius=10, fill=chip_fill)
            chip_text = _format_value(column, row.get(column), export_mode=True)
            _text(draw, (chip_x + 10, chip_y + 7), chip_text, value_font, REPORT_TEXT)
        y += row_height + row_gap

    return top + panel_height


def _build_top_matchups_report_image(title: str, subtitle: str, sections: list[dict]) -> bytes:
    if not HAS_PILLOW:
        raise RuntimeError("Pillow is required for PNG/JPG export.")
    section_title_font = _load_font(36, bold=True)
    section_body_font = _load_font(24, bold=True)
    width = 1480
    header_height = 190
    total_height = header_height + 10
    for section in sections:
        total_height += _export_section_height_estimate(section["title"], section["frame"], str(section.get("subtitle", "")))

    image = _new_export_canvas(width, total_height)
    draw = ImageDraw.Draw(image)
    _draw_kasper_export_header(draw, width, title, subtitle)

    y = header_height + 8
    for section in sections:
        if section["frame"].empty:
            continue
        y = _draw_export_section(draw, y, 12, width - 24, section, section_title_font, section_body_font) + 8

    cropped = image.crop((0, 0, width, min(max(y + 18, 380), image.height)))
    buffer = BytesIO()
    cropped.save(buffer, format="PNG")
    return buffer.getvalue()


def _slate_group_height(section: dict) -> int:
    inner_sections = section.get("sections", [])
    if not inner_sections:
        return 0
    total = 98
    for inner in inner_sections:
        total += _export_section_height_estimate(inner["title"], inner["frame"], str(inner.get("subtitle", ""))) + 4
    return total + 10


def _draw_slate_summary_group(
    draw: ImageDraw.ImageDraw,
    top: int,
    left: int,
    width: int,
    section: dict,
    title_font: ImageFont.ImageFont,
    body_font: ImageFont.ImageFont,
) -> int:
    inner_sections = [inner for inner in section.get("sections", []) if not inner.get("frame", pd.DataFrame()).empty]
    if not inner_sections:
        return top

    panel_height = _slate_group_height({**section, "sections": inner_sections})
    _panel(draw, (left, top, left + width, top + panel_height), fill=REPORT_PANEL_ALT, radius=30)
    chip_width = min(480, width - 56)
    draw.rounded_rectangle((left + 24, top + 16, left + 24 + chip_width, top + 62), radius=18, fill=REPORT_ACCENT)
    draw.rounded_rectangle((left + 34, top + 58, left + 210, top + 72), radius=7, fill="#dbe6f3")
    _text(draw, (left + 42, top + 25), section["title"], _load_font(28, bold=True), "#ffffff")
    if section.get("subtitle"):
        _text(draw, (left + 28, top + 74), str(section["subtitle"]), _load_font(20, bold=True), REPORT_TEXT)

    y = top + 96
    for inner in inner_sections:
        y = _draw_export_section(draw, y, left + 16, width - 32, inner, title_font, body_font) + 4
    return top + panel_height


def _draw_top_slate_table_section(
    draw: ImageDraw.ImageDraw,
    top: int,
    left: int,
    width: int,
    section: dict,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> int:
    frame = section["frame"]
    title = section["title"]
    subtitle = str(section.get("subtitle", "")).strip()
    title_font = _load_font(28, bold=True)
    subtitle_font = _load_font(19, bold=True)
    cell_font = _load_font(16, bold=True)
    header_font = _load_font(15, bold=True)
    header_height = 40
    row_height = 44
    padding_x = 12
    padding_y = 10
    title_block_height = 74 if subtitle else 54
    panel_height = title_block_height + header_height + row_height * max(len(frame), 1) + 24
    _panel(draw, (left, top, left + width, top + panel_height), fill=REPORT_PANEL, radius=24)
    chip_width = min(360, width - 52)
    draw.rounded_rectangle((left + 22, top + 14, left + 22 + chip_width, top + 52), radius=14, fill=REPORT_ACCENT)
    _text(draw, (left + 36, top + 21), title, title_font, "#ffffff")
    if subtitle:
        _text(draw, (left + 28, top + 54), subtitle, subtitle_font, REPORT_MUTED)

    if frame.empty:
        _text(draw, (left + 28, top + title_block_height), "No data available", subtitle_font, REPORT_MUTED)
        return top + panel_height

    y = top + title_block_height
    display_frame = _display_frame(frame).copy()
    headers = list(display_frame.columns)
    source_by_label = {DISPLAY_LABELS.get(col, col): col for col in frame.columns}
    formatted_rows = [[_format_value(source_by_label.get(column, column), row[column], export_mode=True) for column in headers] for _, row in display_frame.iterrows()]

    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    col_widths: list[int] = []
    for col_idx, header in enumerate(headers):
        max_width = _measure(probe, header, header_font)[0]
        for row in formatted_rows:
            max_width = max(max_width, _measure(probe, row[col_idx], cell_font)[0])
        col_widths.append(max_width + padding_x * 2)
    total_width = sum(col_widths)
    if total_width < width - 28:
        extra = (width - 28 - total_width) // max(len(col_widths), 1)
        col_widths = [size + extra for size in col_widths]

    identity_columns = set(_identity_columns_for_section(frame, title))
    x = left + 14
    for idx, header in enumerate(headers):
        draw.rounded_rectangle((x, y, x + col_widths[idx], y + header_height), radius=10, fill="#dbe6f3")
        _text(draw, (x + padding_x, y + 9), header, header_font, REPORT_TEXT_DARK)
        x += col_widths[idx]

    for row_idx, row in enumerate(formatted_rows, start=1):
        current_y = y + header_height + (row_idx - 1) * row_height
        x = left + 14
        for col_idx, value in enumerate(row):
            display_label = headers[col_idx]
            source_column = source_by_label.get(display_label, display_label)
            fill = "#ffffff" if row_idx % 2 else "#f7f9fc"
            if source_column == "hr_form" and HR_FORM_PCT_COLUMN in frame.columns:
                fill = _hr_form_heatmap_hex(frame.iloc[row_idx - 1].get(HR_FORM_PCT_COLUMN)) or fill
            elif source_column in frame.columns and (source_column in PERCENT_COLUMNS or source_column in RATE_COLUMNS):
                fill = _background_hex(
                    source_column,
                    frame.iloc[row_idx - 1][source_column],
                    frame[source_column],
                    lower_is_better=section.get("lower_is_better") or LOWER_IS_BETTER,
                    higher_is_better=section.get("higher_is_better") or HIGHER_IS_BETTER,
                ) or fill
            draw.rounded_rectangle((x, current_y, x + col_widths[col_idx], current_y + row_height - 3), radius=8, fill=fill)
            text_color = REPORT_TEXT_DARK if source_column in identity_columns else REPORT_TEXT
            _text(draw, (x + padding_x, current_y + padding_y), value, cell_font, text_color)
            x += col_widths[col_idx]
    return top + panel_height


def _build_slate_summary_report_image(title: str, subtitle: str, sections: list[dict]) -> bytes:
    if not HAS_PILLOW:
        raise RuntimeError("Pillow is required for PNG/JPG export.")
    section_title_font = _load_font(34, bold=True)
    section_body_font = _load_font(24, bold=True)
    width = 1520
    header_height = 190
    total_height = header_height + 10
    for section in sections:
        if section.get("section_type") == "slate_summary_group":
            total_height += _slate_group_height(section) + 8
        else:
            total_height += _export_section_height_estimate(section["title"], section["frame"], str(section.get("subtitle", ""))) + 8

    image = _new_export_canvas(width, total_height)
    draw = ImageDraw.Draw(image)
    _draw_kasper_export_header(draw, width, title, subtitle)

    y = header_height + 8
    for section in sections:
        if section.get("section_type") == "slate_summary_group":
            y = _draw_slate_summary_group(draw, y, 12, width - 24, section, section_title_font, section_body_font) + 6
            continue
        if section["frame"].empty:
            continue
        y = _draw_export_section(draw, y, 12, width - 24, section, section_title_font, section_body_font) + 8

    cropped = image.crop((0, 0, width, min(max(y + 18, 420), image.height)))
    buffer = BytesIO()
    cropped.save(buffer, format="PNG")
    return buffer.getvalue()


def _draw_export_section(
    draw: ImageDraw.ImageDraw,
    top: int,
    left: int,
    width: int,
    section: dict,
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> int:
    if section.get("section_type") == "top_matchups_game":
        return _draw_top_matchups_game_section(draw, top, left, width, section, font, small_font)
    if section.get("section_type") in {"top_slate_hitters", "top_slate_pitchers"}:
        return _draw_top_slate_table_section(draw, top, left, width, section, font, small_font)
    title = section["title"]
    frame = section["frame"]
    return _draw_dark_table(
        draw,
        top,
        left,
        width,
        {"title": title, "subtitle": section.get("subtitle", "")},
        frame,
        font,
        small_font,
        section.get("lower_is_better"),
        section.get("higher_is_better"),
    )


def _draw_report_stacked_sections(
    draw: ImageDraw.ImageDraw,
    top: int,
    width: int,
    sections: list[dict],
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> int:
    y = top
    for section in sections:
        y = _draw_export_section(draw, y, 24, width - 48, section, font, small_font) + 12
    return y


def _draw_report_two_column_sections(
    draw: ImageDraw.ImageDraw,
    top: int,
    width: int,
    left_sections: list[dict],
    right_sections: list[dict],
    font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> int:
    gap = 22
    col_width = (width - 48 - gap) // 2
    left_x = 24
    right_x = left_x + col_width + gap
    left_y = top
    right_y = top
    for section in left_sections:
        left_y = _draw_export_section(draw, left_y, left_x, col_width, section, font, small_font) + 12
    for section in right_sections:
        right_y = _draw_export_section(draw, right_y, right_x, col_width, section, font, small_font) + 12
    return max(left_y, right_y)


def _collect_team_sections(sections: list[dict], kind: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for section in _extract_sections(sections, kind):
        grouped.setdefault(_section_team(section["title"]), []).append(section)
    return grouped


def _exclude_all_split_sections(sections: list[dict]) -> list[dict]:
    filtered: list[dict] = []
    for section in sections:
        lowered = section["title"].lower()
        if "summary all" in lowered or "arsenal all" in lowered or "count usage all" in lowered:
            continue
        filtered.append(section)
    return filtered


def _split_sections_by_hand(sections: list[dict]) -> tuple[list[dict], list[dict]]:
    vs_rhh = [section for section in sections if "vs rhh" in section["title"].lower()]
    vs_lhh = [section for section in sections if "vs lhh" in section["title"].lower()]
    return vs_rhh, vs_lhh


def _build_compact_poster_image(title: str, subtitle: str, sections: list[dict]) -> bytes:
    if not HAS_PILLOW:
        raise RuntimeError("Pillow is required for PNG/JPG export.")
    title_font = _load_font(56, bold=True)
    section_font = _load_font(34, bold=True)
    body_font = _load_font(28, bold=True)
    small_font = _load_font(24, bold=True)
    hitter_title_font = _load_font(38, bold=True)
    hitter_body_font = _load_font(28, bold=True)
    width = 1600
    image = _new_export_canvas(width, 4200)
    draw = ImageDraw.Draw(image)

    best_section = next((section for section in sections if _section_type(section["title"]) == "best"), None)
    summary_sections = _extract_sections(sections, "summary")
    arsenal_by_team = _collect_team_sections(_exclude_all_split_sections(sections), "arsenal")
    hitter_sections = _extract_sections(sections, "hitters")
    away_summary = next((section for section in summary_sections if "summary all" in section["title"].lower()), None)
    home_summary = next((section for section in summary_sections if section is not away_summary and "summary all" in section["title"].lower()), None)

    y = _draw_report_header(draw, width, title, subtitle, away_summary, home_summary, title_font, body_font)

    if best_section is not None:
        best_frame = _filter_section_columns(best_section["frame"], ["hitter_name", "team", "matchup_score", "iso", "xwoba", "swstr_pct", "pulled_barrel_pct", "hard_hit_pct", "fb_pct", "avg_launch_angle"])
        y = _draw_export_section(
            draw,
            y,
            24,
            width - 48,
            {
                **best_section,
                "title": "Best Matchups",
                "frame": best_frame,
            },
            section_font,
            body_font,
        ) + 18

    team_order = list(arsenal_by_team.keys())
    for team in team_order:
        team_sections = arsenal_by_team.get(team, [])
        if not team_sections:
            continue
        _text(draw, (24, y), f"{team} Pitcher", title_font, REPORT_BORDER)
        y += 42
        vs_rhh, vs_lhh = _split_sections_by_hand(team_sections)
        y = _draw_report_two_column_sections(draw, y, width, vs_rhh[:1], vs_lhh[:1], section_font, small_font) + 12

    if len(hitter_sections) >= 2:
        for section in hitter_sections[:2]:
            frame = _filter_section_columns(section["frame"], ["hitter_name", "matchup_score", "iso", "xwoba", "pulled_barrel_pct", "barrel_bip_pct", "hard_hit_pct", "avg_launch_angle"])
            _text(draw, (24, y), section["title"], title_font, REPORT_BORDER)
            y += 42
            y = _draw_export_section(
                draw,
                y,
                24,
                width - 48,
                {**section, "frame": frame},
                hitter_title_font,
                hitter_body_font,
            ) + 14

    cropped = image.crop((0, 0, width, min(max(y + 30, 1600), image.height)))
    buffer = BytesIO()
    cropped.save(buffer, format="PNG")
    return buffer.getvalue()


def _build_carousel_images(title: str, subtitle: str, sections: list[dict]) -> list[bytes]:
    slides: list[bytes] = []
    best_section = next((section for section in sections if _section_type(section["title"]) == "best"), None)
    summary_sections = _extract_sections(sections, "summary")
    arsenal_sections = _exclude_all_split_sections(_extract_sections(sections, "arsenal"))
    count_sections = _exclude_all_split_sections(_extract_sections(sections, "count"))
    hitter_sections = _extract_sections(sections, "hitters")

    overview_sections: list[dict] = []
    if best_section is not None:
        overview_sections.append(
            {
                "title": "Best Matchups",
                "frame": _filter_section_columns(best_section["frame"], ["hitter_name", "team", "matchup_score", "iso", "xwoba", "swstr_pct", "pulled_barrel_pct", "hard_hit_pct", "fb_pct", "avg_launch_angle"]),
            }
        )
    overview_sections.extend([section for section in summary_sections if "summary all" in section["title"].lower()])
    slides.append(_build_compact_poster_image(title, f"{subtitle} | Slide 1: Overview", overview_sections))

    pitching_sections: list[dict] = []
    pitching_sections.extend(summary_sections)
    pitching_sections.extend(
        {
            **section,
            "frame": _filter_section_columns(section["frame"], ["pitch_name", "usage_pct", "swstr_pct", "hard_hit_pct", "avg_release_speed", "xwoba_con"]),
        }
        for section in arsenal_sections
    )
    slides.append(build_branded_report_image(title, f"{subtitle} | Slide 2: Pitching Detail", pitching_sections))

    if len(hitter_sections) >= 2:
        hitter_slide_sections = [
            {"title": hitter_sections[0]["title"], "frame": _filter_section_columns(hitter_sections[0]["frame"], ["hitter_name", "matchup_score", "iso", "xwoba", "pulled_barrel_pct", "barrel_bip_pct", "hard_hit_pct", "avg_launch_angle"])},
            {"title": hitter_sections[1]["title"], "frame": _filter_section_columns(hitter_sections[1]["frame"], ["hitter_name", "matchup_score", "iso", "xwoba", "pulled_barrel_pct", "barrel_bip_pct", "hard_hit_pct", "avg_launch_angle"])},
        ]
        slides.append(build_branded_report_image(title, f"{subtitle} | Slide 3: Hitter Matchups", hitter_slide_sections))

    if count_sections:
        count_slide_sections = [
            {
                **section,
                "frame": _filter_section_columns(section["frame"], ["pitch_name", "All counts", "Early count", "Even count", "Pitcher ahead", "Pitcher behind", "Two-strike", "Pre two-strike", "Full count"]),
            }
            for section in count_sections
        ]
        slides.append(build_branded_report_image(title, f"{subtitle} | Slide 4: Count Usage", count_slide_sections))

    return slides


def _draw_section(
    draw: ImageDraw.ImageDraw,
    top: int,
    left: int,
    width: int,
    title: str,
    frame: pd.DataFrame,
    lower_is_better: set[str] | None,
    higher_is_better: set[str] | None,
    font: ImageFont.ImageFont,
) -> int:
    padding_x = 12
    padding_y = 8
    row_height = 26
    small_font = _load_font(13)
    panel_height = 46 + _report_table_height(frame, row_height=row_height, header_height=26)
    _panel(draw, (left + 8, top, left + width - 8, top + panel_height), fill=REPORT_PANEL)
    _text(draw, (left + 22, top + 14), title, font, REPORT_BORDER)
    y = top + 46

    display_frame = _display_frame(frame).copy()
    headers = list(display_frame.columns)
    formatted_rows = []
    source_by_label = {DISPLAY_LABELS.get(col, col): col for col in frame.columns}
    for _, row in display_frame.iterrows():
        formatted_rows.append([_format_value(source_by_label.get(column, column), row[column], export_mode=True) for column in headers])

    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    col_widths = []
    for col_idx, header in enumerate(headers):
        bbox = probe.textbbox((0, 0), header, font=small_font)
        max_width = bbox[2] - bbox[0]
        for row in formatted_rows:
            bbox = probe.textbbox((0, 0), row[col_idx], font=small_font)
            max_width = max(max_width, bbox[2] - bbox[0])
        col_widths.append(max_width + padding_x * 2)

    usable_width = width - 32
    total_width = sum(col_widths)
    if total_width < usable_width:
        extra = (usable_width - total_width) // max(len(col_widths), 1)
        col_widths = [size + extra for size in col_widths]

    identity_columns = set(_identity_columns_for_section(frame, title))
    x = left + 16
    for idx, header in enumerate(headers):
        draw.rounded_rectangle((x, y, x + col_widths[idx], y + row_height), radius=8, fill="#dce6f2")
        _text(draw, (x + padding_x, y + padding_y), header, small_font, REPORT_TEXT_DARK)
        x += col_widths[idx]

    for row_idx, row in enumerate(formatted_rows, start=1):
        current_y = y + row_height * row_idx
        x = left + 16
        for col_idx, value in enumerate(row):
            display_label = headers[col_idx]
            source_column = source_by_label.get(display_label, display_label)
            fill = "#ffffff" if row_idx % 2 else "#f7f9fc"
            if source_column == "hr_form" and HR_FORM_PCT_COLUMN in frame.columns:
                fill = _hr_form_heatmap_hex(frame.iloc[row_idx - 1].get(HR_FORM_PCT_COLUMN)) or fill
            elif source_column in frame.columns and (source_column in PERCENT_COLUMNS or source_column in RATE_COLUMNS):
                fill = _background_hex(
                    source_column,
                    frame.iloc[row_idx - 1][source_column],
                    frame[source_column],
                    lower_is_better=lower_is_better or LOWER_IS_BETTER,
                    higher_is_better=higher_is_better or HIGHER_IS_BETTER,
                ) or fill
            draw.rounded_rectangle((x, current_y, x + col_widths[col_idx], current_y + row_height - 2), radius=6, fill=fill)
            text_color = REPORT_TEXT_DARK if source_column in identity_columns else REPORT_TEXT
            _text(draw, (x + padding_x, current_y + padding_y), value, small_font, text_color)
            x += col_widths[col_idx]

    return top + panel_height + 12


def build_branded_report_image(title: str, subtitle: str, sections: list[dict]) -> bytes:
    if not HAS_PILLOW:
        raise RuntimeError("Pillow is required for PNG/JPG export.")
    font = _load_font(36, bold=True)
    body_font = _load_font(26, bold=True)
    width = 1500
    branding_height = 100
    total_height = branding_height + 24
    for section in sections:
        total_height += _export_section_height_estimate(section["title"], section["frame"], str(section.get("subtitle", "")))
    image = _new_export_canvas(width, total_height)
    draw = ImageDraw.Draw(image)
    _panel(draw, (20, 20, width - 20, branding_height), fill=REPORT_PANEL, radius=24)
    _text(draw, (42, 34), "KASPER SCOUTING REPORT", body_font, REPORT_ACCENT)
    _text(draw, (42, 58), title, font, REPORT_TEXT)
    _text(draw, (42, 82), subtitle, body_font, REPORT_TEXT)

    y = branding_height + 14
    for section in sections:
        if section["frame"].empty:
            continue
        y = _draw_export_section(draw, y, 12, width - 24, section, font, body_font) + 8

    cropped = image.crop((0, 0, width, min(max(y + 20, 420), image.height)))
    buffer = BytesIO()
    cropped.save(buffer, format="PNG")
    return buffer.getvalue()


def _build_export_bundle(title: str, subtitle: str, sections: list[dict], layout_mode: str) -> list[bytes]:
    if layout_mode == "Compact single poster":
        return [_build_compact_poster_image(title, subtitle, sections)]
    if layout_mode == "Carousel":
        return _build_carousel_images(title, subtitle, sections)
    return [build_branded_report_image(title, subtitle, sections)]


def png_to_jpg_bytes(png_bytes: bytes) -> bytes:
    if not HAS_PILLOW:
        raise RuntimeError("Pillow is required for PNG/JPG export.")
    source = Image.open(BytesIO(png_bytes)).convert("RGB")
    buffer = BytesIO()
    source.save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def _zip_export_slides(slides: list[bytes], base_filename: str, extension: str) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for idx, slide_bytes in enumerate(slides, start=1):
            archive.writestr(f"{base_filename}-{idx}.{extension}", slide_bytes)
    return buffer.getvalue()


def _zip_full_slate_posters(bundles: list[dict], layout_mode: str) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as archive:
        for bundle in bundles:
            png_slides = _build_export_bundle(
                title=bundle["title"],
                subtitle=bundle["subtitle"],
                sections=bundle["sections"],
                layout_mode=layout_mode,
            )
            base_name = str(bundle.get("safe_name") or bundle["title"]).replace(" ", "_").replace("/", "_")
            for idx, slide_bytes in enumerate(png_slides, start=1):
                suffix = f"-{idx}" if len(png_slides) > 1 else ""
                archive.writestr(f"{base_name}{suffix}.png", slide_bytes)
    return buffer.getvalue()


def render_export_hub(key: str, title: str, export_options: dict[str, list[dict]]) -> None:
    if not export_options:
        return
    st.markdown("#### Game Exports")
    if not HAS_PILLOW:
        st.caption("Install `Pillow` to enable PNG/JPG export.")
        return
    option_labels = list(export_options.keys())
    selected_option = st.selectbox("Export scope", option_labels, key=f"{key}-scope", label_visibility="collapsed")
    layout_options = ["Compact single poster", "Carousel"] if selected_option == "Full game" else ["Standard report"]
    selected_layout = st.selectbox("Export layout", layout_options, key=f"{key}-layout", label_visibility="collapsed")
    sections = export_options[selected_option]
    prep_key = f"{key}-prepared"
    current_signature = (selected_option, selected_layout)
    if st.session_state.get(f"{prep_key}-signature") != current_signature:
        st.session_state[prep_key] = False
        st.session_state[f"{prep_key}-signature"] = current_signature

    if not st.session_state.get(prep_key, False):
        st.button("Prepare exports", key=f"{key}-prepare", use_container_width=True, on_click=lambda: st.session_state.__setitem__(prep_key, True))
        st.caption("Exports are generated on demand to keep the hosted app fast and stable.")
        return

    subtitle = f"{selected_option} | Generated from the Kasper matchup dashboard"
    png_slides = _build_export_bundle(title=title, subtitle=subtitle, sections=sections, layout_mode=selected_layout)
    jpg_slides = [png_to_jpg_bytes(slide) for slide in png_slides]
    safe_name = f"{key}-{selected_option.lower().replace(' ', '_').replace('/', '_')}-{selected_layout.lower().replace(' ', '_')}"
    col1, col2 = st.columns(2)
    with col1:
        if len(png_slides) == 1:
            st.download_button(
                label="Download PNG",
                data=png_slides[0],
                file_name=f"{safe_name}.png",
                mime="image/png",
                use_container_width=True,
                key=f"{key}-png",
            )
        else:
            st.download_button(
                label="Download PNG Carousel (.zip)",
                data=_zip_export_slides(png_slides, safe_name, "png"),
                file_name=f"{safe_name}.zip",
                mime="application/zip",
                use_container_width=True,
                key=f"{key}-png-zip",
            )
    with col2:
        if len(jpg_slides) == 1:
            st.download_button(
                label="Download JPG",
                data=jpg_slides[0],
                file_name=f"{safe_name}.jpg",
                mime="image/jpeg",
                use_container_width=True,
                key=f"{key}-jpg",
            )
        else:
            st.download_button(
                label="Download JPG Carousel (.zip)",
                data=_zip_export_slides(jpg_slides, safe_name, "jpg"),
                file_name=f"{safe_name}-jpg.zip",
                mime="application/zip",
                use_container_width=True,
                key=f"{key}-jpg-zip",
            )


def render_selected_game_twitter_export(key: str, game: dict, away_hitters: pd.DataFrame, home_hitters: pd.DataFrame) -> None:
    st.markdown("#### Twitter Card")
    if not HAS_TWITTER_EXPORT_PILLOW:
        st.caption("Install `Pillow` to enable Twitter card export.")
        return
    hitters = pd.concat([away_hitters, home_hitters], ignore_index=True, sort=False)
    if hitters.empty:
        st.info("No hitter rows are available for this Twitter card.")
        return
    try:
        png_bytes = build_twitter_game_card(game, hitters)
    except Exception as exc:
        st.warning(f"Twitter card could not be generated for this game: {exc}")
        return
    away = str(game.get("away_team", "away")).replace("/", "_").replace(" ", "_")
    home = str(game.get("home_team", "home")).replace("/", "_").replace(" ", "_")
    game_pk = str(game.get("game_pk", "game"))
    st.download_button(
        label="Download Twitter Card PNG",
        data=png_bytes,
        file_name=f"{game_pk}_{away}_at_{home}_twitter_card.png",
        mime="image/png",
        use_container_width=True,
        key=f"{key}-twitter-card-png",
    )


def render_slate_export_hub(key: str, title: str, sections: list[dict], heading: str = "Export Top Matchups") -> None:
    if not sections:
        return
    st.markdown(f"#### {heading}")
    if not HAS_PILLOW:
        st.caption("Install `Pillow` to enable PNG/JPG export.")
        return
    prep_key = f"{key}-prepared"
    if not st.session_state.get(prep_key, False):
        st.button("Prepare exports", key=f"{key}-prepare", use_container_width=True, on_click=lambda: st.session_state.__setitem__(prep_key, True))
        st.caption("Exports are generated on demand to keep the hosted app fast and stable.")
        return

    subtitle = "Top matchup hitters by game | Generated from the Kasper matchup dashboard"
    png_bytes = _build_top_matchups_report_image(title=title, subtitle=subtitle, sections=sections)
    jpg_bytes = png_to_jpg_bytes(png_bytes)
    safe_name = f"{key}-top_matchups"
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download PNG",
            data=png_bytes,
            file_name=f"{safe_name}.png",
            mime="image/png",
            use_container_width=True,
            key=f"{key}-png",
        )
    with col2:
        st.download_button(
            label="Download JPG",
            data=jpg_bytes,
            file_name=f"{safe_name}.jpg",
            mime="image/jpeg",
            use_container_width=True,
            key=f"{key}-jpg",
        )


def render_slate_export_controls(
    key: str,
    title: str,
    export_options: dict[str, list[dict]],
    full_slate_bundles: list[dict] | None = None,
    heading: str = "Export Top Matchups",
) -> None:
    has_sections = any(bool(option_sections) for option_sections in export_options.values())
    has_full_slate = bool(full_slate_bundles)
    if not has_sections and not has_full_slate:
        return

    st.markdown(f"#### {heading}")
    option_labels = [label for label, option_sections in export_options.items() if option_sections]
    selected_option = st.selectbox("Slate export scope", option_labels, key=f"{key}-scope", label_visibility="collapsed") if option_labels else None
    sections = export_options.get(selected_option, []) if selected_option else []
    zip_bytes = None
    safe_name = f"{key}-full_slate_compact_posters.zip"
    if has_full_slate:
        zip_bytes = _zip_full_slate_posters(full_slate_bundles, "Compact single poster")

    if not HAS_PILLOW:
        if has_full_slate:
            st.download_button(
                label="Export Full Slate",
                data=zip_bytes,
                file_name=safe_name,
                mime="application/zip",
                use_container_width=True,
                key=f"{key}-zip",
            )
        st.caption("Install `Pillow` to enable PNG/JPG export.")
        return

    prep_key = f"{key}-prepared"
    if not has_sections:
        st.download_button(
            label="Export Full Slate",
            data=zip_bytes,
            file_name=safe_name,
            mime="application/zip",
            use_container_width=True,
            key=f"{key}-zip",
        )
        return

    if not st.session_state.get(prep_key, False):
        if has_full_slate:
            col1, col2 = st.columns(2)
            with col1:
                st.button(
                    "Prepare exports",
                    key=f"{key}-prepare",
                    use_container_width=True,
                    on_click=lambda: st.session_state.__setitem__(prep_key, True),
                )
            with col2:
                st.download_button(
                    label="Export Full Slate",
                    data=zip_bytes,
                    file_name=safe_name,
                    mime="application/zip",
                    use_container_width=True,
                    key=f"{key}-zip",
                )
        else:
            st.button(
                "Prepare exports",
                key=f"{key}-prepare",
                use_container_width=True,
                on_click=lambda: st.session_state.__setitem__(prep_key, True),
            )
        st.caption("Exports are generated on demand to keep the hosted app fast and stable.")
        return

    subtitle = f"{selected_option or 'Top Slate Hitters'} | Generated from the Kasper matchup dashboard"
    if selected_option == "Both":
        png_bytes = _build_slate_summary_report_image(title=title, subtitle=subtitle, sections=sections)
    else:
        png_bytes = _build_top_matchups_report_image(title=title, subtitle=subtitle, sections=sections)
    jpg_bytes = png_to_jpg_bytes(png_bytes)
    image_safe_name = f"{key}-{(selected_option or 'top_slate_hitters').lower().replace(' ', '_')}"
    columns = st.columns(3 if has_full_slate else 2)
    with columns[0]:
        st.download_button(
            label="Download PNG",
            data=png_bytes,
            file_name=f"{image_safe_name}.png",
            mime="image/png",
            use_container_width=True,
            key=f"{key}-png",
        )
    with columns[1]:
        st.download_button(
            label="Download JPG",
            data=jpg_bytes,
            file_name=f"{image_safe_name}.jpg",
            mime="image/jpeg",
            use_container_width=True,
            key=f"{key}-jpg",
        )
    if has_full_slate:
        with columns[2]:
            st.download_button(
                label="Export Full Slate",
                data=zip_bytes,
                file_name=safe_name,
                mime="application/zip",
                use_container_width=True,
                key=f"{key}-zip",
            )
