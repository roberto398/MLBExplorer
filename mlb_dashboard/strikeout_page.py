"""Strikeouts & Walks projection page for Kasper.

Projects K and BB totals for each day's starting pitchers using a transparent
pitch-count → batters-faced → K/BB chain, enriched with pitch-mix + zone-aware
matchup breakdowns and per-start trend charts.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st

from .branding import apply_branding_head, page_icon_path
from .config import AppConfig
from .dashboard_views import latest_built_date
from .query_engine import StatcastQueryEngine, QueryFilters
from .strikeout_projections import build_slate_projections
from .team_logos import team_logo_data_uri, team_logo_img_html
from .ui_components import render_metric_grid, render_sticky_logo_game_nav

try:
    import altair as alt

    HAS_ALTAIR = True
except ImportError:  # pragma: no cover
    alt = None
    HAS_ALTAIR = False


# --------------------------------------------------------------------------- #
# Auth                                                                          #
# --------------------------------------------------------------------------- #

def _require_admin_password() -> bool:
    required = st.secrets.get("ADMIN_PASSWORD")
    if not required:
        st.error("Strikeouts is locked. Add ADMIN_PASSWORD to Streamlit secrets to enable this page.")
        return False
    entry = st.text_input("Strikeouts page password", type="password")
    if not entry:
        st.info("Enter the Strikeouts password to view this page.")
        return False
    if str(entry) != str(required):
        st.error("Incorrect password.")
        return False
    return True


# --------------------------------------------------------------------------- #
# Data loading                                                                  #
# --------------------------------------------------------------------------- #

def _hosted_base_url() -> str:
    import os
    return os.getenv("MLB_HOSTED_BASE_URL", "").rstrip("/")


def _default_date(config: AppConfig) -> date:
    latest = latest_built_date(config.daily_dir)
    return latest or date.today()


@st.cache_data(show_spinner=False)
def _load_remote_slate(base_url: str, target_date: date) -> pd.DataFrame:
    return pd.read_parquet(f"{base_url}/daily/{target_date.isoformat()}/slate.parquet")


def _load_slate(config: AppConfig, target_date: date) -> tuple[list[dict], str, date]:
    local_path = config.daily_dir / target_date.isoformat() / "slate.json"
    if local_path.exists():
        engine = StatcastQueryEngine(config)
        return engine.load_daily_slate(target_date), "local", target_date
    base_url = _hosted_base_url()
    if not base_url:
        return [], "none", target_date
    last_error: Exception | None = None
    for offset in range(8):
        candidate = target_date - timedelta(days=offset)
        try:
            df = _load_remote_slate(base_url, candidate)
            return df.to_dict("records"), "hosted", candidate
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return [], "none", target_date


@st.cache_data(show_spinner=False)
def _load_remote_daily_parquet(base_url: str, target_date: date, filename: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(f"{base_url}/daily/{target_date.isoformat()}/{filename}")
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _load_all_data(
    target_date: date,
    pitcher_ids: tuple[int, ...],
    team_list: tuple[str, ...],
    split: str,
    recent_window: str,
    weighted_mode: str,
) -> dict:
    """Load all data needed for projections; cached by date + pitcher IDs."""
    config = AppConfig()
    base_url = _hosted_base_url()
    is_local = config.db_path.exists()

    if is_local:
        engine = StatcastQueryEngine(config)
        filters = QueryFilters(split=split, recent_window=recent_window, weighted_mode=weighted_mode)
        pitcher_metrics = engine.get_pitcher_cards(list(pitcher_ids), filters) if pitcher_ids else pd.DataFrame()
        outcomes = engine.load_pitcher_start_outcomes_for_projections(target_date, list(pitcher_ids), n_starts=30)
        if outcomes.empty:
            outcomes = engine.load_pitcher_game_outcomes_for_projections(list(pitcher_ids), n_starts=30)
        pitcher_fzc = engine.load_daily_pitcher_family_zone_context(target_date)
        batter_fzp = engine.load_daily_batter_family_zone_profiles(target_date)
        hitters_by_team: dict[str, list] = {}
        for team in team_list:
            pool = engine.get_team_hitter_pool(team, None, filters)
            hitters_by_team[team] = pool.to_dict("records") if not pool.empty else []
    else:
        # Hosted: load everything from remote parquet files; no DuckDB available.
        # Pitcher start outcomes are loaded from the compact daily Strikeouts artifact when published.
        raw_pitchers = _load_remote_daily_parquet(base_url, target_date, "top_slate_pitchers.parquet")
        if not raw_pitchers.empty and "split_key" in raw_pitchers.columns:
            raw_pitchers = raw_pitchers.loc[
                raw_pitchers["split_key"].eq(split)
                & raw_pitchers.get("recent_window", pd.Series(recent_window, index=raw_pitchers.index)).eq(recent_window)
                & raw_pitchers.get("weighted_mode", pd.Series(weighted_mode, index=raw_pitchers.index)).eq(weighted_mode)
            ]
        if pitcher_ids and not raw_pitchers.empty and "pitcher_id" in raw_pitchers.columns:
            raw_pitchers = raw_pitchers.loc[raw_pitchers["pitcher_id"].isin(pitcher_ids)]
        pitcher_metrics = raw_pitchers

        outcomes = _load_remote_daily_parquet(base_url, target_date, "strikeouts/pitcher_start_outcomes.parquet")
        if pitcher_ids and not outcomes.empty and "pitcher_id" in outcomes.columns:
            outcomes = outcomes.loc[pd.to_numeric(outcomes["pitcher_id"], errors="coerce").isin(pitcher_ids)].copy()

        pitcher_fzc = _load_remote_daily_parquet(base_url, target_date, "daily_pitcher_family_zone_context.parquet")
        batter_fzp = _load_remote_daily_parquet(base_url, target_date, "daily_batter_family_zone_profiles.parquet")

        raw_hitters = _load_remote_daily_parquet(base_url, target_date, "top_slate_hitters.parquet")
        if not raw_hitters.empty and "split_key" in raw_hitters.columns:
            raw_hitters = raw_hitters.loc[
                raw_hitters["split_key"].eq(split)
                & raw_hitters.get("recent_window", pd.Series(recent_window, index=raw_hitters.index)).eq(recent_window)
                & raw_hitters.get("weighted_mode", pd.Series(weighted_mode, index=raw_hitters.index)).eq(weighted_mode)
            ]
        hitters_by_team = {}
        if not raw_hitters.empty and "team" in raw_hitters.columns:
            for team in team_list:
                pool = raw_hitters.loc[raw_hitters["team"] == team]
                hitters_by_team[team] = pool.to_dict("records") if not pool.empty else []

    return {
        "pitcher_metrics": pitcher_metrics.to_dict("records") if not pitcher_metrics.empty else [],
        "outcomes": outcomes.to_dict("records") if not outcomes.empty else [],
        "pitcher_fzc": pitcher_fzc.to_dict("records") if not pitcher_fzc.empty else [],
        "batter_fzp": batter_fzp.to_dict("records") if not batter_fzp.empty else [],
        "hitters_by_team": hitters_by_team,
    }


def _pitcher_ids_from_games(games: list[dict]) -> list[int]:
    ids = []
    for g in games:
        for key in ("away_probable_pitcher_id", "home_probable_pitcher_id"):
            pid = g.get(key)
            try:
                if pid is not None and pid == pid:  # NaN != NaN
                    ids.append(int(pid))
            except (TypeError, ValueError):
                pass
    return ids


def _teams_from_games(games: list[dict]) -> list[str]:
    teams = set()
    for g in games:
        for key in ("away_team", "home_team"):
            t = g.get(key)
            if t:
                teams.add(str(t))
    return sorted(teams)


# --------------------------------------------------------------------------- #
# Formatting helpers                                                            #
# --------------------------------------------------------------------------- #

def _fmt_pct(v: object) -> str:
    if v is None or pd.isna(v):
        return "--"
    return f"{float(v) * 100:.1f}%"


def _confidence_color(conf: str) -> str:
    return {"High": "#14532d", "Medium": "#715400", "Low": "#8b1e1e"}.get(conf, "#52606d")


def _confidence_bg(conf: str) -> str:
    return {"High": "#cfeeda", "Medium": "#ffefad", "Low": "#f8d4d4"}.get(conf, "#eef2f7")


def _pill(text: str, bg: str, color: str) -> str:
    return (
        f"<span style='background:{bg};color:{color};padding:3px 9px;"
        f"border-radius:999px;font-size:0.72rem;font-weight:700;"
        f"white-space:nowrap;display:inline-block'>{text}</span>"
    )


def _chain_text(row: dict) -> str:
    pc = row.get("avg_pitch_count", 88.0)
    ppbf = row.get("avg_p_per_bf", 3.88)
    bf = row.get("projected_bf", 0)
    k = row.get("projected_k", 0)
    bb = row.get("projected_bb", 0)
    proxy_note = " (proxy)" if row.get("using_proxy") else ""
    return (
        f"~{pc:.0f} pitches ÷ {ppbf:.2f} P/PA "
        f"→ **{bf:.1f} BF** → **{k:.1f} K** / **{bb:.1f} BB**{proxy_note}"
    )


# --------------------------------------------------------------------------- #
# Table enhancement helpers (options 1, 4, 6)                                 #
# --------------------------------------------------------------------------- #

_CONF_ROW_BG = {
    "High":   "background-color: rgba(16, 185, 129, 0.18)",
    "Medium": "background-color: rgba(245, 158, 11, 0.20)",
    "Low":    "background-color: rgba(220, 38, 38, 0.15)",
}



def _bar_overlay(col: str, original_frame: pd.DataFrame, styles: pd.DataFrame) -> None:
    """Replace background-color on `col` with a blue-intensity 'bar' proportional to value."""
    if col not in original_frame.columns or col not in styles.columns:
        return
    vals = pd.to_numeric(original_frame[col], errors="coerce")
    vmin, vmax = vals.min(), vals.max()
    if pd.isna(vmax) or pd.isna(vmin) or vmax <= vmin:
        return
    for idx in original_frame.index:
        v = vals.loc[idx]
        if pd.isna(v):
            continue
        opacity = 0.08 + ((v - vmin) / (vmax - vmin)) * 0.42  # 0.08 – 0.50
        existing = styles.loc[idx, col]
        parts = [p.strip() for p in existing.split(";") if p.strip() and not p.strip().startswith("background-color:")]
        parts.append(f"background-color: rgba(37, 99, 235, {opacity:.3f})")
        styles.loc[idx, col] = "; ".join(parts)


def _confidence_row_highlight(
    display_frame: pd.DataFrame,
    styles: pd.DataFrame,
    conf_col: str,
    label_cols: list[str],
) -> None:
    """Tint label-column cells with confidence tier color."""
    if conf_col not in display_frame.columns:
        return
    cols_present = [c for c in label_cols if c in styles.columns]
    for idx in display_frame.index:
        bg = _CONF_ROW_BG.get(str(display_frame.loc[idx, conf_col]))
        if bg is None:
            continue
        for col in cols_present:
            existing = styles.loc[idx, col]
            parts = [p.strip() for p in existing.split(";") if p.strip() and not p.strip().startswith("background-color:")]
            parts.append(bg)
            styles.loc[idx, col] = "; ".join(parts)


def _max_min_highlight(
    original_frame: pd.DataFrame,
    styles: pd.DataFrame,
    higher_best: set[str],
    lower_best: set[str],
) -> None:
    """Bold the best and worst cells per column; background colors from heatmap are preserved."""
    for col in higher_best | lower_best:
        if col not in original_frame.columns or col not in styles.columns:
            continue
        vals = pd.to_numeric(original_frame[col], errors="coerce").dropna()
        if len(vals) < 2:
            continue
        best_idx = vals.idxmax() if col in higher_best else vals.idxmin()
        worst_idx = vals.idxmin() if col in higher_best else vals.idxmax()
        for idx in {best_idx, worst_idx}:
            existing = styles.loc[idx, col]
            parts = [p.strip() for p in existing.split(";") if p.strip() and not p.strip().startswith("font-weight:")]
            parts.append("font-weight: 700")
            styles.loc[idx, col] = "; ".join(p for p in parts if p)


def _so_slate_post_styler(
    display_frame: pd.DataFrame,
    original_frame: pd.DataFrame,
    styles: pd.DataFrame,
) -> pd.DataFrame:
    styles = styles.copy()
    _confidence_row_highlight(
        display_frame, styles, "Confidence",
        label_cols=["Pitcher", "Team", "Opp", "Hand", "Confidence"],
    )
    _max_min_highlight(
        original_frame, styles,
        higher_best={"Proj K", "Proj BF", "Avg Pitches", "K Rate", "Matchup K%", "Blend K%", "Mix Whiff", "W. Starts"},
        lower_best={"Proj BB", "BB Rate", "P/PA"},
    )
    return styles


def _so_matchup_post_styler(
    display_frame: pd.DataFrame,
    original_frame: pd.DataFrame,
    styles: pd.DataFrame,
) -> pd.DataFrame:
    styles = styles.copy()
    _bar_overlay("K Prob", original_frame, styles)
    _max_min_highlight(
        original_frame, styles,
        higher_best={"K%", "Mix Whiff", "Family Fit", "Zone Fit", "Matchup", "K Prob"},
        lower_best=set(),
    )
    return styles


# --------------------------------------------------------------------------- #
# Section 1: Slate summary table                                               #
# --------------------------------------------------------------------------- #

def _render_slate_summary(projections: pd.DataFrame) -> None:
    st.markdown("### Slate Summary")
    if projections.empty:
        st.info("No projection data available.")
        return

    display_cols = [
        "pitcher_name", "team", "opp_team", "p_throws",
        "projected_k", "projected_bb", "projected_bf",
        "avg_pitch_count", "avg_p_per_bf",
        "pitcher_k_rate", "lineup_k_rate", "blended_k_rate",
        "pitcher_bb_rate", "pitcher_mix_whiff",
        "weighted_starts", "confidence",
    ]
    display_cols = [c for c in display_cols if c in projections.columns]
    display = projections[display_cols].copy().rename(columns={
        "pitcher_name": "Pitcher",
        "opp_team": "Opp",
        "p_throws": "Hand",
        "projected_k": "Proj K",
        "projected_bb": "Proj BB",
        "projected_bf": "Proj BF",
        "avg_pitch_count": "Avg Pitches",
        "avg_p_per_bf": "P/PA",
        "pitcher_k_rate": "K Rate",
        "lineup_k_rate": "Matchup K%",
        "blended_k_rate": "Blend K%",
        "pitcher_bb_rate": "BB Rate",
        "pitcher_mix_whiff": "Mix Whiff",
        "weighted_starts": "W. Starts",
        "confidence": "Confidence",
    })
    # Replace team abbreviation columns with data URIs so table renders logos
    if "team" in display.columns:
        display.insert(1, "Team", display.pop("team").map(lambda t: team_logo_data_uri(str(t)) or str(t)))
    if "Opp" in display.columns:
        display["Opp"] = display["Opp"].map(lambda t: team_logo_data_uri(str(t)) or str(t))

    render_metric_grid(
        display,
        key="so-slate-summary",
        height=420,
        use_lightweight=True,
        higher_is_better={
            "Proj K",       # more projected Ks = higher ceiling
            "Proj BF",      # more batters faced = more K opportunities
            "Avg Pitches",  # deeper outings = more K chances
            "K Rate",       # pitcher core K rate from SwStr/Ball/CSW/PutAway
            "Matchup K%",   # pitch-mix + zone-aware opponent matchup
            "Blend K%",     # blended final K rate
            "Mix Whiff",    # pitch-mix weighted whiff rate
            "W. Starts",    # more starts = higher confidence
        },
        lower_is_better={
            "Proj BB",      # fewer BBs = cleaner outing
            "BB Rate",      # lower walk rate = better command
            "P/PA",         # fewer pitches per batter = more efficient
        },
        post_styler=_so_slate_post_styler,
    )


# --------------------------------------------------------------------------- #
# Section 2: Per-game projection cards                                          #
# --------------------------------------------------------------------------- #

def _render_projection_card(row: dict) -> None:
    conf = str(row.get("confidence", "Low"))
    bg = _confidence_bg(conf)
    color = _confidence_color(conf)

    pitcher_name = row.get("pitcher_name", "TBD")
    team = row.get("team", "")
    hand = row.get("p_throws", "R")
    proj_k = row.get("projected_k", 0.0)
    proj_bb = row.get("projected_bb", 0.0)
    sample = row.get("sample_starts", 0)
    w_starts = row.get("weighted_starts", 0.0)

    logo_html = team_logo_img_html(team, size=28) if team else ""
    st.markdown(
        f"""
        <div style='display:flex;align-items:center;gap:8px;margin-bottom:4px'>
            {logo_html}
            <span style='font-size:1.1rem;font-weight:700'>{pitcher_name}</span>
            <span style='font-size:0.85rem;color:#666'>({hand}HP)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    k_col, bb_col = st.columns(2)
    with k_col:
        st.metric("Proj K", f"{proj_k:.1f}")
    with bb_col:
        st.metric("Proj BB", f"{proj_bb:.1f}")

    # Transparent chain
    st.markdown(_chain_text(row))

    # Rate breakdown
    pk_rate = row.get("pitcher_k_rate", 0)
    lk_rate = row.get("lineup_k_rate", 0)
    mix_whiff = row.get("pitcher_mix_whiff", 0)
    st.markdown(
        f"Pitcher core K rate: **{pk_rate:.1%}** | "
        f"Matchup K rate: **{lk_rate:.1%}** | "
        f"Mix whiff: **{mix_whiff:.1%}**"
    )

    # Confidence pill
    pill_html = _pill(f"Confidence: {conf}", bg, color)
    sample_note = f"({sample} starts, {w_starts:.1f} wtd.)"
    st.markdown(f"{pill_html} {sample_note}", unsafe_allow_html=True)


def _render_game_cards(game: dict, projections: pd.DataFrame) -> None:
    game_pk = game.get("game_pk")
    away_team = str(game.get("away_team", "") or "")
    home_team = str(game.get("home_team", "") or "")
    game_rows = projections.loc[projections["game_pk"] == game_pk] if "game_pk" in projections.columns else pd.DataFrame()

    away_logo = team_logo_img_html(away_team, size=22) if away_team else away_team
    home_logo = team_logo_img_html(home_team, size=22) if home_team else home_team
    title_html = (
        f"<div style='display:inline-flex;align-items:center;gap:6px'>"
        f"{away_logo}"
        f"<span style='color:#666;font-weight:600'>@</span>"
        f"{home_logo}"
        f"</div>"
    )
    with st.expander(f"{away_team} @ {home_team}", expanded=True):
        st.markdown(title_html, unsafe_allow_html=True)
        if game_rows.empty:
            st.caption("No probable pitchers found for this game.")
            return
        cols = st.columns(2)
        for idx, (_, prow) in enumerate(game_rows.iterrows()):
            with cols[idx % 2]:
                with st.container(border=True):
                    _render_projection_card(prow.to_dict())


# --------------------------------------------------------------------------- #
# Section 3: Matchup breakdown                                                  #
# --------------------------------------------------------------------------- #

def _render_matchup_breakdown(pitcher_row: dict, hitter_k_probs: list[dict]) -> None:
    pitcher_name = pitcher_row.get("pitcher_name", "")
    st.markdown(f"#### {pitcher_name} — Per-Hitter Matchup")
    if not hitter_k_probs:
        st.caption("No lineup hitter data available for this pitcher.")
        return
    df = pd.DataFrame(hitter_k_probs)
    display = df.rename(columns={
        "hitter_name": "Hitter",
        "bats": "Bats",
        "strikeout_rate": "K%",
        "mix_whiff": "Mix Whiff",
        "family_vuln": "Family Fit",
        "zone_whiff": "Zone Fit",
        "matchup_scalar": "Matchup",
        "k_prob": "K Prob",
    })
    # Insert Team logo column after Hitter
    if "team" in display.columns:
        display.insert(1, "Team", display.pop("team").map(lambda t: team_logo_data_uri(str(t)) or str(t)))
    # Keep numeric so _build_lightweight_grid_payload can color them
    for col in ["K%", "Mix Whiff", "Family Fit", "Zone Fit", "Matchup", "K Prob"]:
        if col in display.columns:
            display[col] = pd.to_numeric(display[col], errors="coerce")
    render_metric_grid(
        display,
        key=f"so-matchup-{pitcher_row.get('pitcher_id', 0)}",
        height=320,
        use_lightweight=True,
        higher_is_better={"K%", "Mix Whiff", "Family Fit", "Zone Fit", "Matchup", "K Prob"},
        post_styler=_so_matchup_post_styler,
    )


# --------------------------------------------------------------------------- #
# Section 4: K/BB trend chart                                                   #
# --------------------------------------------------------------------------- #

def _render_trend_chart(pitcher_row: dict, outcomes: pd.DataFrame) -> None:
    pitcher_name = pitcher_row.get("pitcher_name", "")
    pitcher_id = pitcher_row.get("pitcher_id", 0)
    proj_k = pitcher_row.get("projected_k", None)

    st.markdown(f"#### {pitcher_name} — Recent Start Trend")

    pid_outcomes = outcomes.loc[outcomes["pitcher_id"] == pitcher_id].copy() if not outcomes.empty and "pitcher_id" in outcomes.columns else pd.DataFrame()

    if pid_outcomes.empty:
        st.caption("No historical start data available.")
        return

    if "slate_date" in pid_outcomes.columns:
        pid_outcomes["slate_date"] = pd.to_datetime(pid_outcomes["slate_date"], errors="coerce")
        pid_outcomes = pid_outcomes.sort_values("slate_date").tail(15)

    chart_data: list[dict] = []
    for _, row in pid_outcomes.iterrows():
        d = row.get("slate_date")
        label = pd.Timestamp(d).strftime("%m/%d") if pd.notna(d) else "?"
        chart_data.append({
            "Start": label,
            "Strikeouts": int(row.get("strikeouts", 0) or 0),
            "Walks": int(row.get("walks", 0) or 0),
        })

    if not chart_data:
        st.caption("No starts to chart.")
        return

    chart_df = pd.DataFrame(chart_data)

    if HAS_ALTAIR and alt is not None:
        long = chart_df.melt("Start", value_vars=["Strikeouts", "Walks"], var_name="Stat", value_name="Count")
        color_scale = alt.Scale(domain=["Strikeouts", "Walks"], range=["#2563eb", "#d97706"])
        bars = (
            alt.Chart(long)
            .mark_bar(opacity=0.85)
            .encode(
                x=alt.X("Start:N", sort=None, title="Start"),
                y=alt.Y("Count:Q", title="Count"),
                color=alt.Color("Stat:N", scale=color_scale),
                xOffset="Stat:N",
                tooltip=["Start", "Stat", "Count"],
            )
        )
        chart = bars
        if proj_k is not None:
            ref_df = pd.DataFrame([{"proj_k": proj_k}])
            ref_line = (
                alt.Chart(ref_df)
                .mark_rule(strokeDash=[6, 3], color="#1e40af", strokeWidth=2)
                .encode(y="proj_k:Q")
            )
            chart = bars + ref_line
        st.altair_chart(
            chart.properties(height=280, title=f"Last {len(chart_df)} Starts — K (blue) & BB (amber) | Dashed = Proj K"),
            use_container_width=True,
        )
    else:
        st.bar_chart(chart_df.set_index("Start")[["Strikeouts", "Walks"]])


# --------------------------------------------------------------------------- #
# Section 5: Arsenal K drivers                                                  #
# --------------------------------------------------------------------------- #

def _render_arsenal_k_drivers(pitcher_row: dict, pitcher_fzc: pd.DataFrame) -> None:
    pitcher_name = pitcher_row.get("pitcher_name", "")
    pitcher_id = pitcher_row.get("pitcher_id", 0)
    st.markdown(f"#### {pitcher_name} — Arsenal K Drivers")

    if pitcher_fzc.empty or "pitcher_id" not in pitcher_fzc.columns:
        st.caption("No family-zone context available.")
        return

    rows = pitcher_fzc.loc[pitcher_fzc["pitcher_id"] == pitcher_id].copy()
    if rows.empty:
        st.caption("No family-zone data for this pitcher.")
        return

    display_cols = [c for c in ["pitch_family", "zone_bucket", "usage_rate_overall", "whiff_rate"] if c in rows.columns]
    display = rows[display_cols].copy().rename(columns={
        "pitch_family": "Family",
        "zone_bucket": "Zone",
        "usage_rate_overall": "Usage%",
        "whiff_rate": "Whiff Rate",
    })
    if "Usage%" in display.columns:
        display["Usage%"] = pd.to_numeric(display["Usage%"], errors="coerce").map(lambda v: f"{v:.1%}" if pd.notna(v) else "--")
    if "Whiff Rate" in display.columns:
        display["Whiff Rate"] = pd.to_numeric(display["Whiff Rate"], errors="coerce").map(lambda v: f"{v:.1%}" if pd.notna(v) else "--")
    if "Family" in display.columns:
        display = display.sort_values(["Family", "Zone"])

    render_metric_grid(
        display,
        key=f"so-arsenal-{pitcher_id}",
        height=260,
        use_lightweight=True,
        higher_is_better={"Whiff Rate"},
    )

    # Altair: grouped horizontal bars by family + zone
    if HAS_ALTAIR and alt is not None and not rows.empty:
        chart_rows = rows.copy()
        chart_rows["label"] = chart_rows["pitch_family"].str.capitalize() + " / " + chart_rows["zone_bucket"].str.capitalize()
        chart_rows["whiff_pct"] = pd.to_numeric(chart_rows.get("whiff_rate", 0), errors="coerce").fillna(0)
        zone_color = alt.Scale(domain=["heart", "shadow"], range=["#60a5fa", "#1e40af"])
        bar = (
            alt.Chart(chart_rows)
            .mark_bar()
            .encode(
                y=alt.Y("label:N", sort="-x", title="Family / Zone"),
                x=alt.X("whiff_pct:Q", title="Whiff Rate", axis=alt.Axis(format=".0%")),
                color=alt.Color("zone_bucket:N", scale=zone_color, title="Zone"),
                tooltip=["pitch_family", "zone_bucket", alt.Tooltip("whiff_pct:Q", format=".1%"), alt.Tooltip("usage_rate_overall:Q", format=".1%", title="Usage%")],
            )
            .properties(height=220, title="Whiff Rate by Family × Zone")
        )
        st.altair_chart(bar, use_container_width=True)


# --------------------------------------------------------------------------- #
# Per-game detail (sections 3-5 for a selected game)                           #
# --------------------------------------------------------------------------- #

def _render_game_detail(
    game: dict,
    projections: pd.DataFrame,
    outcomes: pd.DataFrame,
    pitcher_fzc: pd.DataFrame,
    hitters_by_team: dict[str, pd.DataFrame],
    section: str = "Matchup",
) -> None:
    game_pk = game.get("game_pk")
    game_rows = projections.loc[projections["game_pk"] == game_pk].to_dict("records") if "game_pk" in projections.columns else []

    if not game_rows:
        st.info("No projection data for this game.")
        return

    for prow in game_rows:
        if section == "Matchup":
            _render_matchup_breakdown(prow, prow.get("_hitter_k_probs", []))
        elif section == "Trend":
            _render_trend_chart(prow, outcomes)
        elif section == "Arsenal":
            _render_arsenal_k_drivers(prow, pitcher_fzc)
        st.divider()


# --------------------------------------------------------------------------- #
# Main entry point                                                              #
# --------------------------------------------------------------------------- #

def main() -> None:
    st.set_page_config(page_title="Strikeouts", page_icon=page_icon_path(), layout="wide")
    apply_branding_head()
    st.title("Strikeouts & Walks")
    st.caption(
        "Projects K and BB totals via a transparent chain: avg pitch count → pitches-per-batter "
        "→ batters faced → K/BB. Matchup blend: 70% pitcher historical rate + 30% pitch-mix/zone-aware lineup."
    )

    if not _require_admin_password():
        return

    config = AppConfig()
    target_date = st.sidebar.date_input("Slate date", value=_default_date(config))

    split = st.sidebar.selectbox("Split", ["overall", "vs_rhp", "vs_lhp"], index=0)
    recent_window = st.sidebar.selectbox("Window", ["season", "last_30", "last_15"], index=0)
    weighted_mode = st.sidebar.selectbox("Weighted mode", ["weighted", "unweighted"], index=0)

    games, source, loaded_date = _load_slate(config, target_date)
    if source == "none" or not games:
        st.error("No slate data found for this date.")
        return
    if loaded_date != target_date:
        st.caption(f"Using most recent available slate: {loaded_date.isoformat()}")

    pitcher_ids = _pitcher_ids_from_games(games)
    team_list = _teams_from_games(games)

    if not pitcher_ids:
        st.info("No probable pitchers found on today's slate.")
        return

    with st.spinner("Loading projections..."):
        raw = _load_all_data(
            loaded_date,
            tuple(sorted(set(pitcher_ids))),
            tuple(sorted(set(team_list))),
            split,
            recent_window,
            weighted_mode,
        )

    pitcher_metrics = pd.DataFrame(raw["pitcher_metrics"])
    outcomes_frame = pd.DataFrame(raw["outcomes"])
    pitcher_fzc = pd.DataFrame(raw["pitcher_fzc"])
    batter_fzp = pd.DataFrame(raw["batter_fzp"])
    hitters_by_team_raw = raw["hitters_by_team"]
    hitters_by_team = {team: pd.DataFrame(rows) for team, rows in hitters_by_team_raw.items()}

    projections = build_slate_projections(
        games,
        pitcher_metrics,
        outcomes_frame,
        hitters_by_team,
        pitcher_fzc,
        batter_fzp,
    )

    if projections.empty:
        st.info("No projections could be built for this slate.")
        return

    # --- Section 1: Slate summary (always shown) ---
    _render_slate_summary(projections)

    # --- Sticky game navigator ---
    selected_label, selected_games, selected_section = render_sticky_logo_game_nav(
        games,
        key_prefix="so-game-nav",
        sections=["Projection", "Matchup", "Trend", "Arsenal"],
        default_section="Projection",
    )

    if not selected_games:
        return  # "Slate Summary" card selected — summary table above is sufficient

    selected_game = selected_games[0]

    if selected_section == "Projection":
        _render_game_cards(selected_game, projections)
    else:
        _render_game_detail(
            selected_game,
            projections,
            outcomes_frame,
            pitcher_fzc,
            hitters_by_team,
            section=selected_section,
        )
