"""Strikeout and walk projection logic for Kasper's Strikeouts page.

Projection chain: avg pitch count per start -> pitches per batter -> batters faced -> K/BB totals.
Matchup blend:    70% pitcher core rate + 30% pitch-mix-and-zone-aware matchup component.
Year weighting:   game outcome starts are weighted by the same year schedule used in metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

OUTCOME_YEAR_WEIGHTS: dict[int, float] = {
    2026: 1.35,
    2025: 1.00,
    2024: 0.90,
    2023: 0.70,
    2022: 0.50,
}
DEFAULT_YEAR_WEIGHT = 0.35

LEAGUE_AVG_PITCH_COUNT = 88.0
LEAGUE_AVG_P_PER_BF = 3.88
LEAGUE_AVG_K_RATE = 0.230
LEAGUE_AVG_BB_RATE = 0.083
LEAGUE_AVG_SWSTR = 0.110
LEAGUE_AVG_BALL = 0.340
LEAGUE_AVG_CSW = 0.290
LEAGUE_AVG_PUTAWAY = 0.200
LEAGUE_AVG_HITTER_K_RATE = 0.220
LEAGUE_AVG_CALLED_STRIKE = 0.170
LEAGUE_AVG_XWOBA = 0.315

SWSTR_TO_K_CALIBRATION = 2.1
BALL_TO_BB_CALIBRATION = 0.245

MIN_WEIGHTED_STARTS = 5.0
PITCHER_K_RATE_MIN = 0.12
PITCHER_K_RATE_MAX = 0.38
MATCHUP_K_RATE_MIN = 0.12
MATCHUP_K_RATE_MAX = 0.40


@dataclass
class ProjectionResult:
    pitcher_id: int
    pitcher_name: str
    team: str
    p_throws: str
    projected_k: float
    projected_bb: float
    projected_bf: float
    avg_pitch_count: float
    avg_p_per_bf: float
    pitcher_k_rate: float
    pitcher_bb_rate: float
    lineup_k_rate: float
    blended_k_rate: float
    sample_starts: int
    weighted_starts: float
    confidence: str
    using_proxy: bool
    pitcher_mix_whiff: float = 0.0
    hitter_k_probs: list[dict] = field(default_factory=list)


def _year_weight(game_year: int) -> float:
    return OUTCOME_YEAR_WEIGHTS.get(game_year, DEFAULT_YEAR_WEIGHT)


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float | None:
    w = weights[values.notna() & weights.gt(0)]
    v = values[values.notna() & weights.gt(0)]
    if w.sum() == 0:
        return None
    return float((v * w).sum() / w.sum())


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _metric_or_default(value: object, default: float) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) else float(default)


def _bounded_score(value: float, low: float, high: float, inverse: bool = False) -> float:
    if abs(high - low) < 1e-9:
        return 0.5
    score = _clamp01((float(value) - low) / (high - low))
    return 1.0 - score if inverse else score


def _add_year_weights(outcomes: pd.DataFrame) -> pd.DataFrame:
    out = outcomes.copy()
    if "slate_date" in out.columns:
        years = pd.to_datetime(out["slate_date"], errors="coerce").dt.year.fillna(0).astype(int)
    else:
        years = pd.Series(0, index=out.index)
    out["_yw"] = years.map(lambda y: _year_weight(y))
    return out


def _compute_outcome_rates(outcomes: pd.DataFrame) -> tuple[float, float, float, float, int, float]:
    """Return (avg_pitch_count, avg_p_per_bf, k_rate, bb_rate, n_starts, weighted_starts)."""
    if outcomes.empty:
        return LEAGUE_AVG_PITCH_COUNT, LEAGUE_AVG_P_PER_BF, LEAGUE_AVG_K_RATE, LEAGUE_AVG_BB_RATE, 0, 0.0

    df = _add_year_weights(outcomes)
    usable = df.loc[df["batters_faced"].gt(0)].copy()
    n_starts = len(usable)
    w_starts = float(usable["_yw"].sum())

    pc_usable = usable.loc[usable.get("pitch_count", pd.Series(0, index=usable.index)).gt(0)]
    if len(pc_usable) >= 3 and "pitch_count" in pc_usable.columns:
        avg_pitch_count = _weighted_mean(
            pd.to_numeric(pc_usable["pitch_count"], errors="coerce"),
            pc_usable["_yw"],
        ) or LEAGUE_AVG_PITCH_COUNT
        p_per_bf_series = (
            pd.to_numeric(pc_usable["pitch_count"], errors="coerce")
            / pd.to_numeric(pc_usable["batters_faced"], errors="coerce").replace(0, pd.NA)
        )
        avg_p_per_bf = _weighted_mean(p_per_bf_series, pc_usable["_yw"]) or LEAGUE_AVG_P_PER_BF
    else:
        avg_pitch_count = LEAGUE_AVG_PITCH_COUNT
        avg_p_per_bf = LEAGUE_AVG_P_PER_BF

    k_per_bf = pd.to_numeric(usable["strikeouts"], errors="coerce") / pd.to_numeric(
        usable["batters_faced"], errors="coerce"
    ).replace(0, pd.NA)
    bb_per_bf = pd.to_numeric(usable["walks"], errors="coerce") / pd.to_numeric(
        usable["batters_faced"], errors="coerce"
    ).replace(0, pd.NA)

    k_rate = _weighted_mean(k_per_bf, usable["_yw"]) or LEAGUE_AVG_K_RATE
    bb_rate = _weighted_mean(bb_per_bf, usable["_yw"]) or LEAGUE_AVG_BB_RATE
    return avg_pitch_count, avg_p_per_bf, k_rate, bb_rate, n_starts, w_starts


def compute_pitcher_core_k_rate(pitcher_metrics_row: pd.Series) -> float:
    """Build the pitcher-side strikeout rate from SwStr / Ball / CSW / PutAway."""
    swstr_pct = _metric_or_default(pitcher_metrics_row.get("swstr_pct"), LEAGUE_AVG_SWSTR)
    ball_pct = _metric_or_default(pitcher_metrics_row.get("ball_pct"), LEAGUE_AVG_BALL)
    csw_pct = _metric_or_default(pitcher_metrics_row.get("csw_pct"), LEAGUE_AVG_CSW)
    putaway_pct = _metric_or_default(pitcher_metrics_row.get("putaway_pct"), LEAGUE_AVG_PUTAWAY)

    swstr_score = _bounded_score(swstr_pct, 0.08, 0.17)
    ball_control_score = _bounded_score(ball_pct, 0.28, 0.40, inverse=True)
    csw_score = _bounded_score(csw_pct, 0.24, 0.36)
    putaway_score = _bounded_score(putaway_pct, 0.12, 0.32)

    pitcher_core_score = (
        (swstr_score * 0.50)
        + (ball_control_score * 0.25)
        + (csw_score * 0.15)
        + (putaway_score * 0.10)
    )
    return _clamp(
        PITCHER_K_RATE_MIN + (pitcher_core_score * (PITCHER_K_RATE_MAX - PITCHER_K_RATE_MIN)),
        PITCHER_K_RATE_MIN,
        PITCHER_K_RATE_MAX,
    )


def compute_pitcher_mix_whiff(pitcher_id: int, pitcher_fzc: pd.DataFrame) -> float:
    """Compute pitcher's pitch-mix-weighted expected whiff rate from family-zone context."""
    if pitcher_fzc.empty or "pitcher_id" not in pitcher_fzc.columns:
        return LEAGUE_AVG_SWSTR
    rows = pitcher_fzc.loc[pitcher_fzc["pitcher_id"] == pitcher_id].copy()
    if rows.empty:
        return LEAGUE_AVG_SWSTR
    usage = pd.to_numeric(rows.get("usage_rate_overall", pd.Series(dtype=float)), errors="coerce").fillna(0)
    whiff = pd.to_numeric(rows.get("whiff_rate", pd.Series(dtype=float)), errors="coerce").fillna(0)
    total = float((usage * whiff).sum())
    return total if total > 0 else LEAGUE_AVG_SWSTR


def _family_zone_vulnerability_scalar(
    batter_id: int,
    p_throws: str,
    pitcher_fzc_rows: pd.DataFrame,
    batter_fzp: pd.DataFrame,
) -> tuple[float, float, float]:
    """Return family, zone-whiff, and combined matchup scalars for this hitter."""
    if batter_fzp.empty or pitcher_fzc_rows.empty:
        return 1.0, 1.0, 1.0

    hand_key = "vs_rhp" if str(p_throws).upper() == "R" else "vs_lhp"
    hand_series = batter_fzp.get("pitcher_hand_key", pd.Series("overall", index=batter_fzp.index))
    hitter_rows = batter_fzp.loc[
        (batter_fzp["batter_id"] == batter_id)
        & hand_series.eq(hand_key)
    ].copy()
    if hitter_rows.empty:
        hitter_rows = batter_fzp.loc[
            (batter_fzp["batter_id"] == batter_id)
            & hand_series.eq("overall")
        ].copy()
    if hitter_rows.empty:
        return 1.0, 1.0, 1.0

    pitcher_usage = pitcher_fzc_rows.loc[
        pitcher_fzc_rows.get("pitch_family").notna()
        & pitcher_fzc_rows.get("zone_bucket").notna()
    ].copy()
    if pitcher_usage.empty:
        return 1.0, 1.0, 1.0
    pitcher_usage["usage_rate_overall"] = pd.to_numeric(
        pitcher_usage.get("usage_rate_overall"), errors="coerce"
    ).fillna(0.0)
    total_usage = float(pitcher_usage["usage_rate_overall"].sum())
    if total_usage <= 0:
        return 1.0, 1.0, 1.0
    pitcher_usage["share"] = pitcher_usage["usage_rate_overall"] / total_usage

    hitter_rows["weighted_sample_size"] = pd.to_numeric(
        hitter_rows.get("weighted_sample_size"), errors="coerce"
    ).fillna(
        pd.to_numeric(
            hitter_rows.get("sample_size", pd.Series(0.0, index=hitter_rows.index)),
            errors="coerce",
        ).fillna(0.0)
    )
    hitter_rows["xwoba"] = pd.to_numeric(hitter_rows.get("xwoba"), errors="coerce")

    family_scalar_sum = 0.0
    zone_scalar_sum = 0.0
    weight_sum = 0.0
    for _, pu_row in pitcher_usage.iterrows():
        fam = pu_row["pitch_family"]
        zone_bucket = pu_row["zone_bucket"]
        share = float(pu_row["share"])
        whiff_rate = _metric_or_default(pu_row.get("whiff_rate"), LEAGUE_AVG_SWSTR)
        called_strike_rate = _metric_or_default(pu_row.get("called_strike_rate"), LEAGUE_AVG_CALLED_STRIKE)

        hzone = hitter_rows.loc[
            hitter_rows["pitch_family"].eq(fam)
            & hitter_rows["zone_bucket"].eq(zone_bucket)
        ]
        if hzone.empty:
            hzone = hitter_rows.loc[hitter_rows["pitch_family"].eq(fam)]

        if hzone.empty:
            family_scalar = 1.0
        else:
            weights = pd.to_numeric(hzone["weighted_sample_size"], errors="coerce").fillna(0.0)
            hxwoba = _weighted_mean(hzone["xwoba"], weights) if weights.gt(0).any() else None
            sample = float(weights.sum())
            if hxwoba is None or sample < 20:
                family_scalar = 1.0
            else:
                family_scalar = _clamp(1.0 + ((LEAGUE_AVG_XWOBA - float(hxwoba)) / LEAGUE_AVG_XWOBA * 0.75), 0.75, 1.35)

        zone_scalar = _clamp(
            (0.75 * (whiff_rate / LEAGUE_AVG_SWSTR)) + (0.25 * (called_strike_rate / LEAGUE_AVG_CALLED_STRIKE)),
            0.75,
            1.35,
        )
        family_scalar_sum += share * family_scalar
        zone_scalar_sum += share * zone_scalar
        weight_sum += share

    if weight_sum <= 0:
        return 1.0, 1.0, 1.0
    family_scalar = float(family_scalar_sum / weight_sum)
    zone_scalar = float(zone_scalar_sum / weight_sum)
    combined_scalar = _clamp((family_scalar * 0.60) + (zone_scalar * 0.40), 0.75, 1.35)
    return family_scalar, zone_scalar, combined_scalar


def _hitter_k_baseline(hitter: pd.Series) -> float:
    strikeout_rate = pd.to_numeric(pd.Series([hitter.get("strikeout_rate")]), errors="coerce").iloc[0]
    if pd.notna(strikeout_rate):
        return _clamp(float(strikeout_rate), 0.10, 0.38)
    swstr = _metric_or_default(hitter.get("swstr_pct"), LEAGUE_AVG_SWSTR)
    return _clamp(swstr * SWSTR_TO_K_CALIBRATION, 0.10, 0.38)


def compute_pitch_mix_lineup_k_rate(
    pitcher_id: int,
    p_throws: str,
    opp_hitters: pd.DataFrame,
    pitcher_fzc: pd.DataFrame,
    batter_fzp: pd.DataFrame,
    pitcher_mix_whiff: float,
) -> tuple[float, list[dict]]:
    """Compute the pitch-mix-and-zone-aware matchup K rate (30% component)."""
    if opp_hitters.empty:
        return LEAGUE_AVG_HITTER_K_RATE, []

    pitcher_rows = pitcher_fzc.loc[pitcher_fzc["pitcher_id"] == pitcher_id] if not pitcher_fzc.empty else pd.DataFrame()

    hitter_k_probs: list[dict] = []
    k_rates: list[float] = []

    for _, hitter in opp_hitters.iterrows():
        hitter_k_baseline = _hitter_k_baseline(hitter)

        batter_id = hitter.get("batter_id") or hitter.get("player_id")
        family_scalar = 1.0
        zone_scalar = 1.0
        matchup_scalar = 1.0
        if batter_id is not None and not pitcher_rows.empty and not batter_fzp.empty:
            try:
                family_scalar, zone_scalar, matchup_scalar = _family_zone_vulnerability_scalar(
                    int(batter_id), p_throws, pitcher_rows, batter_fzp
                )
            except Exception:
                family_scalar, zone_scalar, matchup_scalar = 1.0, 1.0, 1.0

        hitter_k_rate = _clamp(hitter_k_baseline * matchup_scalar, MATCHUP_K_RATE_MIN, MATCHUP_K_RATE_MAX)
        k_rates.append(hitter_k_rate)
        hitter_k_probs.append(
            {
                "hitter_name": hitter.get("hitter_name", ""),
                "team": hitter.get("team", ""),
                "bats": hitter.get("bats", ""),
                "strikeout_rate": round(hitter_k_baseline, 3),
                "mix_whiff": round(pitcher_mix_whiff, 3),
                "family_vuln": round(family_scalar, 3),
                "zone_whiff": round(zone_scalar, 3),
                "matchup_scalar": round(matchup_scalar, 3),
                "k_prob": round(hitter_k_rate, 3),
            }
        )

    lineup_k_rate = float(sum(k_rates) / len(k_rates)) if k_rates else LEAGUE_AVG_HITTER_K_RATE
    return lineup_k_rate, hitter_k_probs


def _confidence_label(weighted_starts: float) -> str:
    if weighted_starts >= 15:
        return "High"
    if weighted_starts >= MIN_WEIGHTED_STARTS:
        return "Medium"
    return "Low"


def compute_pitcher_projection(
    pitcher_metrics_row: pd.Series,
    outcomes: pd.DataFrame,
    opp_hitters: pd.DataFrame,
    pitcher_fzc: pd.DataFrame,
    batter_fzp: pd.DataFrame,
) -> ProjectionResult:
    """Compute a full K/BB projection for one starting pitcher."""
    pitcher_id = int(pitcher_metrics_row.get("pitcher_id", 0) or 0)
    pitcher_name = str(pitcher_metrics_row.get("pitcher_name", "") or "")
    team = str(pitcher_metrics_row.get("team", "") or "")
    p_throws = str(pitcher_metrics_row.get("p_throws", "R") or "R")

    avg_pitch_count, avg_p_per_bf, _hist_k_rate, hist_bb_rate, n_starts, w_starts = _compute_outcome_rates(outcomes)
    using_proxy = w_starts < MIN_WEIGHTED_STARTS
    projected_bf = avg_pitch_count / avg_p_per_bf if avg_p_per_bf > 0 else LEAGUE_AVG_PITCH_COUNT / LEAGUE_AVG_P_PER_BF

    pitcher_k_rate = compute_pitcher_core_k_rate(pitcher_metrics_row)
    ball_pct = _metric_or_default(pitcher_metrics_row.get("ball_pct"), LEAGUE_AVG_BALL)
    pitcher_bb_rate = hist_bb_rate if w_starts >= MIN_WEIGHTED_STARTS else (ball_pct * BALL_TO_BB_CALIBRATION)

    pitcher_mix_whiff = compute_pitcher_mix_whiff(pitcher_id, pitcher_fzc)
    lineup_k_rate, hitter_k_probs = compute_pitch_mix_lineup_k_rate(
        pitcher_id, p_throws, opp_hitters, pitcher_fzc, batter_fzp, pitcher_mix_whiff
    )

    blended_k_rate = (pitcher_k_rate * 0.70) + (lineup_k_rate * 0.30)
    projected_k = projected_bf * blended_k_rate
    projected_bb = projected_bf * pitcher_bb_rate

    return ProjectionResult(
        pitcher_id=pitcher_id,
        pitcher_name=pitcher_name,
        team=team,
        p_throws=p_throws,
        projected_k=round(projected_k, 1),
        projected_bb=round(projected_bb, 1),
        projected_bf=round(projected_bf, 1),
        avg_pitch_count=round(avg_pitch_count, 1),
        avg_p_per_bf=round(avg_p_per_bf, 2),
        pitcher_k_rate=round(pitcher_k_rate, 3),
        pitcher_bb_rate=round(pitcher_bb_rate, 3),
        lineup_k_rate=round(lineup_k_rate, 3),
        blended_k_rate=round(blended_k_rate, 3),
        sample_starts=n_starts,
        weighted_starts=round(w_starts, 1),
        confidence=_confidence_label(w_starts),
        using_proxy=using_proxy,
        pitcher_mix_whiff=round(pitcher_mix_whiff, 3),
        hitter_k_probs=hitter_k_probs,
    )


def build_slate_projections(
    games: list[dict],
    pitcher_metrics: pd.DataFrame,
    outcomes_frame: pd.DataFrame,
    hitters_by_team: dict[str, pd.DataFrame],
    pitcher_fzc: pd.DataFrame,
    batter_fzp: pd.DataFrame,
) -> pd.DataFrame:
    """Build one projection row per starting pitcher for the full slate."""
    rows: list[dict] = []

    metrics_by_id: dict[int, pd.Series] = {}
    if not pitcher_metrics.empty and "pitcher_id" in pitcher_metrics.columns:
        for _, row in pitcher_metrics.iterrows():
            pid = int(row.get("pitcher_id", 0) or 0)
            if pid:
                metrics_by_id[pid] = row

    outcomes_by_id: dict[int, pd.DataFrame] = {}
    if not outcomes_frame.empty and "pitcher_id" in outcomes_frame.columns:
        for pid, grp in outcomes_frame.groupby("pitcher_id"):
            outcomes_by_id[int(pid)] = grp

    for game in games:
        game_pk = game.get("game_pk")
        away_team = str(game.get("away_team", "") or "")
        home_team = str(game.get("home_team", "") or "")

        for role, team, opp_team, pitcher_id_key, pitcher_name_key in [
            ("away", away_team, home_team, "away_probable_pitcher_id", "away_probable_pitcher_name"),
            ("home", home_team, away_team, "home_probable_pitcher_id", "home_probable_pitcher_name"),
        ]:
            pitcher_id = game.get(pitcher_id_key)
            pitcher_name = game.get(pitcher_name_key, "TBD") or "TBD"
            try:
                if pitcher_id is None or pitcher_id != pitcher_id:
                    continue
                pitcher_id = int(pitcher_id)
            except (TypeError, ValueError):
                continue

            metrics_row = metrics_by_id.get(
                pitcher_id,
                pd.Series(
                    {
                        "pitcher_id": pitcher_id,
                        "pitcher_name": pitcher_name,
                        "team": team,
                        "p_throws": "R",
                        "swstr_pct": LEAGUE_AVG_SWSTR,
                        "ball_pct": LEAGUE_AVG_BALL,
                        "csw_pct": LEAGUE_AVG_CSW,
                        "putaway_pct": LEAGUE_AVG_PUTAWAY,
                    }
                ),
            )
            opp_hitters = hitters_by_team.get(opp_team, pd.DataFrame())
            outcomes = outcomes_by_id.get(pitcher_id, pd.DataFrame())

            proj = compute_pitcher_projection(metrics_row, outcomes, opp_hitters, pitcher_fzc, batter_fzp)
            rows.append(
                {
                    "game_pk": game_pk,
                    "role": role,
                    "team": team,
                    "opp_team": opp_team,
                    "away_team": away_team,
                    "home_team": home_team,
                    "pitcher_id": pitcher_id,
                    "pitcher_name": proj.pitcher_name or pitcher_name,
                    "p_throws": proj.p_throws,
                    "projected_k": proj.projected_k,
                    "projected_bb": proj.projected_bb,
                    "projected_bf": proj.projected_bf,
                    "avg_pitch_count": proj.avg_pitch_count,
                    "avg_p_per_bf": proj.avg_p_per_bf,
                    "pitcher_k_rate": proj.pitcher_k_rate,
                    "pitcher_bb_rate": proj.pitcher_bb_rate,
                    "lineup_k_rate": proj.lineup_k_rate,
                    "blended_k_rate": proj.blended_k_rate,
                    "pitcher_mix_whiff": proj.pitcher_mix_whiff,
                    "sample_starts": proj.sample_starts,
                    "weighted_starts": proj.weighted_starts,
                    "confidence": proj.confidence,
                    "using_proxy": proj.using_proxy,
                    "_hitter_k_probs": proj.hitter_k_probs,
                }
            )

    return pd.DataFrame(rows)
