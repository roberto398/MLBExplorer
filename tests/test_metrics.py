from datetime import date, timedelta

import pandas as pd

from mlb_dashboard.config import AppConfig, DEFAULT_RECENT_WINDOWS, DEFAULT_SPLITS
from mlb_dashboard.build import (
    BuildContext,
    _aggregate_hitter_metrics,
    _build_hitter_hr_form,
    _build_pitcher_start_outcomes,
    _filter_pitcher_start_outcomes_for_slate,
    _save_per_game_files,
    _save_top_slate_board_files,
)
from mlb_dashboard.dashboard_views import BEST_MATCHUP_COLUMNS, HITTER_PRESETS, add_hitter_matchup_score, build_slate_summary_best_matchups, build_slate_summary_matchup_overview, filter_excluded_pitchers_from_hitter_pool, normalize_series
from mlb_dashboard.local_store import (
    compute_hitter_rolling,
    compute_pitcher_rolling,
    load_local_source_payload,
    read_hitter_exit_velo_events,
    read_latest_prop_odds_snapshot,
    read_prop_odds_history,
    write_props_odds_snapshot,
    write_statcast_events,
    write_tracking_payload,
    read_hitter_snapshots_for_date,
)
from mlb_dashboard.metrics import add_metric_flags, is_barrel
from mlb_dashboard.ui_components import SLATE_SUMMARY_SELECTION, resolve_logo_game_selection


def test_game_selection_defaults_to_first_game_detail():
    slate = [
        {"away_team": "AAA", "home_team": "BBB", "game_pk": 1},
        {"away_team": "CCC", "home_team": "DDD", "game_pk": 2},
    ]
    selection, selected_games = resolve_logo_game_selection(slate, None)

    assert selection == "AAA @ BBB"
    assert [game["game_pk"] for game in selected_games] == [1]


def test_game_selection_summary_prepares_no_detail_games():
    slate = [{"away_team": "AAA", "home_team": "BBB", "game_pk": 1}]
    selection, selected_games = resolve_logo_game_selection(slate, SLATE_SUMMARY_SELECTION)

    assert selection == "Slate Summary"
    assert selected_games == []


def test_game_selection_resolves_selected_game_pk():
    slate = [
        {"away_team": "AAA", "home_team": "BBB", "game_pk": 1},
        {"away_team": "CCC", "home_team": "DDD", "game_pk": 2},
    ]

    selection, selected_games = resolve_logo_game_selection(slate, "2")

    assert selection == "CCC @ DDD"
    assert [game["game_pk"] for game in selected_games] == [2]


def test_add_metric_flags_marks_core_metrics():
    frame = pd.DataFrame(
        [
            {
                "launch_speed": 101.0,
                "launch_angle": 27.0,
                "bb_type": "fly_ball",
                "description": "swinging_strike",
                "estimated_woba_using_speedangle": 0.7,
                "release_speed": 95.0,
                "release_spin_rate": 2300.0,
                "hc_x": 90.0,
                "stand": "R",
                "balls": 1,
                "strikes": 2,
            }
        ]
    )
    enriched = add_metric_flags(frame)
    row = enriched.iloc[0]
    assert bool(row["is_barrel"])
    assert bool(row["is_fly_ball"])
    assert bool(row["is_hard_hit"])
    assert bool(row["is_swinging_strike"])


def test_hitter_hr_form_uses_rolling_composite_against_baseline():
    rows = []
    for game_idx in range(1, 16):
        hot = game_idx > 10
        rows.append(
            {
                "game_date": pd.Timestamp("2026-04-01") + pd.Timedelta(days=game_idx),
                "game_year": 2026,
                "game_pk": game_idx,
                "batter": 10,
                "pitcher": 20,
                "player_name": "Pitcher A",
                "pitch_type": "FF",
                "pitch_name": "4-Seam Fastball",
                "stand": "R",
                "p_throws": "R",
                "description": "hit_into_play",
                "events": "field_out",
                "launch_speed": 108.0 if hot else 90.0,
                "launch_angle": 22.0 if hot else 0.0,
                "bb_type": "fly_ball",
                "estimated_woba_using_speedangle": 0.4,
                "at_bat_number": game_idx,
                "pitch_number": 1,
                "balls": 0,
                "strikes": 0,
                "hc_x": 90.0,
                "release_speed": 95.0,
                "release_spin_rate": 2300.0,
            }
        )
    form = _build_hitter_hr_form(add_metric_flags(pd.DataFrame(rows)))

    assert form.shape[0] == 1
    assert form.loc[0, "batter"] == 10
    assert form.loc[0, "hr_form"].endswith("↑")
    assert 0.50 < form.loc[0, "hr_form_pct"] <= 0.95


def test_hitter_hr_form_sparse_data_returns_blank_display():
    rows = []
    for game_idx in range(1, 4):
        rows.append(
            {
                "game_date": pd.Timestamp("2026-04-01") + pd.Timedelta(days=game_idx),
                "game_year": 2026,
                "game_pk": game_idx,
                "batter": 10,
                "pitcher": 20,
                "player_name": "Pitcher A",
                "pitch_type": "FF",
                "pitch_name": "4-Seam Fastball",
                "stand": "R",
                "p_throws": "R",
                "description": "hit_into_play",
                "events": "field_out",
                "launch_speed": 100.0,
                "launch_angle": 22.0,
                "bb_type": "fly_ball",
                "estimated_woba_using_speedangle": 0.4,
                "at_bat_number": game_idx,
                "pitch_number": 1,
                "balls": 0,
                "strikes": 0,
                "hc_x": 90.0,
                "release_speed": 95.0,
                "release_spin_rate": 2300.0,
            }
        )
    form = _build_hitter_hr_form(add_metric_flags(pd.DataFrame(rows)))

    assert pd.isna(form.loc[0, "hr_form"])
    assert pd.isna(form.loc[0, "hr_form_pct"])


def test_hr_form_does_not_change_hitter_scores():
    base = pd.DataFrame(
        [
            {
                "hitter_name": "Test Hitter",
                "team": "AAA",
                "batter": 10,
                "stand": "R",
                "xwoba": 0.400,
                "swstr_pct": 0.10,
                "barrel_bbe_pct": 0.15,
                "barrel_bip_pct": 0.12,
                "pulled_barrel_pct": 0.10,
                "sweet_spot_pct": 0.35,
                "hard_hit_pct": 0.45,
                "avg_launch_angle": 22.0,
            }
        ]
    )
    with_form = base.assign(hr_form="72% ↑", hr_form_pct=0.72)

    scored_base = add_hitter_matchup_score(base)
    scored_with_form = add_hitter_matchup_score(with_form)

    pd.testing.assert_series_equal(scored_base["matchup_score"], scored_with_form["matchup_score"], check_names=False)
    pd.testing.assert_series_equal(scored_base["test_score"], scored_with_form["test_score"], check_names=False)


def test_khr_score_blends_matchup_and_hr_form_as_composite_score():
    hitters = pd.DataFrame(
        [
            {
                "batter": 1,
                "hitter_name": "Hot Form",
                "swstr_pct": 0.10,
                "barrel_bbe_pct": 0.15,
                "pulled_barrel_pct": 0.10,
                "sweet_spot_pct": 0.35,
                "avg_launch_angle": 22.0,
                "barrel_bip_pct": 0.12,
                "hard_hit_pct": 0.45,
                "xwoba": 0.400,
                "hr_form_pct": 0.80,
            },
            {
                "batter": 2,
                "hitter_name": "Missing Form",
                "swstr_pct": 0.16,
                "barrel_bbe_pct": 0.08,
                "pulled_barrel_pct": 0.04,
                "sweet_spot_pct": 0.25,
                "avg_launch_angle": 10.0,
                "barrel_bip_pct": 0.05,
                "hard_hit_pct": 0.32,
                "xwoba": 0.330,
                "hr_form_pct": pd.NA,
            },
        ]
    )

    scored = add_hitter_matchup_score(hitters).sort_values("batter").reset_index(drop=True)
    expected_form = pd.to_numeric(scored["hr_form_pct"], errors="coerce").mul(100.0).fillna(50.0)
    expected = ((scored["matchup_score"] * 0.60) + (expected_form * 0.40)).clip(lower=0.0, upper=100.0)

    pd.testing.assert_series_equal(scored["khr_score"], expected, check_names=False)
    assert scored["khr_score"].between(0, 100).all()


def test_hitter_iso_is_aggregated_from_extra_bases_per_at_bat():
    frame = pd.DataFrame(
        [
            {"game_year": 2026, "batter": 1, "stand": "R", "team": "AAA", "events": "single", "bb_type": "line_drive", "estimated_woba_using_speedangle": 0.9, "description": "hit_into_play", "launch_speed": 100.0, "launch_angle": 20.0, "hc_x": 90.0},
            {"game_year": 2026, "batter": 1, "stand": "R", "team": "AAA", "events": "double", "bb_type": "line_drive", "estimated_woba_using_speedangle": 1.2, "description": "hit_into_play", "launch_speed": 101.0, "launch_angle": 21.0, "hc_x": 90.0},
            {"game_year": 2026, "batter": 1, "stand": "R", "team": "AAA", "events": "home_run", "bb_type": "fly_ball", "estimated_woba_using_speedangle": 2.0, "description": "hit_into_play", "launch_speed": 105.0, "launch_angle": 28.0, "hc_x": 90.0},
            {"game_year": 2026, "batter": 1, "stand": "R", "team": "AAA", "events": "field_out", "bb_type": "ground_ball", "estimated_woba_using_speedangle": 0.0, "description": "hit_into_play", "launch_speed": 80.0, "launch_angle": -5.0, "hc_x": 90.0},
            {"game_year": 2026, "batter": 1, "stand": "R", "team": "AAA", "events": "walk", "bb_type": None, "estimated_woba_using_speedangle": pd.NA, "description": "ball", "launch_speed": pd.NA, "launch_angle": pd.NA, "hc_x": pd.NA},
        ]
    )
    frame["release_speed"] = 95.0
    frame["release_spin_rate"] = 2300.0
    frame["balls"] = 0
    frame["strikes"] = 0
    metrics = _aggregate_hitter_metrics(add_metric_flags(frame), "unweighted", {2026: 1.0})

    assert metrics.loc[0, "iso"] == 1.0


def test_pitcher_start_outcomes_use_first_pitcher_and_aggregate_starter_rows():
    frame = pd.DataFrame(
        [
            # Away offense, home fielding starter 10 faces two batters and throws three pitches.
            {"game_date": "2026-04-10", "game_year": 2026, "game_pk": 1, "fielding_team": "HOM", "pitcher": 10, "pitcher_name": "Home Starter", "at_bat_number": 1, "pitch_number": 1, "events": ""},
            {"game_date": "2026-04-10", "game_year": 2026, "game_pk": 1, "fielding_team": "HOM", "pitcher": 10, "pitcher_name": "Home Starter", "at_bat_number": 1, "pitch_number": 2, "events": "strikeout"},
            {"game_date": "2026-04-10", "game_year": 2026, "game_pk": 1, "fielding_team": "HOM", "pitcher": 10, "pitcher_name": "Home Starter", "at_bat_number": 2, "pitch_number": 1, "events": "walk"},
            {"game_date": "2026-04-10", "game_year": 2026, "game_pk": 1, "fielding_team": "HOM", "pitcher": 11, "pitcher_name": "Home Reliever", "at_bat_number": 3, "pitch_number": 1, "events": "field_out"},
            # Home offense, away fielding starter 20 is a separate starter for the same game.
            {"game_date": "2026-04-10", "game_year": 2026, "game_pk": 1, "fielding_team": "AWY", "pitcher": 20, "pitcher_name": "Away Starter", "at_bat_number": 1, "pitch_number": 1, "events": "field_out"},
        ]
    )

    outcomes = _build_pitcher_start_outcomes(frame, target_date=date(2026, 4, 11))

    home = outcomes.loc[outcomes["pitcher_id"].eq(10)].iloc[0]
    assert set(outcomes["pitcher_id"]) == {10, 20}
    assert 11 not in set(outcomes["pitcher_id"])
    assert home["pitch_count"] == 3
    assert home["batters_faced"] == 2
    assert home["strikeouts"] == 1
    assert home["walks"] == 1
    assert bool(home["started"])


def test_pitcher_start_outcomes_exclude_current_slate_and_filter_last_30():
    rows = []
    for idx in range(35):
        game_date = date(2026, 3, 1) + timedelta(days=idx)
        rows.append(
            {
                "slate_date": game_date,
                "game_date": game_date,
                "game_year": 2026,
                "game_pk": idx,
                "team": "AAA",
                "pitcher_id": 10,
                "pitcher_name": "Starter",
                "started": True,
                "batters_faced": 20,
                "pitch_count": 80 + idx,
                "strikeouts": 5,
                "walks": 2,
            }
        )
    rows.append({**rows[-1], "game_date": date(2026, 4, 5), "slate_date": date(2026, 4, 5), "game_pk": 999, "pitch_count": 120})
    outcomes = pd.DataFrame(rows)
    schedule = [{"away_probable_pitcher_id": 10, "home_probable_pitcher_id": 99}]

    filtered = _filter_pitcher_start_outcomes_for_slate(outcomes, schedule, date(2026, 4, 5), n_starts=30)

    assert len(filtered) == 30
    assert filtered["pitcher_id"].eq(10).all()
    assert pd.to_datetime(filtered["game_date"]).dt.date.max() < date(2026, 4, 5)
    assert 120 not in set(filtered["pitch_count"])


def test_hr_form_columns_follow_zone_fit_in_hitter_tables():
    for preset_columns in HITTER_PRESETS.values():
        assert "zone_fit_score" in preset_columns
        assert "hr_form" in preset_columns
        assert "khr_score" in preset_columns
        if "xwoba" in preset_columns:
            assert "iso" in preset_columns
            assert preset_columns.index("iso") == preset_columns.index("xwoba") - 1
        assert preset_columns.index("hr_form") == preset_columns.index("zone_fit_score") + 1
        assert preset_columns.index("khr_score") == preset_columns.index("hr_form") + 1
    assert BEST_MATCHUP_COLUMNS.index("hr_form") == BEST_MATCHUP_COLUMNS.index("zone_fit_score") + 1
    assert BEST_MATCHUP_COLUMNS.index("khr_score") == BEST_MATCHUP_COLUMNS.index("hr_form") + 1
    assert BEST_MATCHUP_COLUMNS.index("iso") == BEST_MATCHUP_COLUMNS.index("xwoba") - 1


def test_slate_summary_best_matchups_keeps_top_three_per_game():
    frame = pd.DataFrame(
        [
            {"game_pk": 1, "hitter_name": "A", "matchup_score": 70.0, "xwoba": 0.400},
            {"game_pk": 1, "hitter_name": "B", "matchup_score": 80.0, "xwoba": 0.390},
            {"game_pk": 1, "hitter_name": "C", "matchup_score": 60.0, "xwoba": 0.410},
            {"game_pk": 1, "hitter_name": "D", "matchup_score": 50.0, "xwoba": 0.420},
            {"game_pk": 2, "hitter_name": "E", "matchup_score": 90.0, "xwoba": 0.380},
            {"game_pk": 2, "hitter_name": "F", "matchup_score": 85.0, "xwoba": 0.390},
        ]
    )

    summary = build_slate_summary_best_matchups(frame, per_game=3)

    assert summary["hitter_name"].tolist() == ["B", "A", "C", "E", "F"]
    assert summary.groupby("game_pk").size().max() <= 3


def test_slate_summary_matchup_overview_is_one_row_per_game():
    frame = pd.DataFrame(
        [
            {"game": "AAA @ BBB", "game_pk": 1, "team": "AAA", "hitter_name": "A", "matchup_score": 70.0, "xwoba": 0.400},
            {"game": "AAA @ BBB", "game_pk": 1, "team": "BBB", "hitter_name": "B", "matchup_score": 80.0, "xwoba": 0.390},
            {"game": "AAA @ BBB", "game_pk": 1, "team": "AAA", "hitter_name": "C", "matchup_score": 60.0, "xwoba": 0.410},
            {"game": "CCC @ DDD", "game_pk": 2, "team": "CCC", "hitter_name": "D", "matchup_score": 90.0, "xwoba": 0.380},
        ]
    )

    overview = build_slate_summary_matchup_overview(frame, per_game=3)

    assert overview.shape[0] == 2
    assert overview.columns.tolist() == ["Game", "Top 1", "Top 2", "Top 3"]
    assert overview.loc[0, "Top 1"] == "BBB: B (80.0)"


def test_official_barrel_bands_expand_with_exit_velocity():
    frame = pd.DataFrame(
        [
            {"launch_speed": 98.0, "launch_angle": 26.0},
            {"launch_speed": 98.4, "launch_angle": 27.0},
            {"launch_speed": 98.0, "launch_angle": 25.0},
            {"launch_speed": 100.0, "launch_angle": 33.0},
            {"launch_speed": 100.0, "launch_angle": 34.0},
            {"launch_speed": 112.0, "launch_angle": 50.0},
            {"launch_speed": 112.0, "launch_angle": 51.0},
            {"launch_speed": 116.0, "launch_angle": 8.0},
            {"launch_speed": 116.0, "launch_angle": 51.0},
        ]
    )
    result = is_barrel(frame).tolist()
    assert result == [True, True, False, True, False, True, False, True, False]


def test_ohtani_two_way_exception_stays_in_hitter_pool():
    hitters = pd.DataFrame(
        [
            {"batter": 660271, "hitter_name": "Shohei Ohtani"},
            {"batter": 123456, "hitter_name": "Pitcher Hitter"},
            {"batter": 999999, "hitter_name": "Everyday Batter"},
        ]
    )
    exclusions = pd.DataFrame(
        [
            {"player_id": 660271, "pitcher_name": "Ohtani, Shohei", "exclude_from_hitter_tables": True},
            {"player_id": 123456, "pitcher_name": "Pitcher Hitter", "exclude_from_hitter_tables": True},
        ]
    )

    filtered = filter_excluded_pitchers_from_hitter_pool(hitters, exclusions)

    assert filtered["batter"].tolist() == [660271, 999999]


def test_test_score_boosts_zone_fit_without_changing_shape_formula():
    hitters = pd.DataFrame(
        [
            {
                "batter": 1,
                "hitter_name": "Launch Band Hitter",
                "swstr_pct": 0.12,
                "barrel_bbe_pct": 0.20,
                "pulled_barrel_pct": 0.18,
                "sweet_spot_pct": 0.38,
                "avg_launch_angle": 22.0,
                "barrel_bip_pct": 0.15,
                "hard_hit_pct": 0.48,
                "xwoba": 0.410,
            },
            {
                "batter": 2,
                "hitter_name": "Low Launch Hitter",
                "swstr_pct": 0.18,
                "barrel_bbe_pct": 0.10,
                "pulled_barrel_pct": 0.08,
                "sweet_spot_pct": 0.24,
                "avg_launch_angle": 4.0,
                "barrel_bip_pct": 0.08,
                "hard_hit_pct": 0.36,
                "xwoba": 0.330,
            },
        ]
    )

    scored = add_hitter_matchup_score(hitters)
    pulled_barrel_scale = normalize_series(scored["pulled_barrel_pct"])
    pulled_barrel_bonus = ((pulled_barrel_scale - 0.5).clip(lower=0.0) / 0.5) * 0.08
    expected_test_score = ((
        (normalize_series(scored["swstr_pct"], inverse=True) * 0.325)
        + (normalize_series(scored["barrel_bbe_pct"]) * 0.30)
        + (scored["shape_score"] * 0.20)
        + (scored["zone_fit_score"] * 0.175)
    ) * 100.0 * (1.0 + pulled_barrel_bonus)).clip(lower=0.0, upper=100.0)

    pd.testing.assert_series_equal(
        scored["test_shape_score"],
        scored["shape_score"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        scored["test_score"],
        expected_test_score,
        check_names=False,
    )


def test_pitch_level_rolling_windows_are_complete():
    rows = []
    for game_idx in range(1, 17):
        rows.append(
            {
                "batter": 100,
                "player_name": "Rolling Hitter",
                "pitcher": 200,
                "pitcher_name": "Rolling Pitcher",
                "game_pk": game_idx,
                "game_date": pd.Timestamp("2026-04-01") + pd.Timedelta(days=game_idx),
                "game_year": 2026,
                "stand": "R",
                "p_throws": "R",
                "pitch_type": "FF",
                "pitch_name": "4-Seam Fastball",
                "bb_type": "fly_ball" if game_idx % 3 == 0 else "ground_ball",
                "description": "swinging_strike" if game_idx % 5 == 0 else "called_strike",
                "launch_speed": 101.0 if game_idx % 4 == 0 else 96.0,
                "launch_angle": 26.0 if game_idx % 4 == 0 else 10.0 + game_idx,
                "hc_x": 90.0,
                "estimated_woba_using_speedangle": 0.300 + (game_idx / 1000.0),
                "release_speed": 91.0 + (game_idx / 10.0),
                "release_spin_rate": 2300.0,
                "at_bat_number": game_idx,
                "pitch_number": 1,
                "balls": 0,
                "strikes": 1,
                "is_tracked_bbe": True,
                "is_barrel": game_idx % 4 == 0,
                "is_pulled_barrel": game_idx % 8 == 0,
                "is_hard_hit": game_idx % 2 == 0,
                "is_fly_ball": game_idx % 3 == 0,
                "is_batted_ball": True,
                "launch_angle_value": 10.0 + game_idx,
                "xwoba_value": 0.300 + (game_idx / 1000.0),
                "release_speed_value": 91.0 + (game_idx / 10.0),
            }
        )
    live_pitch_mix = pd.DataFrame(rows)

    hitter_rolling = compute_hitter_rolling(live_pitch_mix)
    pitcher_rolling = compute_pitcher_rolling(live_pitch_mix)

    assert hitter_rolling["rolling_window"].tolist() == ["Rolling 5", "Rolling 10", "Rolling 15"]
    assert pitcher_rolling["rolling_window"].tolist() == ["Rolling 5", "Rolling 10", "Rolling 15"]
    assert set(hitter_rolling["games_in_window"].tolist()) == {5, 10, 15}
    assert set(pitcher_rolling["games_in_window"].tolist()) == {5, 10, 15}
    assert hitter_rolling[["pulled_barrel_pct", "barrel_bip_pct", "hard_hit_pct", "fb_pct", "avg_launch_angle", "xwoba"]].notna().all().all()
    assert pitcher_rolling[["avg_release_speed", "barrel_bip_pct", "hard_hit_pct", "fb_pct", "avg_launch_angle"]].notna().all().all()
    assert pitcher_rolling["player_name"].eq("Rolling Pitcher").all()


def test_local_source_payload_respects_target_date(tmp_path):
    config = AppConfig(workspace=tmp_path, csv_dir=tmp_path, artifacts_dir=tmp_path / "artifacts", db_path=tmp_path / "artifacts" / "statcast.duckdb")
    rows = [
        {"game_date": "2026-04-01", "game_year": 2026, "game_pk": 1, "batter": 10, "pitcher": 20, "player_name": "Pitcher A", "pitch_type": "FF", "pitch_name": "4-Seam Fastball", "stand": "R", "p_throws": "R", "description": "swinging_strike", "launch_speed": 100.0, "launch_angle": 25.0, "bb_type": "fly_ball", "estimated_woba_using_speedangle": 0.6, "at_bat_number": 1, "pitch_number": 1},
        {"game_date": "2026-04-02", "game_year": 2026, "game_pk": 2, "batter": 10, "pitcher": 20, "player_name": "Pitcher A", "pitch_type": "FF", "pitch_name": "4-Seam Fastball", "stand": "R", "p_throws": "R", "description": "called_strike", "launch_speed": 101.0, "launch_angle": 26.0, "bb_type": "fly_ball", "estimated_woba_using_speedangle": 0.7, "at_bat_number": 1, "pitch_number": 1},
    ]
    write_statcast_events(config, pd.DataFrame(rows))

    payload = load_local_source_payload(config, target_date=pd.Timestamp("2026-04-01").date())

    assert payload.live_pitch_mix["game_pk"].tolist() == [1]
    assert payload.hitter_rolling["games_in_window"].max() == 1


def test_local_tracking_upsert_is_idempotent(tmp_path):
    config = AppConfig(workspace=tmp_path, csv_dir=tmp_path, artifacts_dir=tmp_path / "artifacts", db_path=tmp_path / "artifacts" / "statcast.duckdb")
    snapshots = pd.DataFrame(
        [
            {"slate_date": pd.Timestamp("2026-04-01").date(), "game_pk": 1, "batter_id": 10, "split_key": "overall", "recent_window": "season", "weighted_mode": "weighted", "matchup_score": 50.0},
            {"slate_date": pd.Timestamp("2026-04-01").date(), "game_pk": 1, "batter_id": 10, "split_key": "overall", "recent_window": "season", "weighted_mode": "weighted", "matchup_score": 60.0},
        ]
    )

    write_tracking_payload(config, snapshots, pd.DataFrame(), pd.DataFrame())
    write_tracking_payload(config, snapshots.tail(1), pd.DataFrame(), pd.DataFrame())
    loaded = read_hitter_snapshots_for_date(config, pd.Timestamp("2026-04-01").date())

    assert len(loaded) == 1
    assert float(loaded["matchup_score"].iloc[0]) == 60.0


def test_local_odds_cache_history_and_latest(tmp_path):
    config = AppConfig(workspace=tmp_path, csv_dir=tmp_path, artifacts_dir=tmp_path / "artifacts", db_path=tmp_path / "artifacts" / "statcast.duckdb")
    odds = pd.DataFrame(
        [
            {"fetched_at": "2026-04-01T10:00:00Z", "cache_key": "old", "provider": "test", "event_id": "e1", "commence_time": "2026-04-02T00:00:00Z", "away_team": "A", "home_team": "B", "sportsbook": "Book", "sportsbook_key": "book", "market_key": "batter_home_runs", "market": "HR", "player_name": "Batter One", "odds_american": 300, "line": 0.5, "player_event_market_key": "p1"},
            {"fetched_at": "2026-04-01T11:00:00Z", "cache_key": "new", "provider": "test", "event_id": "e1", "commence_time": "2026-04-02T00:00:00Z", "away_team": "A", "home_team": "B", "sportsbook": "Book", "sportsbook_key": "book", "market_key": "batter_home_runs", "market": "HR", "player_name": "Batter One", "odds_american": 280, "line": 0.5, "player_event_market_key": "p1"},
        ]
    )

    write_props_odds_snapshot(config, odds)
    history = read_prop_odds_history(config, pd.Timestamp("2026-04-02").date(), pd.Timestamp("2026-04-02").date(), ("batter_home_runs",))
    latest = read_latest_prop_odds_snapshot(config, pd.Timestamp("2026-04-02").date(), ("batter_home_runs",))

    assert len(history) == 2
    assert latest["cache_key"].unique().tolist() == ["new"]


def test_exit_velo_reader_keeps_last_25_games_per_batter(tmp_path):
    config = AppConfig(workspace=tmp_path, csv_dir=tmp_path, artifacts_dir=tmp_path / "artifacts", db_path=tmp_path / "artifacts" / "statcast.duckdb")
    rows = []
    for idx in range(30):
        rows.append(
            {
                "game_date": (pd.Timestamp("2026-04-01") + pd.Timedelta(days=idx)).date().isoformat(),
                "game_year": 2026,
                "game_pk": idx + 1,
                "batter": 10,
                "batter_name": "EV Hitter",
                "pitcher": 20,
                "player_name": "EV Pitcher",
                "pitcher_name": "EV Pitcher",
                "home_team": "H",
                "away_team": "A",
                "inning_topbot": "Top",
                "stand": "R",
                "p_throws": "L",
                "pitch_type": "FF",
                "pitch_name": "4-Seam Fastball",
                "launch_speed": 95.0 + idx,
                "launch_angle": 10.0 + idx,
                "bb_type": "line_drive",
                "estimated_woba_using_speedangle": 0.4,
                "at_bat_number": 1,
                "pitch_number": 1,
            }
        )
    write_statcast_events(config, pd.DataFrame(rows))

    events = read_hitter_exit_velo_events(config)

    assert events["game_pk"].nunique() == 25
    assert events["game_pk"].min() == 6
    assert events["team"].eq("A").all()


def test_per_game_artifacts_are_scoped_to_game(tmp_path):
    config = AppConfig(workspace=tmp_path, csv_dir=tmp_path, artifacts_dir=tmp_path / "artifacts", db_path=tmp_path / "artifacts" / "statcast.duckdb")
    context = BuildContext(config=config, target_date=pd.Timestamp("2026-04-15").date(), csv_dir=tmp_path)
    schedule = [
        {"game_pk": 1, "away_team": "AAA", "home_team": "BBB", "away_probable_pitcher_id": 20, "home_probable_pitcher_id": 21},
        {"game_pk": 2, "away_team": "CCC", "home_team": "DDD", "away_probable_pitcher_id": 30, "home_probable_pitcher_id": 31},
    ]
    rosters = [
        {"team": "AAA", "player_id": 10, "player_name": "Away Hitter"},
        {"team": "BBB", "player_id": 11, "player_name": "Home Hitter"},
        {"team": "CCC", "player_id": 12, "player_name": "Other Hitter"},
    ]
    hitter_snapshots = pd.DataFrame(
        [
            {"game_pk": 1, "team": "AAA", "batter_id": 10, "hitter_name": "Away Hitter", "matchup_score": 70},
            {"game_pk": 2, "team": "CCC", "batter_id": 12, "hitter_name": "Other Hitter", "matchup_score": 60},
        ]
    )
    pitcher_snapshots = pd.DataFrame(
        [
            {"game_pk": 1, "team": "AAA", "pitcher_id": 20, "pitcher_name": "Away Pitcher", "pitcher_score": 55},
            {"game_pk": 2, "team": "CCC", "pitcher_id": 30, "pitcher_name": "Other Pitcher", "pitcher_score": 50},
        ]
    )
    board_winners = pd.DataFrame([{"game_pk": 1, "board_name": "Best", "board_rank": 1, "batter_id": 10}])
    hitter_rolling = pd.DataFrame([{"player_name": "Away Hitter", "rolling_window": "Rolling 5"}])
    pitcher_rolling = pd.DataFrame([{"player_name": "Away Pitcher", "rolling_window": "Rolling 5"}])
    pitcher_summary = pd.DataFrame([{"pitcher_id": 20, "batter_side_key": "all"}, {"pitcher_id": 30, "batter_side_key": "all"}])
    pitcher_arsenal = pd.DataFrame([{"pitcher_id": 20, "pitch_name": "Fastball"}, {"pitcher_id": 30, "pitch_name": "Slider"}])
    pitcher_arsenal_by_hand = pd.DataFrame([{"pitcher_id": 20, "batter_side_key": "vs_lhh", "pitch_name": "Fastball"}])
    pitcher_usage = pd.DataFrame([{"pitcher_id": 20, "batter_side_key": "all", "pitch_name": "Fastball"}])
    batter_zones = pd.DataFrame([{"batter_id": 10, "zone": 5}, {"batter_id": 12, "zone": 6}])
    pitcher_zones = pd.DataFrame([{"pitcher_id": 20, "zone": 5}, {"pitcher_id": 30, "zone": 6}])
    batter_family = pd.DataFrame([{"batter_id": 10, "pitch_family": "fastball"}, {"batter_id": 12, "pitch_family": "breaking"}])
    pitcher_family = pd.DataFrame([{"pitcher_id": 20, "pitch_family": "fastball"}, {"pitcher_id": 30, "pitch_family": "breaking"}])
    pitcher_movement = pd.DataFrame([{"pitcher_id": 20, "pitch_name": "Fastball"}, {"pitcher_id": 30, "pitch_name": "Slider"}])

    _save_per_game_files(
        context,
        schedule,
        rosters,
        hitter_snapshots,
        pitcher_snapshots,
        board_winners,
        hitter_rolling,
        pitcher_rolling,
        pitcher_summary,
        pitcher_arsenal,
        pitcher_arsenal_by_hand,
        pitcher_usage,
        batter_zones,
        pitcher_zones,
        batter_family,
        pitcher_family,
        pitcher_movement,
    )

    game_dir = config.daily_dir / "2026-04-15" / "games" / "1"
    matchup = pd.read_parquet(game_dir / "matchup.parquet")
    zones = pd.read_parquet(game_dir / "zones.parquet")
    detail = pd.read_parquet(game_dir / "pitcher_detail.parquet")

    assert {"matchup.parquet", "rolling.parquet", "pitcher_detail.parquet", "zones.parquet", "pitch_shape.parquet", "exports.parquet"}.issubset({path.name for path in game_dir.iterdir()})
    assert set(matchup["game_pk"].dropna().astype(int)) == {1}
    assert 12 not in set(pd.to_numeric(zones.get("batter_id"), errors="coerce").dropna().astype(int))
    assert 30 not in set(pd.to_numeric(detail.get("pitcher_id"), errors="coerce").dropna().astype(int))


def test_top_slate_board_artifacts_include_all_filter_combinations(tmp_path):
    config = AppConfig(workspace=tmp_path, csv_dir=tmp_path, artifacts_dir=tmp_path / "artifacts", db_path=tmp_path / "artifacts" / "statcast.duckdb")
    context = BuildContext(config=config, target_date=pd.Timestamp("2026-04-15").date(), csv_dir=tmp_path)
    hitter_rows = []
    pitcher_rows = []
    for game_pk, game_label, hitter_name, pitcher_name in [
        (1, "AAA @ BBB", "Hitter One", "Pitcher One"),
        (2, "CCC @ DDD", "Hitter Two", "Pitcher Two"),
    ]:
        for split_key in DEFAULT_SPLITS:
            for recent_window in DEFAULT_RECENT_WINDOWS:
                for weighted_mode in ("weighted", "unweighted"):
                    hitter_rows.append(
                        {
                            "game_label": game_label,
                            "game_pk": game_pk,
                            "team": game_label[:3],
                            "opponent": game_label[-3:],
                            "hitter_name": hitter_name,
                            "batter_id": game_pk * 10,
                            "opposing_pitcher_name": pitcher_name,
                            "split_key": split_key,
                            "recent_window": recent_window,
                            "weighted_mode": weighted_mode,
                            "matchup_score": 70.0,
                            "test_score": 68.0,
                            "ceiling_score": 65.0,
                            "zone_fit_score": 0.62,
                            "hr_form": "72% ↑",
                            "khr_score": 70.8,
                            "hr_form_pct": 0.72,
                            "iso": 0.21,
                            "xwoba": 0.4,
                            "pitch_count": 100,
                            "bip": 40,
                        }
                    )
                    pitcher_rows.append(
                        {
                            "game_label": game_label,
                            "game_pk": game_pk,
                            "team": game_label[:3],
                            "opponent": game_label[-3:],
                            "pitcher_name": pitcher_name,
                            "pitcher_id": game_pk * 20,
                            "p_throws": "R",
                            "split_key": split_key,
                            "recent_window": recent_window,
                            "weighted_mode": weighted_mode,
                            "pitcher_score": 60.0,
                            "strikeout_score": 55.0,
                            "xwoba": 0.31,
                            "csw_pct": 0.29,
                            "swstr_pct": 0.12,
                        }
                    )

    _save_top_slate_board_files(context, pd.DataFrame(hitter_rows), pd.DataFrame(pitcher_rows))

    top_hitters = pd.read_parquet(config.daily_dir / "2026-04-15" / "top_slate_hitters.parquet")
    top_pitchers = pd.read_parquet(config.daily_dir / "2026-04-15" / "top_slate_pitchers.parquet")
    filtered_hitters = pd.read_parquet(config.daily_dir / "2026-04-15" / "top_boards" / "overall__season__weighted__hitters.parquet")
    filtered_pitchers = pd.read_parquet(config.daily_dir / "2026-04-15" / "top_boards" / "overall__season__weighted__pitchers.parquet")
    slate_summary = pd.read_parquet(config.daily_dir / "2026-04-15" / "slate_summary.parquet")
    expected_combos = len(DEFAULT_SPLITS) * len(DEFAULT_RECENT_WINDOWS) * 2

    assert top_hitters[["split_key", "recent_window", "weighted_mode"]].drop_duplicates().shape[0] == expected_combos
    assert top_pitchers[["split_key", "recent_window", "weighted_mode"]].drop_duplicates().shape[0] == expected_combos
    assert filtered_hitters[["split_key", "recent_window", "weighted_mode"]].drop_duplicates().to_dict("records") == [
        {"split_key": "overall", "recent_window": "season", "weighted_mode": "weighted"}
    ]
    assert filtered_pitchers[["split_key", "recent_window", "weighted_mode"]].drop_duplicates().to_dict("records") == [
        {"split_key": "overall", "recent_window": "season", "weighted_mode": "weighted"}
    ]
    assert {"split_key", "recent_window", "weighted_mode", "Game", "Top 1"}.issubset(slate_summary.columns)
    assert top_hitters["game_pk"].nunique() == 2
    assert top_pitchers["game_pk"].nunique() == 2
    assert {"game", "hitter_name", "matchup_score", "test_score", "ceiling_score", "hr_form", "khr_score", "hr_form_pct", "iso"}.issubset(top_hitters.columns)
    assert {"game", "pitcher_name", "pitcher_score", "strikeout_score"}.issubset(top_pitchers.columns)
