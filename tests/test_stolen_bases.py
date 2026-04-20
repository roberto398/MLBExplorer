from __future__ import annotations

from datetime import date

import pandas as pd

from mlb_dashboard.config import AppConfig
from mlb_dashboard.stolen_bases import (
    build_daily_stolen_base_board,
    build_stolen_base_profiles,
    parse_stolen_base_events,
)


def _sample_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_date": "2026-04-01",
                "game_pk": 1,
                "away_team": "TEX",
                "home_team": "ATH",
                "inning_topbot": "Top",
                "on_1b": 10,
                "on_2b": pd.NA,
                "on_3b": pd.NA,
                "pitcher": 20,
                "fielder_2": 30,
                "des": "Joc Pederson steals (1) 2nd base.",
                "at_bat_number": 1,
                "pitch_number": 2,
                "batter": 99,
                "batter_name": "Other",
                "player_name": "Pitcher A",
            },
            {
                "game_date": "2026-04-02",
                "game_pk": 2,
                "away_team": "TEX",
                "home_team": "ATH",
                "inning_topbot": "Top",
                "on_1b": 10,
                "on_2b": pd.NA,
                "on_3b": pd.NA,
                "pitcher": 20,
                "fielder_2": 30,
                "des": "Joc Pederson caught stealing 2nd base, catcher X to shortstop Y.",
                "at_bat_number": 2,
                "pitch_number": 1,
                "batter": 99,
                "batter_name": "Other",
                "player_name": "Pitcher A",
            },
        ]
    )


def test_parse_stolen_base_events_dedupes_and_attributes_runner() -> None:
    events = pd.concat([_sample_events(), _sample_events().head(1)], ignore_index=True)
    rosters = pd.DataFrame([{"team": "TEX", "player_id": 10, "player_name": "Joc Pederson"}])

    parsed = parse_stolen_base_events(events, rosters)

    assert len(parsed) == 2
    assert parsed["runner_id"].tolist() == [10, 10]
    assert parsed["event_type"].tolist() == ["sb", "cs"]
    assert parsed["is_stolen_base"].sum() == 1
    assert parsed["is_caught_stealing"].sum() == 1


def test_stolen_base_board_scores_are_clamped() -> None:
    events = _sample_events()
    rosters = pd.DataFrame([{"team": "TEX", "player_id": 10, "player_name": "Joc Pederson"}])
    hitter_metrics = pd.DataFrame(
        [
            {
                "batter": 10,
                "hitter_name": "Joc Pederson",
                "split_key": "overall",
                "recent_window": "season",
                "weighted_mode": "weighted",
                "xwoba": 0.360,
            }
        ]
    )
    schedule = [
        {
            "game_pk": 3,
            "away_team": "TEX",
            "home_team": "ATH",
            "away_probable_pitcher_id": 40,
            "away_probable_pitcher_name": "Away P",
            "home_probable_pitcher_id": 20,
            "home_probable_pitcher_name": "Pitcher A",
        }
    ]
    player, pitcher, catcher, team, log = build_stolen_base_profiles(events, rosters)

    board, _ = build_daily_stolen_base_board(
        AppConfig(),
        date(2026, 4, 3),
        schedule,
        rosters,
        hitter_metrics,
        player,
        pitcher,
        catcher,
        team,
        log,
        {},
    )

    assert not board.empty
    assert board["sb_score"].between(0, 100).all()
