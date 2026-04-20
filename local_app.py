from __future__ import annotations

import streamlit as st


def _render_matchups() -> None:
    from mlb_dashboard.local_app import main

    main()


def main() -> None:
    pages = {
        "All User Pages": [
            st.Page(_render_matchups, title="Kasper Matchups", url_path="kasper-matchups", default=True),
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
