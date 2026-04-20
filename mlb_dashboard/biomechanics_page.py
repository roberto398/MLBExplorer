"""Admin-only OpenBiomechanics research page for Kasper."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from .biomechanics_ingest import HITTING_OUTPUT, MANIFEST_OUTPUT, METADATA_OUTPUT, OUTPUT_DIR_NAME, PITCHING_OUTPUT
from .branding import apply_branding_head, page_icon_path
from .config import AppConfig
from .ui_components import render_custom_metric_table

try:
    import altair as alt

    HAS_ALTAIR = True
except ImportError:  # pragma: no cover
    alt = None
    HAS_ALTAIR = False


PITCHING_METRICS = {
    "pitch_speed_mph": "Pitch Speed",
    "arm_slot": "Arm Slot",
    "stride_length": "Stride Length",
    "stride_angle": "Stride Angle",
    "max_rotation_hip_shoulder_separation": "Max H-S Sep",
    "max_shoulder_external_rotation": "Max Shoulder ER",
    "max_shoulder_internal_rotational_velo": "Shoulder IR Velo",
    "max_elbow_extension_velo": "Elbow Ext Velo",
    "elbow_varus_moment": "Elbow Varus",
    "torso_rotation_br": "Torso Rot BR",
    "max_pelvis_rotational_velo": "Pelvis Rot Velo",
    "lead_grf_z_max": "Lead GRF Z",
    "rear_grf_z_max": "Rear GRF Z",
}

HITTING_METRICS = {
    "exit_velo_mph_x": "Exit Velo",
    "blast_bat_speed_mph_x": "Blast Bat Speed",
    "bat_speed_mph_contact_x": "Bat Speed Contact",
    "bat_speed_mph_max_x": "Max Bat Speed",
    "attack_angle_contact_x": "Attack Angle",
    "hand_speed_mag_max_x": "Hand Speed",
    "sweet_spot_velo_mph_contact_x": "Sweet Spot X",
    "sweet_spot_velo_mph_contact_y": "Sweet Spot Y",
    "sweet_spot_velo_mph_contact_z": "Sweet Spot Z",
    "bat_torso_angle_connection_x": "Early Connection",
    "x_factor_fp_x": "X-Factor FP",
    "pelvis_angular_velocity_seq_max_x": "Pelvis Seq Velo",
    "torso_angle_fp_z": "Torso FP Z",
}

LICENSE_TEXT = """
OpenBiomechanics data is provided by Driveline Baseball Research & Development under CC BY-NC-SA 4.0 with additional usage restrictions.

This Kasper page is admin-only and intended for exploratory/research review. The data is anonymized, so it should not be treated as MLB-player-specific information, and it is not wired into matchup, kHR, Strikeouts, or betting logic.
"""


def _hosted_base_url() -> str:
    return os.getenv("MLB_HOSTED_BASE_URL", "").rstrip("/")


def _require_admin_password() -> bool:
    required = st.secrets.get("ADMIN_PASSWORD")
    if not required:
        st.error("Bio Mechanics is locked. Add ADMIN_PASSWORD to Streamlit secrets to enable this page.")
        return False
    entry = st.text_input("Bio Mechanics password", type="password")
    if not entry:
        st.info("Enter the Bio Mechanics password to view this page.")
        return False
    if str(entry) != str(required):
        st.error("Incorrect password.")
        return False
    return True


@st.cache_data(show_spinner=False)
def _read_local_parquet(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(file_path)


@st.cache_data(show_spinner=False)
def _read_remote_parquet(url: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(url)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def _read_local_manifest(path: str) -> dict[str, object]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_data(show_spinner=False)
def _read_remote_manifest(url: str) -> dict[str, object]:
    try:
        return json.loads(pd.read_json(url, typ="series").to_json())
    except Exception:
        try:
            import requests

            response = requests.get(url, timeout=20)
            response.raise_for_status()
            return response.json()
        except Exception:
            return {}


def _load_artifacts(config: AppConfig) -> tuple[dict[str, pd.DataFrame], dict[str, object], str]:
    local_dir = config.artifacts_dir / OUTPUT_DIR_NAME
    local_pitching = local_dir / PITCHING_OUTPUT
    local_hitting = local_dir / HITTING_OUTPUT
    if local_pitching.exists() or local_hitting.exists():
        return (
            {
                "pitching": _read_local_parquet(str(local_pitching)),
                "hitting": _read_local_parquet(str(local_hitting)),
                "metadata": _read_local_parquet(str(local_dir / METADATA_OUTPUT)),
            },
            _read_local_manifest(str(local_dir / MANIFEST_OUTPUT)),
            "local",
        )

    base_url = _hosted_base_url()
    if base_url:
        remote_dir = f"{base_url}/{OUTPUT_DIR_NAME}"
        return (
            {
                "pitching": _read_remote_parquet(f"{remote_dir}/{PITCHING_OUTPUT}"),
                "hitting": _read_remote_parquet(f"{remote_dir}/{HITTING_OUTPUT}"),
                "metadata": _read_remote_parquet(f"{remote_dir}/{METADATA_OUTPUT}"),
            },
            _read_remote_manifest(f"{remote_dir}/{MANIFEST_OUTPUT}"),
            "hosted",
        )

    return ({"pitching": pd.DataFrame(), "hitting": pd.DataFrame(), "metadata": pd.DataFrame()}, {}, "missing")


def _numeric_columns(frame: pd.DataFrame, metric_map: dict[str, str]) -> list[str]:
    preferred = [column for column in metric_map if column in frame.columns]
    fallback = [
        column
        for column in frame.columns
        if column not in preferred
        and pd.to_numeric(frame[column], errors="coerce").notna().any()
    ]
    return preferred + fallback[:20]


def _display_columns(frame: pd.DataFrame, metric_map: dict[str, str], id_columns: list[str]) -> list[str]:
    columns = [column for column in id_columns if column in frame.columns]
    columns.extend([column for column in metric_map if column in frame.columns and column not in columns])
    return columns[:18]


def _label_frame(frame: pd.DataFrame, metric_map: dict[str, str]) -> pd.DataFrame:
    return frame.rename(columns={column: metric_map.get(column, column) for column in frame.columns})


def _filter_multiselect(frame: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    if column not in frame.columns:
        return frame
    options = sorted(frame[column].dropna().astype(str).unique().tolist())
    if not options:
        return frame
    selected = st.multiselect(label, options, default=options)
    if not selected:
        return frame.iloc[0:0].copy()
    return frame.loc[frame[column].astype(str).isin(selected)].copy()


def _render_percentile_cards(frame: pd.DataFrame, metric_columns: list[str], metric_map: dict[str, str]) -> None:
    present = [column for column in metric_columns if column in frame.columns][:4]
    if not present:
        st.info("No numeric metrics are available for this selection.")
        return
    cols = st.columns(len(present))
    for col, metric in zip(cols, present):
        numeric = pd.to_numeric(frame[metric], errors="coerce").dropna()
        if numeric.empty:
            value = "-"
            caption = "No values"
        else:
            value = f"{numeric.median():.1f}"
            caption = f"P10 {numeric.quantile(0.10):.1f} | P90 {numeric.quantile(0.90):.1f}"
        col.metric(metric_map.get(metric, metric), value, caption)


def _render_scatter(frame: pd.DataFrame, metric_columns: list[str], metric_map: dict[str, str], key_prefix: str) -> None:
    if len(metric_columns) < 2:
        st.info("Need at least two numeric metrics for a scatterplot.")
        return
    default_x = 0
    default_y = 1
    labels = {metric_map.get(column, column): column for column in metric_columns}
    x_label = st.selectbox("X metric", list(labels), index=default_x, key=f"{key_prefix}-x")
    y_label = st.selectbox("Y metric", list(labels), index=default_y, key=f"{key_prefix}-y")
    x_col = labels[x_label]
    y_col = labels[y_label]
    chart_data = frame[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
    if chart_data.empty:
        st.info("No chartable rows for this metric pair.")
        return
    if HAS_ALTAIR:
        chart = (
            alt.Chart(chart_data.rename(columns={x_col: x_label, y_col: y_label}))
            .mark_circle(size=48, opacity=0.65)
            .encode(x=alt.X(x_label, scale=alt.Scale(zero=False)), y=alt.Y(y_label, scale=alt.Scale(zero=False)))
            .interactive()
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.scatter_chart(chart_data.rename(columns={x_col: x_label, y_col: y_label}), x=x_label, y=y_label)


def _render_correlation_table(frame: pd.DataFrame, metric_columns: list[str], metric_map: dict[str, str], key: str) -> None:
    present = [column for column in metric_columns if column in frame.columns][:12]
    if len(present) < 2:
        st.info("Need at least two numeric metrics for correlations.")
        return
    numeric = frame[present].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr().round(2).rename(index=metric_map, columns=metric_map).reset_index().rename(columns={"index": "Metric"})
    render_custom_metric_table(corr, key=key, height=360)


def _render_dataset_tab(frame: pd.DataFrame, metric_map: dict[str, str], id_columns: list[str], title: str, key_prefix: str) -> None:
    st.subheader(title)
    if frame.empty:
        st.warning("Biomechanics artifacts are not published yet. Run biomechanics ingest and publish artifacts.")
        return

    filtered = frame.copy()
    filter_cols = [
        column
        for column in ("p_throws", "pitch_type", "side", "bat_side", "dataset")
        if column in filtered.columns
    ]
    for column in filter_cols:
        filtered = _filter_multiselect(filtered, column, column.replace("_", " ").title())

    metric_columns = _numeric_columns(filtered, metric_map)
    _render_percentile_cards(filtered, metric_columns, metric_map)

    chart_col, corr_col = st.columns([1, 1])
    with chart_col:
        st.markdown("#### Metric Scatter")
        _render_scatter(filtered, metric_columns, metric_map, key_prefix)
    with corr_col:
        st.markdown("#### Metric Correlations")
        _render_correlation_table(filtered, metric_columns, metric_map, f"{key_prefix}-corr")

    st.markdown("#### Rows")
    display_columns = _display_columns(filtered, metric_map, id_columns)
    display = _label_frame(filtered[display_columns].head(500), metric_map)
    render_custom_metric_table(display, key=f"{key_prefix}-rows", height=460)


def _render_overview(bundle: dict[str, pd.DataFrame], manifest: dict[str, object], source: str) -> None:
    pitching = bundle["pitching"]
    hitting = bundle["hitting"]
    metadata = bundle["metadata"]
    st.warning("Admin research only. OpenBiomechanics is anonymized and has non-commercial/additional usage restrictions.")
    cols = st.columns(4)
    cols[0].metric("Source", source)
    cols[1].metric("Pitching POI", f"{len(pitching):,}")
    cols[2].metric("Hitting POI", f"{len(hitting):,}")
    cols[3].metric("Metadata Rows", f"{len(metadata):,}")
    if manifest:
        st.caption(f"Manifest created: {manifest.get('created_at_utc', '-')}")
    st.markdown(
        """
        This page is a research surface for anonymized motion-capture point-of-interest metrics. It is intentionally separate from slate scoring and betting projections.
        """
    )


def _render_metric_dictionary() -> None:
    rows = []
    for group, metric_map in (("Pitching", PITCHING_METRICS), ("Hitting", HITTING_METRICS)):
        for column, label in metric_map.items():
            rows.append({"Group": group, "Column": column, "Display": label})
    render_custom_metric_table(pd.DataFrame(rows), key="biomech-metric-dictionary", height=620)


def main() -> None:
    st.set_page_config(page_title="Bio Mechanics", page_icon=page_icon_path(), layout="wide")
    apply_branding_head()
    st.title("Bio Mechanics")
    if not _require_admin_password():
        return

    config = AppConfig()
    bundle, manifest, source = _load_artifacts(config)

    tabs = st.tabs(["Overview", "Pitching", "Hitting", "Metric Dictionary", "License"])
    with tabs[0]:
        _render_overview(bundle, manifest, source)
    with tabs[1]:
        _render_dataset_tab(
            bundle["pitching"],
            PITCHING_METRICS,
            ["session_pitch", "session", "p_throws", "pitch_type", "pitch_speed_mph"],
            "Pitching Biomechanics",
            "bio-pitching",
        )
    with tabs[2]:
        _render_dataset_tab(
            bundle["hitting"],
            HITTING_METRICS,
            ["session_swing", "session", "exit_velo_mph_x"],
            "Hitting Biomechanics",
            "bio-hitting",
        )
    with tabs[3]:
        _render_metric_dictionary()
    with tabs[4]:
        st.markdown(LICENSE_TEXT)
        st.markdown("[OpenBiomechanics on GitHub](https://github.com/drivelineresearch/openbiomechanics)")


if __name__ == "__main__":
    main()
