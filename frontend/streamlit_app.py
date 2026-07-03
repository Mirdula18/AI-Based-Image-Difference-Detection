"""Stage 10: Frontend.

Simple upload UI for two drawing files that calls the FastAPI /compare
endpoint and displays all six required outputs: original A, original B,
diff visualization, highlighted regions, statistics, and the AI summary.

Run with:
    streamlit run frontend/streamlit_app.py
"""
import os

import requests
import streamlit as st

API_URL = os.environ.get("DIFF_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Drawing Difference Detection", layout="wide")
st.title("Drawing / Image Difference Detection")
st.caption("Upload two revisions of a drawing (JPG, PNG, or PDF) to compare them.")

col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("Original (A)", type=["jpg", "jpeg", "png", "pdf"], key="file_a")
with col2:
    file_b = st.file_uploader("Revised (B)", type=["jpg", "jpeg", "png", "pdf"], key="file_b")

mode = st.radio(
    "Diff mode",
    options=["linework", "photo"],
    horizontal=True,
    help="'linework' is tuned for CAD/line-art drawings; 'photo' uses SSIM for general images.",
)

run_clicked = st.button("Compare", type="primary", disabled=not (file_a and file_b))

if run_clicked and file_a and file_b:
    with st.spinner("Running comparison pipeline..."):
        files = {
            "file_a": (file_a.name, file_a.getvalue()),
            "file_b": (file_b.name, file_b.getvalue()),
        }
        try:
            response = requests.post(
                f"{API_URL}/compare", files=files, params={"mode": mode}, timeout=120
            )
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:
            st.error(f"Request to API failed: {exc}")
            result = None

    if result:
        stats = result["stats"]
        images = result["images"]

        if not stats["registration_success"]:
            st.warning(
                "Registration did not reach a confident alignment "
                f"(method: {stats['registration_method']}, score: {stats['alignment_score']}). "
                "Diff results below may not be reliable."
            )

        st.subheader("Original Images")
        oc1, oc2 = st.columns(2)
        with oc1:
            st.image(f"{API_URL}{images['original_a']}", caption="Original (A)", use_container_width=True)
        with oc2:
            st.image(f"{API_URL}{images['original_b']}", caption="Revised (B)", use_container_width=True)

        st.subheader("Diff Visualization")
        vc1, vc2 = st.columns(2)
        with vc1:
            st.image(
                f"{API_URL}{images['annotated_regions']}",
                caption="Highlighted changed regions (on aligned B)",
                use_container_width=True,
            )
        with vc2:
            st.image(
                f"{API_URL}{images['added_removed_overlay']}",
                caption="Added (red) / Removed (blue) overlay",
                use_container_width=True,
            )

        st.image(f"{API_URL}{images['heatmap']}", caption="Diff heatmap", use_container_width=True)

        st.subheader("Statistics")
        m1, m2, m3 = st.columns(3)
        m1.metric("Changed regions", stats["total_region_count"])
        m2.metric("% area changed", f"{stats['percent_area_changed']}%")
        m3.metric("Alignment score", stats["alignment_score"])
        st.json(stats)

        st.subheader("AI Summary")
        st.write(result["summary"])
