"""Streamlit uncertainty UI for the research copilot.

Run: streamlit run src/clinical_signal_copilot/ui/streamlit_app.py

*** NOT A MEDICAL DEVICE. NOT FOR CLINICAL USE. ***
"""

from __future__ import annotations

import numpy as np

from clinical_signal_copilot import DISCLAIMER
from clinical_signal_copilot.pipeline import PipelineConfig, run_pipeline
from clinical_signal_copilot.signals.synthetic import SyntheticConfig
from clinical_signal_copilot.ssl.contrastive import ContrastiveConfig


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Clinical Signal Copilot (Research)", layout="wide")
    st.error(DISCLAIMER)
    st.title("Clinical Signal Copilot — Uncertainty UI")
    st.caption("Fully synthetic multi-channel signals · SSL · conformal · fairness")

    with st.sidebar:
        st.header("Controls")
        seed = st.number_input("Seed", value=42, step=1)
        n_records = st.slider("Records", 8, 48, 20)
        epochs = st.slider("SSL epochs", 1, 10, 3)
        run = st.button("Run pipeline", type="primary")

    if run or "result" not in st.session_state:
        with st.spinner("Running research pipeline…"):
            result = run_pipeline(
                PipelineConfig(
                    synthetic=SyntheticConfig(n_records=int(n_records), duration_s=10.0, seed=int(seed)),
                    contrastive=ContrastiveConfig(epochs=int(epochs), max_windows=256, seed=int(seed)),
                    seed=int(seed),
                )
            )
            st.session_state["result"] = result

    result = st.session_state["result"]
    c1, c2, c3 = st.columns(3)
    c1.metric("SSL F1", f"{result['metrics_ssl']['f1']:.3f}")
    c2.metric("Supervised-only F1", f"{result['metrics_sup']['f1']:.3f}")
    c3.metric("Conformal coverage", f"{result['coverage']['empirical_coverage']:.3f}")

    st.subheader("Case studies")
    st.json(result["case_studies"])

    st.subheader("Example synthetic waveform")
    ds = result["dataset"]
    ch = st.selectbox("Channel", list(range(ds.n_channels)))
    rec = st.slider("Record", 0, ds.n_records - 1, 0)
    t = np.arange(ds.n_samples) / ds.fs
    st.line_chart({"signal": ds.signals[rec, ch]})
    st.write(f"Subgroup={ds.subgroups[rec]} · SQI={ds.sqi[rec].round(3).tolist()} · fs={ds.fs}")

    st.subheader("Copilot report (markdown)")
    st.markdown(result["report_markdown"])
    st.error(DISCLAIMER)


if __name__ == "__main__":
    main()
