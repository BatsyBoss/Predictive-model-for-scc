"""
app.py
------
AI-Powered Self-Compacting Concrete (SCC) Mix Design Studio.

A Streamlit app with three tabs:
  1. User Mode   — predict compressive strength from a mix, OR reverse-design
                    3 candidate mixes for a target strength.
  2. Admin Mode  — password-protected dataset upload, model retraining,
                    performance metrics, and diagnostic charts.
  3. AI Chatbot  — an LLM-backed assistant for SCC / concrete / app-usage
                    questions, with optional live web search grounding.

Run locally:
    streamlit run app.py

See README.md for setup, secrets configuration, and deployment notes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make `src` importable regardless of cwd

import pandas as pd
import streamlit as st

from src.config import (
    APP_TITLE,
    DEFAULT_ADMIN_PASSWORD,
    FEATURE_BOUNDS,
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    TARGET_COLUMN,
    MODEL_PATH,
    WC_RATIO_BOUNDS,
)
from src.data_utils import (
    DataValidationError,
    clean_dataset,
    load_default_dataset,
    load_uploaded_dataset,
    validate_dataset,
)
from src.model_utils import (
    TrainingResult,
    load_saved_model,
    predict_single,
    save_model,
    train_model,
)
from src.optimization import generate_mix_designs
from src.visualization import (
    fig_to_png_bytes,
    plot_actual_vs_predicted,
    plot_distribution,
    plot_feature_importance,
    plot_mix_comparison_bar,
    plot_predicted_vs_target,
    plot_strength_gauge,
)
from src.chatbot import (
    get_llm_response,
    needs_web_search,
    offline_fallback_response,
    search_web,
)

st.set_page_config(page_title=APP_TITLE, page_icon="🧱", layout="wide")


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------
def _get_secret(key: str, default=None):
    """Safely read st.secrets without blowing up when no secrets.toml exists."""
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


def _bundle_from_result(result: TrainingResult) -> dict:
    """Build the lightweight 'active model' bundle stored in session state."""
    return {
        "model": result.model,
        "model_type": result.model_type,
        "metrics": result.metrics,
        "feature_importance": result.feature_importance,
        "trained_at": result.trained_at,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
    }


def _download_png_button(fig, filename: str, label: str, key: str):
    """Render a PNG download button for a Plotly figure, degrading gracefully
    if the Kaleido export engine isn't available in this environment."""
    png_bytes = fig_to_png_bytes(fig)
    if png_bytes:
        st.download_button(label, data=png_bytes, file_name=filename,
                            mime="image/png", key=key)
    else:
        st.caption(
            "⚠️ PNG export unavailable (needs the `kaleido` package + a local "
            "Chrome install). The chart above is still fully interactive."
        )


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
def initialize_state():
    if "active_dataset" not in st.session_state:
        cleaned, _ = clean_dataset(load_default_dataset())
        st.session_state.active_dataset = cleaned
        st.session_state.active_dataset_name = "Bundled sample SCC dataset (synthetic demo data)"

    if "model_bundle" not in st.session_state:
        saved = load_saved_model(MODEL_PATH)
        if saved is not None:
            st.session_state.model_bundle = saved
            st.session_state.last_training_result = None
        else:
            # First-ever run: no saved model on disk yet, so train once on
            # the bundled sample dataset and persist it.
            with st.spinner("First run: training initial model on the bundled sample dataset..."):
                result = train_model(st.session_state.active_dataset)
                save_model(result, MODEL_PATH)
                st.session_state.model_bundle = _bundle_from_result(result)
                st.session_state.last_training_result = result

    st.session_state.setdefault("admin_authenticated", False)
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("prediction_log", [])
    st.session_state.setdefault("last_reverse_designs", None)
    st.session_state.setdefault("last_reverse_target", None)


# ---------------------------------------------------------------------------
# USER MODE
# ---------------------------------------------------------------------------
def render_user_mode():
    st.header("🧪 User Mode")
    st.caption("Predict compressive strength for a given mix, or generate mix designs for a target strength.")

    mode = st.radio(
        "What would you like to do?",
        ["Predict strength from a mix", "Reverse-design mixes for a target strength"],
        horizontal=True,
    )

    model = st.session_state.model_bundle["model"]

    if mode == "Predict strength from a mix":
        _render_predict_mode(model)
    else:
        _render_reverse_design_mode(model)


def _render_predict_mode(model):
    st.subheader("Enter mix proportions")
    b = FEATURE_BOUNDS
    col1, col2, col3 = st.columns(3)
    with col1:
        cement = st.number_input(FEATURE_LABELS["Cement"], *b["Cement"], value=420.0, step=5.0)
        water = st.number_input(FEATURE_LABELS["Water"], *b["Water"], value=180.0, step=1.0)
    with col2:
        fine_agg = st.number_input(FEATURE_LABELS["Fine_Aggregate"], *b["Fine_Aggregate"], value=850.0, step=5.0)
        coarse_agg = st.number_input(FEATURE_LABELS["Coarse_Aggregate"], *b["Coarse_Aggregate"], value=800.0, step=5.0)
    with col3:
        superplasticizer = st.number_input(FEATURE_LABELS["Superplasticizer"], *b["Superplasticizer"], value=6.0, step=0.5)
        age = st.number_input(FEATURE_LABELS["Age"], *b["Age"], value=28, step=1)

    wc_ratio = water / cement if cement > 0 else float("nan")
    lo_wc, hi_wc = WC_RATIO_BOUNDS
    st.caption(f"Water/Cement ratio: **{wc_ratio:.3f}**"
               + (f" — outside the typical SCC range ({lo_wc}–{hi_wc}); prediction may be extrapolating."
                  if not (lo_wc <= wc_ratio <= hi_wc) else ""))

    if st.button("🔮 Predict Compressive Strength", type="primary"):
        input_dict = {
            "Cement": cement, "Water": water, "Fine_Aggregate": fine_agg,
            "Coarse_Aggregate": coarse_agg, "Superplasticizer": superplasticizer, "Age": age,
        }
        prediction = predict_single(model, input_dict)

        st.session_state.prediction_log.append({**input_dict, "Predicted_Strength": round(prediction, 2)})

        m1, m2 = st.columns([1, 2])
        with m1:
            st.metric("Predicted Compressive Strength", f"{prediction:.2f} MPa")
        with m2:
            st.plotly_chart(plot_strength_gauge(prediction), use_container_width=True)

    if st.session_state.prediction_log:
        with st.expander(f"📜 Prediction history this session ({len(st.session_state.prediction_log)})"):
            log_df = pd.DataFrame(st.session_state.prediction_log)
            st.dataframe(log_df, use_container_width=True)
            st.download_button(
                "⬇️ Download prediction results (CSV)",
                data=log_df.to_csv(index=False),
                file_name="scc_prediction_results.csv",
                mime="text/csv",
            )


def _render_reverse_design_mode(model):
    st.subheader("Enter target compressive strength")
    col1, col2 = st.columns(2)
    with col1:
        target = st.number_input("Target Compressive Strength (MPa)", 10.0, 100.0, 40.0, 1.0)
    with col2:
        age = st.selectbox("Design / testing age (days)", [1, 3, 7, 14, 28, 56, 90, 180], index=4)

    if st.button("⚙️ Generate 3 Mix Designs", type="primary"):
        with st.spinner("Searching the mix-design space..."):
            designs = generate_mix_designs(model, target_strength=target, age=age, n_designs=3)
        st.session_state.last_reverse_designs = designs
        st.session_state.last_reverse_target = target

    designs = st.session_state.last_reverse_designs
    target_used = st.session_state.last_reverse_target
    if designs is None:
        return

    if not designs["Achievable"].all():
        st.warning(
            "⚠️ At least one generated design misses the target by more than 15%. "
            "The requested strength may be difficult to reach within realistic SCC "
            "proportioning limits — consider a lower target or a longer curing age."
        )
    else:
        st.success("✅ Generated 3 mix designs close to your target strength.")

    display_df = designs.rename(columns=FEATURE_LABELS).rename(columns={
        "Predicted_Strength": "Predicted Strength (MPa)",
        "Percent_Error": "Error (%)",
        "Relative_Cost_Index": "Relative Cost Index (0-100, illustrative)",
        "Achievable": "Within 15% of target",
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        bar_fig = plot_mix_comparison_bar(designs)
        st.plotly_chart(bar_fig, use_container_width=True)
        _download_png_button(bar_fig, "mix_design_comparison.png", "⬇️ Download chart (PNG)", key="png_mix_bar")
    with c2:
        pred_fig = plot_predicted_vs_target(designs, target_used)
        st.plotly_chart(pred_fig, use_container_width=True)
        _download_png_button(pred_fig, "predicted_vs_target.png", "⬇️ Download chart (PNG)", key="png_pred_target")

    st.download_button(
        "⬇️ Download mix designs (CSV)",
        data=designs.to_csv(index=False),
        file_name="scc_reverse_mix_designs.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# ADMIN MODE
# ---------------------------------------------------------------------------
def render_admin_mode():
    st.header("🔐 Admin Mode")

    if not st.session_state.admin_authenticated:
        st.info("Enter the admin password to upload data, retrain the model, and view diagnostics.")
        pwd = st.text_input("Admin password", type="password")
        if st.button("Login"):
            correct_password = _get_secret("ADMIN_PASSWORD", DEFAULT_ADMIN_PASSWORD)
            if pwd == correct_password:
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        st.caption(
            "Demo password is `admin123` unless overridden. Set your own via "
            "`.streamlit/secrets.toml` → `ADMIN_PASSWORD = \"...\"` before deploying."
        )
        return

    top_l, top_r = st.columns([4, 1])
    with top_l:
        st.success("🔓 Admin session active")
    with top_r:
        if st.button("Log out"):
            st.session_state.admin_authenticated = False
            st.rerun()

    # --- Dataset upload -----------------------------------------------
    st.divider()
    st.subheader("📂 Dataset")
    uploaded = st.file_uploader(
        "Upload a new training dataset (CSV)",
        type="csv",
        help=f"Required columns: {', '.join(FEATURE_COLUMNS + [TARGET_COLUMN])}",
    )
    if uploaded is not None:
        try:
            raw_df = load_uploaded_dataset(uploaded)
            validate_dataset(raw_df)
            cleaned, report = clean_dataset(raw_df)
            st.session_state.active_dataset = cleaned
            st.session_state.active_dataset_name = uploaded.name
            st.success(f"Loaded **{uploaded.name}** — {report['rows_out']} usable rows.")
            if report["missing_values_filled"]:
                st.info(f"Filled missing values via median imputation: {report['missing_values_filled']}")
            if report["rows_dropped_negative"]:
                st.warning(f"Dropped {report['rows_dropped_negative']} row(s) with invalid (negative or zero) values.")
        except DataValidationError as e:
            st.error(f"Dataset validation failed: {e}")
        except Exception as e:
            st.error(f"Could not read the uploaded file: {e}")

    st.caption(
        f"Active dataset: **{st.session_state.active_dataset_name}** "
        f"({len(st.session_state.active_dataset)} rows)"
    )
    with st.expander("Preview active dataset"):
        st.dataframe(st.session_state.active_dataset.head(20), use_container_width=True)

    # --- Retrain ---------------------------------------------------------
    st.divider()
    st.subheader("🚂 Train / Retrain Model")
    if st.button("Retrain Model on Active Dataset", type="primary"):
        with st.spinner("Training model..."):
            result = train_model(st.session_state.active_dataset)
        st.session_state.last_training_result = result
        st.session_state.model_bundle = _bundle_from_result(result)
        st.success(
            f"Retrained **{result.model_type}** — "
            f"R² = {result.metrics['r2_score']:.3f}, RMSE = {result.metrics['rmse']:.2f} MPa "
            f"({result.n_train} train / {result.n_test} test samples)"
        )

    if st.session_state.last_training_result is not None:
        if st.button("💾 Save Current Model to Disk"):
            path = save_model(st.session_state.last_training_result, MODEL_PATH)
            st.success(f"Model saved to `{path}`. It will now be used across the app (including User Mode) and reloaded automatically next run.")
    else:
        st.caption("Active model was loaded from a previously saved file. Click **Retrain** above to train a new version in this session.")

    # --- Metrics -----------------------------------------------------
    st.divider()
    st.subheader("📊 Model Performance")
    bundle = st.session_state.model_bundle
    metrics = bundle["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² Score", f"{metrics['r2_score']:.3f}")
    c2.metric("RMSE", f"{metrics['rmse']:.2f} MPa")
    c3.metric("CV R² (5-fold)", f"{metrics['cv_r2_mean']:.3f}" if metrics.get("cv_r2_mean") is not None else "N/A")
    c4.metric("Training Samples", metrics["n_samples"])
    st.caption(f"Model type: **{bundle['model_type']}** · Last trained: {bundle['trained_at']}")

    # --- Diagnostic charts --------------------------------------------
    st.divider()
    st.subheader("📈 Diagnostic Charts")
    tr = st.session_state.last_training_result
    if tr is not None:
        avp_fig = plot_actual_vs_predicted(tr.y_test, tr.y_pred)
        st.plotly_chart(avp_fig, use_container_width=True)
        _download_png_button(avp_fig, "actual_vs_predicted.png", "⬇️ Download chart (PNG)", key="png_avp")
    else:
        st.info("Actual-vs-Predicted needs a training run from this session — click **Retrain Model** above.")

    fi_fig = plot_feature_importance(bundle["feature_importance"])
    st.plotly_chart(fi_fig, use_container_width=True)
    _download_png_button(fi_fig, "feature_importance.png", "⬇️ Download chart (PNG)", key="png_fi")

    st.divider()
    st.subheader("📊 Dataset Distributions")
    d1, d2 = st.columns(2)
    active_df = st.session_state.active_dataset
    with d1:
        hist_strength = plot_distribution(active_df, "Compressive_Strength", "Compressive Strength Distribution")
        st.plotly_chart(hist_strength, use_container_width=True)
        _download_png_button(hist_strength, "strength_distribution.png", "⬇️ Download chart (PNG)", key="png_strength_hist")
    with d2:
        hist_cement = plot_distribution(active_df, "Cement", "Cement Content Distribution", color="#F97316")
        st.plotly_chart(hist_cement, use_container_width=True)
        _download_png_button(hist_cement, "cement_distribution.png", "⬇️ Download chart (PNG)", key="png_cement_hist")

    # --- Downloads -----------------------------------------------------
    st.divider()
    st.subheader("⬇️ Downloads")
    metrics_df = pd.DataFrame([{"Metric": k, "Value": v} for k, v in metrics.items()])
    st.download_button(
        "Download Model Metrics (CSV)",
        data=metrics_df.to_csv(index=False),
        file_name="scc_model_metrics.csv",
        mime="text/csv",
    )
    if tr is not None:
        preds_df = tr.X_test.copy()
        preds_df["Actual_Strength"] = tr.y_test.values
        preds_df["Predicted_Strength"] = tr.y_pred
        st.download_button(
            "Download Test-Set Predictions (CSV)",
            data=preds_df.to_csv(index=False),
            file_name="scc_test_predictions.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# AI CHATBOT
# ---------------------------------------------------------------------------
def render_chatbot():
    st.header("💬 AI Assistant")
    st.caption("Ask about SCC mix design, concrete materials, or how to use this app.")

    with st.expander("⚙️ AI Configuration", expanded=not st.session_state.get("chat_api_key")):
        provider = st.selectbox("LLM Provider", ["Anthropic (Claude)", "OpenAI"], key="chat_provider")
        default_key = _get_secret("ANTHROPIC_API_KEY" if provider == "Anthropic (Claude)" else "OPENAI_API_KEY", "")
        api_key = st.text_input("API Key", value=default_key, type="password", key="chat_api_key")
        default_model = "claude-sonnet-5" if provider == "Anthropic (Claude)" else "gpt-4o-mini"
        model_name = st.text_input(
            "Model name", value=default_model, key="chat_model_name",
            help="Model catalogs change over time — double check your provider's current model list.",
        )
        web_enabled = st.checkbox(
            "🌐 Allow live web search when a question needs current info",
            value=True, key="chat_web_enabled",
        )
        st.caption(
            "Your key is only used for this session's API calls and is never written to disk. "
            "Without a key, the assistant falls back to a small built-in FAQ."
        )

    hist_col, clear_col = st.columns([5, 1])
    with clear_col:
        if st.button("🗑️ Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_msg = st.chat_input("Ask about SCC mix design, properties, or how to use this app...")
    if not user_msg:
        return

    st.session_state.chat_history.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            do_search = web_enabled and needs_web_search(user_msg)
            web_results = search_web(user_msg) if do_search else None

            if api_key:
                try:
                    reply = get_llm_response(
                        history=st.session_state.chat_history,
                        provider=provider,
                        api_key=api_key,
                        model_name=model_name,
                        web_results=web_results,
                    )
                except Exception as e:
                    reply = (
                        f"⚠️ Couldn't reach the {provider} API ({e}). "
                        f"Falling back to the built-in FAQ:\n\n{offline_fallback_response(user_msg)}"
                    )
            else:
                reply = offline_fallback_response(user_msg)
                if do_search and web_results:
                    reply += "\n\n_Web results were found but need an API key to be summarized — add one above for full answers._"

        st.markdown(reply)
        if do_search and web_results:
            with st.expander("🔎 Web sources consulted"):
                for r in web_results:
                    title = r.get("title", "source")
                    href = r.get("href", "")
                    st.markdown(f"- [{title}]({href})" if href else f"- {title}")

    st.session_state.chat_history.append({"role": "assistant", "content": reply})


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    initialize_state()

    st.title(f"🧱 {APP_TITLE}")
    st.caption(
        "Machine-learning powered compressive strength prediction and reverse "
        "mix design for Self-Compacting Concrete (SCC)."
    )

    with st.sidebar:
        st.subheader("Status")
        st.metric("Active dataset rows", len(st.session_state.active_dataset))
        st.metric("Model R²", f"{st.session_state.model_bundle['metrics']['r2_score']:.3f}")
        st.caption(f"Model type: {st.session_state.model_bundle['model_type']}")
        st.divider()
        st.caption(
            "⚠️ This is a demo/educational tool. Always validate final SCC mix "
            "designs with laboratory trial batches (slump-flow, L-box, V-funnel) "
            "and applicable standards (e.g. EFNARC / ACI 237R / EN 206) before use."
        )

    tab_user, tab_admin, tab_chat = st.tabs(["🧪 User Mode", "🔐 Admin Mode", "💬 AI Chatbot"])
    with tab_user:
        render_user_mode()
    with tab_admin:
        render_admin_mode()
    with tab_chat:
        render_chatbot()


if __name__ == "__main__":
    main()
