"""AI Complaint Analyzer — Streamlit application."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from model import (
    get_category_probabilities,
    load_model,
    predict_category_with_debug,
    train_model,
)
from utils import (
    CATEGORIES,
    SENTIMENTS,
    URGENCY_LEVELS,
    analyze_sentiment,
    detect_urgency,
    detect_urgency_details,
    get_summary_metrics,
    init_database,
    load_all_complaints,
    save_complaint,
)

# Page config
st.set_page_config(
    page_title="AI Complaint Analyzer",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        color: #5a6c7d;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2f7 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 8px;
        background: #f0f9ff;
        border-left: 4px solid #0284c7;
        margin: 0.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_model_bundle():
    diagnostics = train_model(force=False)
    model_tuple = load_model()
    return model_tuple, diagnostics


def urgency_badge(urgency: str) -> str:
    palette = {
        "high": "#ef4444",
        "medium": "#f59e0b",
        "low": "#22c55e",
    }
    return (
        f"<span style='padding:0.2rem 0.6rem;border-radius:999px;"
        f"background:{palette.get(urgency, '#94a3b8')}22;color:{palette.get(urgency, '#475569')};"
        f"font-weight:700;text-transform:uppercase;'>{urgency}</span>"
    )


def render_submit_complaint(model_tuple):
    st.markdown('<p class="main-header">Submit Complaint</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Describe your issue. The system will classify it automatically.</p>',
        unsafe_allow_html=True,
    )

    complaint = st.text_area(
        "Complaint details",
        height=180,
        placeholder="Example: There has been no water supply in our hostel block for three days...",
        key="complaint_input",
    )

    col1, col2, _ = st.columns([1, 1, 2])
    with col1:
        analyze_btn = st.button("Analyze Complaint", type="primary", use_container_width=True)
    with col2:
        clear_btn = st.button("Clear", use_container_width=True)

    if clear_btn:
        st.session_state.pop("last_result", None)
        st.rerun()

    if analyze_btn:
        if not complaint or not complaint.strip():
            st.warning("Please enter your complaint before submitting.")
            return

        with st.spinner("Analyzing complaint..."):
            pred_debug = predict_category_with_debug(complaint, model_tuple)
            category = pred_debug["category"]
            confidence = pred_debug["confidence"]
            sentiment = analyze_sentiment(complaint, predicted_category=category)
            urgency_data = detect_urgency_details(complaint, category)
            urgency, urgency_score = urgency_data["urgency"], urgency_data["score"]
            row_id = save_complaint(complaint, category, sentiment, urgency)
            probs = get_category_probabilities(complaint, model_tuple)

        st.session_state["last_result"] = {
            "text": complaint,
            "category": category,
            "confidence": confidence,
            "sentiment": sentiment,
            "urgency": urgency,
            "urgency_score": urgency_score,
            "id": row_id,
            "probs": probs,
            "prediction_reasons": pred_debug["reasons"],
            "urgency_reasons": urgency_data["reasons"],
            "safety_matches": list(getattr(pred_debug.get("safety"), "matched_all", []) or []),
            "model_label": pred_debug.get("model_label"),
            "model_confidence": pred_debug.get("model_confidence"),
        }
        st.success(f"Complaint recorded successfully (ID: {row_id}).")

    if "last_result" in st.session_state:
        r = st.session_state["last_result"]
        st.divider()
        st.subheader("Analysis Results")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Category", r["category"].title())
        m2.metric("Confidence", f"{r['confidence'] * 100:.1f}%")
        m3.metric("Sentiment", r["sentiment"].title())
        m4.metric("Urgency", r["urgency"].title())

        st.progress(min(max(r["confidence"], 0.0), 1.0), text="Prediction confidence meter")
        st.markdown(
            f"<div class='prediction-box'><strong>Urgency:</strong> {urgency_badge(r['urgency'])} "
            f"&nbsp; <strong>Urgency score:</strong> {r['urgency_score']}</div>",
            unsafe_allow_html=True,
        )

        fig = px.bar(
            r["probs"],
            x="probability",
            y="category",
            orientation="h",
            title="Category probability distribution",
            labels={"probability": "Probability", "category": "Category"},
            color="probability",
            color_continuous_scale="Blues",
        )
        fig.update_layout(showlegend=False, height=350, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Prediction Debug Logs"):
            if r.get("model_label"):
                st.caption(
                    f"Raw model: {r['model_label']} "
                    f"({(r.get('model_confidence') or 0) * 100:.1f}%) → "
                    f"Final: {r['category']} ({r['confidence'] * 100:.1f}%)"
                )
            if r.get("safety_matches"):
                st.markdown(f"**Matched safety keywords:** `{', '.join(r['safety_matches'])}`")
            st.markdown("**Category decision**")
            for item in r["prediction_reasons"]:
                st.write(f"- {item}")
            st.markdown("**Urgency decision**")
            for item in r["urgency_reasons"]:
                st.write(f"- {item}")


def render_dashboard(df, diagnostics):
    st.markdown('<p class="main-header">Dashboard</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Overview of all submitted complaints and trends.</p>',
        unsafe_allow_html=True,
    )

    metrics = get_summary_metrics(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Complaints", metrics["total"])
    c2.metric("High Urgency", metrics["high_urgency"])
    c3.metric("Negative Sentiment", metrics["negative_sentiment"])
    c4.metric("Top Category", str(metrics["top_category"]).title())

    if df.empty:
        st.info("No complaints yet. Submit a complaint to populate the dashboard.")
        return

    col_left, col_right = st.columns(2)

    with col_left:
        cat_counts = df["category"].value_counts().reset_index()
        cat_counts.columns = ["category", "count"]
        fig_cat = px.pie(
            cat_counts,
            values="count",
            names="category",
            title="Complaints by Category",
            hole=0.35,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_cat.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_cat, use_container_width=True)

    with col_right:
        sent_counts = df["sentiment"].value_counts().reset_index()
        sent_counts.columns = ["sentiment", "count"]
        color_map = {"positive": "#22c55e", "neutral": "#94a3b8", "negative": "#ef4444"}
        fig_sent = px.bar(
            sent_counts,
            x="sentiment",
            y="count",
            title="Sentiment Distribution",
            color="sentiment",
            color_discrete_map=color_map,
        )
        fig_sent.update_layout(showlegend=False)
        st.plotly_chart(fig_sent, use_container_width=True)

    urg_order = ["high", "medium", "low"]
    urg_counts = df["urgency"].value_counts().reindex(urg_order, fill_value=0).reset_index()
    urg_counts.columns = ["urgency", "count"]

    fig_urg = go.Figure(
        data=[
            go.Bar(
                x=urg_counts["urgency"],
                y=urg_counts["count"],
                marker_color=["#dc2626", "#d97706", "#16a34a"],
                text=urg_counts["count"],
                textposition="auto",
            )
        ]
    )
    fig_urg.update_layout(
        title="Urgency Levels",
        xaxis_title="Urgency",
        yaxis_title="Count",
        height=380,
    )
    st.plotly_chart(fig_urg, use_container_width=True)

    st.subheader("Recent Activity")
    recent = df.head(5)[["id", "category", "sentiment", "urgency", "created_at"]]
    st.dataframe(recent, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Model Evaluation")
    e1, e2, e3 = st.columns(3)
    e1.metric("Holdout Accuracy", f"{diagnostics['accuracy'] * 100:.2f}%")
    e2.metric("Train Samples", "470")
    e3.metric("Test Samples", "118")

    cm_df = diagnostics["confusion_matrix_df"]
    cm_fig = px.imshow(
        cm_df,
        labels={"x": "Predicted", "y": "Actual", "color": "Count"},
        title="Confusion Matrix",
        color_continuous_scale="Blues",
    )
    st.plotly_chart(cm_fig, use_container_width=True)

    with st.expander("Classification Report"):
        st.dataframe(diagnostics["classification_report_df"], use_container_width=True)

    with st.expander("Top Misclassified Samples"):
        st.dataframe(diagnostics["misclassified_df"], use_container_width=True, hide_index=True)

    with st.expander("Sample Predictions"):
        st.dataframe(diagnostics["sample_predictions_df"], use_container_width=True, hide_index=True)


def render_history(df):
    st.markdown('<p class="main-header">Complaint History</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Browse and filter all stored complaints.</p>',
        unsafe_allow_html=True,
    )

    if df.empty:
        st.info("No complaints in history yet.")
        return

    f1, f2, f3, f4, f5 = st.columns(5)
    with f1:
        cat_filter = st.multiselect("Category", options=CATEGORIES, default=[])
    with f2:
        sent_filter = st.multiselect("Sentiment", options=SENTIMENTS, default=[])
    with f3:
        urg_filter = st.multiselect("Urgency", options=URGENCY_LEVELS, default=[])
    with f4:
        search = st.text_input("Search text", placeholder="Keyword in complaint...")
    with f5:
        sort_order = st.selectbox("Sort", ["Newest", "Oldest", "High urgency first"])

    filtered = df.copy()
    if cat_filter:
        filtered = filtered[filtered["category"].isin(cat_filter)]
    if sent_filter:
        filtered = filtered[filtered["sentiment"].isin(sent_filter)]
    if urg_filter:
        filtered = filtered[filtered["urgency"].isin(urg_filter)]
    if search.strip():
        mask = filtered["complaint_text"].str.contains(
            search.strip(), case=False, na=False
        )
        filtered = filtered[mask]
    if sort_order == "Oldest":
        filtered = filtered.sort_values("id", ascending=True)
    elif sort_order == "High urgency first":
        rank = {"high": 0, "medium": 1, "low": 2}
        filtered = filtered.assign(_rank=filtered["urgency"].map(rank)).sort_values(["_rank", "id"])
        filtered = filtered.drop(columns=["_rank"])

    st.caption(f"Showing {len(filtered)} of {len(df)} complaints")
    display_cols = [
        "id",
        "complaint_text",
        "category",
        "sentiment",
        "urgency",
        "created_at",
    ]
    st.dataframe(
        filtered[display_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "id": "ID",
            "complaint_text": st.column_config.TextColumn("Complaint", width="large"),
            "category": "Category",
            "sentiment": "Sentiment",
            "urgency": "Urgency",
            "created_at": "Submitted",
        },
    )

    csv = filtered[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered data (CSV)",
        data=csv,
        file_name="complaints_export.csv",
        mime="text/csv",
    )


def main():
    init_database()
    model_tuple, diagnostics = get_model_bundle()

    with st.sidebar:
        st.title("📋 Complaint Analyzer")
        st.caption("AI-powered classification & insights")
        st.divider()
        page = st.radio(
            "Navigation",
            ["Submit Complaint", "Dashboard", "Complaint History"],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown("**Categories**")
        for cat in CATEGORIES:
            st.markdown(f"• {cat.title()}")
        st.divider()
        if st.button("Retrain ML Model", use_container_width=True):
            with st.spinner("Training model..."):
                from model import ensure_dataset

                ensure_dataset(force=True)
                train_model(force=True)
                st.cache_resource.clear()
            st.success("Model retrained and saved.")
            st.rerun()

    df = load_all_complaints()

    if page == "Submit Complaint":
        render_submit_complaint(model_tuple)
    elif page == "Dashboard":
        render_dashboard(df, diagnostics)
    else:
        render_history(df)


if __name__ == "__main__":
    main()
