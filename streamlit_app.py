import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Cricket Win Probability Engine", page_icon="🏏", layout="centered")

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return pickle.load(open("models/xgb_model.pkl", "rb"))

model = load_model()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🏏 Cricket Win Probability Engine")
st.markdown("Enter the current match situation to get a live win probability.")

col1, col2 = st.columns(2)

with col1:
    balls_remaining = st.number_input("Balls Remaining", min_value=1, max_value=120, value=30)
    wickets_fallen  = st.number_input("Wickets Fallen",  min_value=0, max_value=10,  value=3)

with col2:
    crr = st.number_input("Current Run Rate (CRR)", min_value=0.0, max_value=36.0, value=8.0, step=0.1)
    rrr = st.number_input("Required Run Rate (RRR)", min_value=0.0, max_value=36.0, value=9.0, step=0.1)

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🎯 Predict Win Probability", use_container_width=True):
    wickets_remaining = 10 - wickets_fallen
    data = np.array([[balls_remaining, wickets_remaining, crr, rrr]])
    prob = float(model.predict_proba(data)[0][1])
    win_pct  = round(prob * 100, 2)
    lose_pct = round((1 - prob) * 100, 2)

    st.divider()

    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=win_pct,
        title={"text": "Win Probability (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": "#2ecc71" if win_pct >= 50 else "#e74c3c"},
            "steps": [
                {"range": [0,  40],  "color": "#fadbd8"},
                {"range": [40, 60],  "color": "#fef9e7"},
                {"range": [60, 100], "color": "#d5f5e3"},
            ],
            "threshold": {"line": {"color": "black", "width": 3}, "value": 50}
        }
    ))
    fig.update_layout(height=300, margin=dict(t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Metrics row
    m1, m2, m3 = st.columns(3)
    m1.metric("Win %",  f"{win_pct}%")
    m2.metric("Lose %", f"{lose_pct}%")
    m3.metric("Overs Left", f"{balls_remaining // 6}.{balls_remaining % 6}")

    # Situation label
    if prob > 0.65:
        st.success("🔥 Strong chance of winning!")
    elif prob > 0.45:
        st.warning("⚖️ Match is evenly poised!")
    else:
        st.error("💀 Under pressure — need something special!")

    # Key stats summary
    st.markdown("#### 📊 Match Snapshot")
    st.markdown(f"""
    | Stat | Value |
    |------|-------|
    | Wickets remaining | {wickets_remaining} |
    | Balls remaining | {balls_remaining} |
    | CRR vs RRR | {crr} vs {rrr} |
    | Run rate pressure | {'High 🔴' if rrr - crr > 2 else 'Medium 🟡' if rrr - crr > 0 else 'Low 🟢'} |
    """)