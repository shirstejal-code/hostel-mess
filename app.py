   rm uv.lock pyproject.toml streamlit_app.py"""Streamlit dashboard: tomorrow's recommendation + confidence range."""
import os
import datetime as dt
import joblib
import pandas as pd
import streamlit as st
from common import make_features, COST_WASTE, COST_SHORT, CAPACITY
from decision import optimal_quantity, prediction_interval, expected_cost_curve

st.set_page_config(page_title="Mess Food-Waste Decider", page_icon="🍛", layout="centered")

# On first launch (e.g. on a fresh cloud deploy) build data + model automatically.
if not os.path.exists("models/model.joblib"):
    with st.spinner("First run: generating data and training model..."):
        import generate_data, train
        if not os.path.exists("data/mess_data.csv"):
            generate_data.main()
        train.main()


@st.cache_resource
def load():
    return joblib.load("models/model.joblib")


art = load()
model, residuals = art["model"], art["residuals"]

st.title("🍛 Hostel Mess Food-Waste Decider")
st.caption("Forecast attendance → quantify uncertainty → decide how many plates to cook.")

with st.sidebar:
    st.header("Costs (Rs)")
    co = st.number_input("Cost of a wasted plate", 1, 500, COST_WASTE)
    cu = st.number_input("Penalty per student left hungry", 1, 500, COST_SHORT)
    level = st.slider("Confidence interval", 0.5, 0.95, 0.8, 0.05)

c1, c2 = st.columns(2)
date = c1.date_input("Date", dt.date.today() + dt.timedelta(days=1))
meal = c2.selectbox("Meal", ["breakfast", "lunch", "dinner"], index=2)
c3, c4 = st.columns(2)
rain = c3.slider("Expected rain (mm)", 0.0, 80.0, 0.0)
temp = c4.slider("Temperature (°C)", 18.0, 42.0, 30.0)
menu = st.slider("Menu popularity (1 = disliked, 5 = favourite)", 1, 5, 3)
c5, c6, c7 = st.columns(3)
holiday = c5.checkbox("Holiday / break")
exam = c6.checkbox("Exam period")
event = c7.checkbox("Campus event")

row = pd.DataFrame([dict(date=str(date), meal=meal, rain_mm=rain, temp_c=temp,
                         is_holiday=int(holiday), is_exam=int(exam),
                         is_event=int(event), menu_score=menu)])
pred = float(model.predict(make_features(row))[0])
qty = int(optimal_quantity(pred, residuals, cu, co))
lo, hi = prediction_interval(pred, residuals, level)

st.divider()
m1, m2, m3 = st.columns(3)
m1.metric("Expected students", f"{pred:.0f}")
m2.metric(f"{int(level*100)}% range", f"{float(lo):.0f} – {float(hi):.0f}")
m3.metric("🍽️ Cook this many", qty, delta=f"{qty - round(pred):+d} buffer vs forecast")

qs, costs = expected_cost_curve(pred, residuals, cu, co)
st.subheader("Expected cost vs plates cooked")
st.line_chart(pd.DataFrame({"Expected cost (Rs)": costs}, index=qs))
st.caption("The lowest point of the curve is the recommended quantity.")

if os.path.exists("results/summary.csv"):
    st.subheader("Backtest on held-out period")
    st.dataframe(pd.read_csv("results/summary.csv"), hide_index=True)
    if os.path.exists("results/cost_comparison.png"):
        st.image("results/cost_comparison.png")
