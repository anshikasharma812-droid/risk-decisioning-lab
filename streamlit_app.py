
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Risk Decisioning Lab",
    page_icon="🎯",
    layout="wide",
)

# ----------------------------
# Styling
# ----------------------------
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 3rem;}
      .hero {
        padding: 1.5rem 0 1rem 0;
      }
      .eyebrow {
        text-transform: uppercase;
        letter-spacing: .16em;
        font-size: .78rem;
        font-weight: 700;
        opacity: .75;
      }
      .hero-title {
        font-size: clamp(2.8rem, 7vw, 5.8rem);
        line-height: .95;
        font-weight: 800;
        letter-spacing: -.055em;
        margin: .3rem 0 1rem;
      }
      .hero-subtitle {
        font-size: 1.1rem;
        max-width: 850px;
        opacity: .8;
      }
      .callout {
        padding: 1rem 1.1rem;
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 14px;
        background: rgba(128,128,128,.06);
        margin: .6rem 0 1rem;
      }
      .strategy {
        padding: 1rem 1.1rem;
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 14px;
      }
      .small {font-size:.88rem; opacity:.75;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Synthetic portfolio
# ----------------------------
@st.cache_data
def make_portfolio(n=8000, seed=20260902):
    rng = np.random.default_rng(seed)

    tenure_months = rng.integers(1, 120, n)
    exposure = np.clip(rng.lognormal(mean=9.5, sigma=0.75, size=n), 5_000, 750_000)
    utilization = np.clip(rng.beta(2.2, 2.0, n), 0, 1)
    payment_stress = rng.beta(1.5, 5.5, n)
    velocity_change = np.clip(rng.normal(0.05, 0.35, n), -0.8, 2.0)

    # Latent probability of bad outcome
    z = (
        -3.0
        + 2.2 * utilization
        + 2.5 * payment_stress
        + 0.8 * np.maximum(velocity_change, 0)
        + 0.45 * (tenure_months < 6)
        + 0.25 * (exposure > 250_000)
    )
    pd_true = 1 / (1 + np.exp(-z))
    bad = rng.binomial(1, pd_true)

    # Imperfect model score: correlated with underlying risk but noisy
    score = np.clip(
        100 * (
            0.10
            + 0.52 * pd_true
            + 0.12 * utilization
            + 0.10 * payment_stress
            + rng.normal(0, 0.07, n)
        ),
        0,
        100,
    )

    segment = np.select(
        [
            tenure_months < 6,
            exposure >= 250_000,
        ],
        [
            "New accounts",
            "High exposure",
        ],
        default="Established",
    )

    return pd.DataFrame(
        {
            "account_id": [f"ACCT-{i:05d}" for i in range(1, n + 1)],
            "risk_score": score.round(1),
            "bad_outcome": bad.astype(bool),
            "tenure_months": tenure_months,
            "exposure": exposure.round(0),
            "utilization": utilization,
            "payment_stress": payment_stress,
            "velocity_change": velocity_change,
            "segment": segment,
        }
    )

df = make_portfolio()

def evaluate(data, threshold, loss_per_bad, fp_cost, review_cost):
    flagged = data["risk_score"] >= threshold
    bad = data["bad_outcome"]

    tp = int((flagged & bad).sum())
    fp = int((flagged & ~bad).sum())
    tn = int((~flagged & ~bad).sum())
    fn = int((~flagged & bad).sum())

    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    fpr = fp / max(fp + tn, 1)
    reviews = tp + fp

    loss_prevented = tp * loss_per_bad
    friction_cost = fp * fp_cost
    review_expense = reviews * review_cost
    net_value = loss_prevented - friction_cost - review_expense

    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "recall": recall,
        "precision": precision,
        "fpr": fpr,
        "reviews": reviews,
        "loss_prevented": loss_prevented,
        "friction_cost": friction_cost,
        "review_expense": review_expense,
        "net_value": net_value,
    }

def best_threshold(data, loss_per_bad, fp_cost, review_cost):
    rows = [
        evaluate(data, t, loss_per_bad, fp_cost, review_cost)
        for t in range(20, 96)
    ]
    frontier = pd.DataFrame(rows)
    best = frontier.loc[frontier["net_value"].idxmax()].to_dict()
    return best, frontier

# ----------------------------
# Hero
# ----------------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Interactive Risk Strategy Simulation</div>
      <div class="hero-title">Risk Decisioning Lab</div>
      <div class="hero-subtitle">
        A model can rank risk. Your job is harder: decide when that score should trigger action.
        Tune the policy and watch risk capture, false positives, customer friction, operating load,
        and economic value move in real time.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="callout">
      <strong>Your mandate:</strong> prevent losses without unnecessarily disrupting legitimate customers.
      The goal is not to maximize a single model metric. It is to find the best risk-adjusted decision.
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Challenge 1
# ----------------------------
st.subheader("Challenge 1 · Choose the intervention threshold")

left, right = st.columns([0.75, 1.25], gap="large")

with left:
    threshold = st.slider(
        "Risk score threshold",
        min_value=20,
        max_value=95,
        value=70,
        step=1,
        help="Accounts scoring at or above this value receive the risk intervention.",
    )

    st.caption("Lower threshold → catch more risk, but affect more good customers.")

    st.markdown("#### Business assumptions")
    loss_per_bad = st.slider(
        "Loss if a bad account is missed",
        2_000,
        30_000,
        8_000,
        1_000,
        format="$%d",
    )
    fp_cost = st.slider(
        "Cost of a false positive",
        0,
        2_000,
        350,
        50,
        format="$%d",
    )
    review_cost = st.slider(
        "Operational review cost per flagged account",
        0,
        250,
        35,
        5,
        format="$%d",
    )

result = evaluate(df, threshold, loss_per_bad, fp_cost, review_cost)
best, frontier = best_threshold(df, loss_per_bad, fp_cost, review_cost)

with right:
    c1, c2, c3 = st.columns(3)
    c1.metric("Risk capture / Recall", f"{result['recall']:.1%}")
    c2.metric("Precision", f"{result['precision']:.1%}")
    c3.metric("False-positive rate", f"{result['fpr']:.1%}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Accounts reviewed", f"{result['reviews']:,}")
    c5.metric("Loss prevented", f"${result['loss_prevented']/1_000_000:.2f}M")
    c6.metric("Net economic value", f"${result['net_value']/1_000_000:.2f}M")

    delta = threshold - int(best["threshold"])

    if abs(delta) <= 2:
        st.success(
            f"Strong risk-adjusted choice. Under these assumptions, the simulated economic "
            f"optimum is around **{int(best['threshold'])}**."
        )
    elif delta < 0:
        st.warning(
            f"You may be **over-intervening**. The simulated economic optimum is around "
            f"**{int(best['threshold'])}**. You're catching additional risk, but incremental "
            f"false positives and reviews are reducing value."
        )
    else:
        st.info(
            f"You may be **leaving loss prevention on the table**. The simulated economic "
            f"optimum is around **{int(best['threshold'])}**."
        )

# ----------------------------
# Frontier
# ----------------------------
st.divider()
st.subheader("Challenge 2 · Explore the decision frontier")

plot_df = frontier.copy()
plot_df["Risk capture"] = plot_df["recall"] * 100
plot_df["False-positive rate"] = plot_df["fpr"] * 100
plot_df["Net value ($M)"] = plot_df["net_value"] / 1_000_000

a, b = st.columns(2, gap="large")

with a:
    long = plot_df.melt(
        id_vars=["threshold"],
        value_vars=["Risk capture", "False-positive rate"],
        var_name="Metric",
        value_name="Percent",
    )
    fig1 = px.line(
        long,
        x="threshold",
        y="Percent",
        color="Metric",
        markers=False,
        labels={"threshold": "Decision threshold"},
        title="Risk capture vs customer friction",
    )
    fig1.add_vline(x=threshold, line_dash="dash")
    st.plotly_chart(fig1, use_container_width=True)

with b:
    fig2 = px.line(
        plot_df,
        x="threshold",
        y="Net value ($M)",
        title="Estimated net economic value",
        labels={"threshold": "Decision threshold"},
    )
    fig2.add_vline(x=threshold, line_dash="dash")
    fig2.add_vline(x=int(best["threshold"]), line_dash="dot")
    st.plotly_chart(fig2, use_container_width=True)

st.caption(
    "The dotted optimum moves when the economics change. That is the point: "
    "threshold selection is a policy decision, not just a model-performance decision."
)

# ----------------------------
# Segment drilldown
# ----------------------------
st.divider()
st.subheader("Challenge 3 · Is one threshold right for everyone?")

selected_segment = st.selectbox(
    "Inspect a customer segment",
    ["Entire portfolio"] + sorted(df["segment"].unique().tolist()),
)

seg_df = df if selected_segment == "Entire portfolio" else df[df["segment"] == selected_segment]
seg_result = evaluate(seg_df, threshold, loss_per_bad, fp_cost, review_cost)
seg_best, _ = best_threshold(seg_df, loss_per_bad, fp_cost, review_cost)

s1, s2, s3, s4 = st.columns(4)
s1.metric("Accounts", f"{len(seg_df):,}")
s2.metric("Observed bad rate", f"{seg_df['bad_outcome'].mean():.1%}")
s3.metric("Current recall", f"{seg_result['recall']:.1%}")
s4.metric("Segment optimum", f"{int(seg_best['threshold'])}")

if selected_segment != "Entire portfolio":
    diff = int(seg_best["threshold"]) - int(best["threshold"])
    if abs(diff) >= 3:
        st.markdown(
            f"""
            <div class="strategy">
              <strong>Interesting:</strong> the economically optimal threshold for
              <strong>{selected_segment}</strong> differs from the global optimum by
              <strong>{diff:+d} points</strong>.
              This is the beginning of segmented risk decisioning.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="strategy">
              This segment behaves similarly to the overall portfolio under the current assumptions.
            </div>
            """,
            unsafe_allow_html=True,
        )

# ----------------------------
# Inspect individual accounts
# ----------------------------
st.divider()
st.subheader("Inspect the decision")

flagged_df = df.assign(
    decision=np.where(df["risk_score"] >= threshold, "Intervene", "Pass")
).sort_values("risk_score", ascending=False)

sample = flagged_df[
    [
        "account_id",
        "segment",
        "risk_score",
        "decision",
        "exposure",
        "tenure_months",
        "utilization",
        "payment_stress",
        "velocity_change",
    ]
].head(30)

st.dataframe(
    sample,
    use_container_width=True,
    hide_index=True,
    column_config={
        "exposure": st.column_config.NumberColumn("Exposure", format="$%.0f"),
        "utilization": st.column_config.ProgressColumn("Utilization", min_value=0, max_value=1),
        "payment_stress": st.column_config.ProgressColumn("Payment stress", min_value=0, max_value=1),
        "velocity_change": st.column_config.NumberColumn("Velocity change", format="%.2f"),
    },
)

# ----------------------------
# Science
# ----------------------------
st.divider()
st.subheader("Decision Science")

with st.expander("Recall"):
    st.write("Of all genuinely risky accounts, what percentage did the control catch?")

with st.expander("Precision"):
    st.write("Of all accounts flagged as risky, what percentage actually had a bad outcome?")

with st.expander("False positives"):
    st.write(
        "Legitimate customers incorrectly subjected to a risk intervention. "
        "In real systems this can create friction, appeals, support volume, and lost revenue."
    )

with st.expander("Why the economic optimum matters"):
    st.write(
        "A lower threshold can improve recall while simultaneously reducing business value if "
        "the marginal risk captured costs too much in false positives and operating effort."
    )

with st.expander("Why this uses synthetic data"):
    st.write(
        "The portfolio, scores, outcomes, thresholds, and economics are simulated. "
        "The project demonstrates risk-decisioning concepts without using proprietary employer "
        "data, customer information, confidential models, or production policies."
    )

st.divider()
st.caption(
    "Portfolio project · Risk strategy · Decision science · Model thresholding · FinTech"
)
