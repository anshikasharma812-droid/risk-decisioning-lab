
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Risk Decisioning Lab", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 3.5rem; padding-bottom: 3rem;}
.hero-title {font-size: clamp(2.8rem,7vw,5.6rem);line-height:1.08;font-weight:800;letter-spacing:-.045em;margin:.55rem 0 1rem;padding-top:.15rem;overflow:visible}
.eyebrow {text-transform:uppercase;letter-spacing:.16em;font-size:.78rem;font-weight:700;opacity:.7}
.subtitle {font-size:1.1rem;max-width:900px;opacity:.8}
.callout {padding:1rem 1.1rem;border:1px solid rgba(128,128,128,.25);border-radius:14px;background:rgba(128,128,128,.06);margin:1rem 0}
.action-flow {padding:1rem;border:1px solid rgba(128,128,128,.25);border-radius:14px;margin:.8rem 0 1.2rem}
.small {font-size:.88rem;opacity:.72}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def make_accounts(n=8000, seed=20260902):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 120, n)
    txn_velocity = np.clip(rng.lognormal(.1,.55,n), .2, 8)
    chargeback = np.clip(rng.beta(1.2,18,n),0,1)
    identity_anomaly = rng.beta(1.3,7,n)
    payment_anomaly = rng.beta(1.5,6,n)
    behavior_change = np.clip(rng.normal(.05,.38,n),-.8,2.2)
    exposure = np.clip(rng.lognormal(9.2,.8,n),2000,600000)

    z=(-3.5 + 1.0*(tenure<6) + 1.6*identity_anomaly + 1.8*payment_anomaly
       + 2.4*chargeback + .5*np.maximum(behavior_change,0) + .12*txn_velocity)
    latent=1/(1+np.exp(-z))
    bad=rng.binomial(1,latent).astype(bool)

    raw_score = (
        18
        + 52*latent
        + 12*identity_anomaly
        + 12*payment_anomaly
        + 16*chargeback
        + 5*np.maximum(behavior_change,0)
        + 2.0*txn_velocity
        + rng.normal(0,8,n)
    )
    score=np.clip(raw_score,0,100)

    segment=np.select([tenure<6, exposure>=200000],["New accounts","High exposure"],default="Established")
    return pd.DataFrame({
        "account_id":[f"ACCT-{i:05d}" for i in range(1,n+1)],
        "risk_score":score.round(1),"bad_outcome":bad,"tenure_months":tenure,
        "transaction_velocity":txn_velocity,"chargeback_rate":chargeback,
        "identity_anomaly":identity_anomaly,"payment_anomaly":payment_anomaly,
        "behavior_change":behavior_change,"exposure":exposure.round(0),"segment":segment
    })

df=make_accounts()

def assign_action(scores, verify_t, review_t, hold_t):
    return np.select(
        [scores>=hold_t, scores>=review_t, scores>=verify_t],
        ["HOLD","REVIEW","VERIFY"], default="ALLOW"
    )

def evaluate(data, verify_t, review_t, hold_t, loss, verify_cost, review_cost, hold_good_cost):
    d=data.copy()
    d["action"]=assign_action(d.risk_score,verify_t,review_t,hold_t)

    # Illustrative effectiveness of each intervention against a truly bad account.
    effectiveness={"ALLOW":0.0,"VERIFY":0.35,"REVIEW":0.70,"HOLD":0.95}
    op_cost={"ALLOW":0.0,"VERIFY":verify_cost,"REVIEW":review_cost,"HOLD":review_cost*0.5}

    prevented=0.0; friction=0.0; ops=0.0
    for action,g in d.groupby("action"):
        bad_n=int(g.bad_outcome.sum()); good_n=len(g)-bad_n
        prevented += bad_n*loss*effectiveness[action]
        ops += len(g)*op_cost[action]
        if action=="VERIFY": friction += good_n*verify_cost
        elif action=="REVIEW": friction += good_n*(verify_cost*1.6)
        elif action=="HOLD": friction += good_n*hold_good_cost

    bad=d.bad_outcome
    intervention=d.action!="ALLOW"
    tp=int((intervention & bad).sum()); fp=int((intervention & ~bad).sum())
    fn=int((~intervention & bad).sum()); tn=int((~intervention & ~bad).sum())
    recall=tp/max(tp+fn,1); precision=tp/max(tp+fp,1); fpr=fp/max(fp+tn,1)

    counts=d.action.value_counts().reindex(["ALLOW","VERIFY","REVIEW","HOLD"],fill_value=0)
    return d, {
        "recall":recall,"precision":precision,"fpr":fpr,"prevented":prevented,
        "friction":friction,"ops":ops,"net":prevented-friction-ops,"counts":counts
    }

st.markdown("""
<div class="eyebrow">Payments & Fraud Risk Strategy Simulation</div>
<div class="hero-title">Risk Decisioning Lab</div>
<div class="subtitle">
A risk model has detected unusual account behavior. Your job is not to decide whether to lend.
Your job is to decide <strong>what the platform should do next</strong>.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="callout">
<strong>Your mandate:</strong> translate model scores into a practical control policy that catches harmful activity
without unnecessarily disrupting legitimate customers.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="action-flow">
<strong>Decision ladder:</strong>&nbsp;&nbsp;
🟢 ALLOW &nbsp;→&nbsp; 🟡 VERIFY &nbsp;→&nbsp; 🟠 REVIEW &nbsp;→&nbsp; 🔴 HOLD
<br><span class="small">As risk rises, the intervention becomes stronger and customer friction increases.</span>
</div>
""", unsafe_allow_html=True)

st.subheader("Challenge 1 · Design the intervention policy")
st.write("Choose the score at which each intervention begins. The thresholds must become progressively stricter.")

c1,c2,c3=st.columns(3)
with c1:
    verify_t=st.slider("🟡 Verify from",20,75,50,1)
with c2:
    review_t=st.slider("🟠 Manual review from",35,90,70,1)
with c3:
    hold_t=st.slider("🔴 Hold from",50,99,88,1)

if not (verify_t < review_t < hold_t):
    st.error("Keep the policy ordered: VERIFY threshold < REVIEW threshold < HOLD threshold.")
    st.stop()

with st.expander("Business assumptions", expanded=False):
    a,b,c,d=st.columns(4)
    with a: loss=st.slider("Loss from missed harmful account",2000,30000,8000,1000,format="$%d")
    with b: verify_cost=st.slider("Verification cost / friction",0,500,75,25,format="$%d")
    with c: review_cost=st.slider("Manual review cost",0,500,120,20,format="$%d")
    with d: hold_good_cost=st.slider("Cost of holding a good account",100,3000,900,100,format="$%d")

decisions,r=evaluate(df,verify_t,review_t,hold_t,loss,verify_cost,review_cost,hold_good_cost)

st.markdown("#### What your policy does")
m1,m2,m3,m4=st.columns(4)
m1.metric("🟢 Allow",f"{r['counts']['ALLOW']:,}")
m2.metric("🟡 Verify",f"{r['counts']['VERIFY']:,}")
m3.metric("🟠 Review",f"{r['counts']['REVIEW']:,}")
m4.metric("🔴 Hold",f"{r['counts']['HOLD']:,}")

m1,m2,m3=st.columns(3)
m1.metric("Risk capture",f"{r['recall']:.1%}",help="Share of truly harmful accounts receiving some intervention.")
m2.metric("False-positive rate",f"{r['fpr']:.1%}",help="Share of legitimate accounts receiving an intervention.")
m3.metric("Net economic value",f"${r['net']/1_000_000:.2f}M")

if r["fpr"] > .35:
    st.warning("Your policy is very aggressive. Risk capture is strong, but a large share of legitimate accounts are being interrupted.")
elif r["recall"] < .60:
    st.info("Your policy protects customer experience, but a meaningful share of harmful activity is still passing through.")
else:
    st.success("Your policy is balancing risk capture and customer friction reasonably well in this simulation.")

st.divider()
st.subheader("Challenge 2 · See where the customers go")

score_fig = px.histogram(
    decisions,
    x="risk_score",
    nbins=40,
    labels={"risk_score":"Risk score"},
    title="Risk-score distribution"
)
score_fig.add_vline(x=verify_t, line_dash="dot")
score_fig.add_vline(x=review_t, line_dash="dash")
score_fig.add_vline(x=hold_t, line_dash="dashdot")
st.plotly_chart(score_fig, use_container_width=True)

dist=(decisions.groupby(["action","bad_outcome"]).size().reset_index(name="accounts"))
dist["Outcome"]=dist["bad_outcome"].map({True:"Harmful account",False:"Legitimate account"})
order=["ALLOW","VERIFY","REVIEW","HOLD"]
fig=px.bar(dist,x="action",y="accounts",color="Outcome",barmode="group",
           category_orders={"action":order},
           labels={"action":"Platform action","accounts":"Accounts"},
           title="Who is affected by your policy?")
st.plotly_chart(fig,use_container_width=True)

st.caption("A strong policy does not simply push more accounts to HOLD. It sends the right populations to proportionate interventions.")

st.divider()
st.subheader("Challenge 3 · Investigate a decision")

choices=["Highest-risk account","Borderline VERIFY case","Borderline REVIEW case","Borderline HOLD case"]
case_type=st.selectbox("Choose a case",choices)

targets={
    "Highest-risk account":decisions.risk_score.max(),
    "Borderline VERIFY case":verify_t,
    "Borderline REVIEW case":review_t,
    "Borderline HOLD case":hold_t,
}
target=targets[case_type]
idx=(decisions.risk_score-target).abs().idxmin()
acct=decisions.loc[idx]

left,right=st.columns([.75,1.25])
with left:
    st.metric("Risk score",f"{acct.risk_score:.0f}/100")
    st.metric("Decision",acct.action)
    st.write(f"**Segment:** {acct.segment}")
    st.write(f"**Tenure:** {acct.tenure_months} months")
    st.write(f"**Exposure:** ${acct.exposure:,.0f}")

with right:
    drivers=pd.DataFrame({
        "Signal":["Identity anomaly","Payment anomaly","Chargeback behavior","Behavior change"],
        "Signal strength":[acct.identity_anomaly,acct.payment_anomaly,acct.chargeback_rate,
                           max(0,min(1,(acct.behavior_change+.2)/1.5))]
    }).sort_values("Signal strength")
    fig2=px.bar(drivers,x="Signal strength",y="Signal",orientation="h",range_x=[0,1],
                title="Illustrative risk-signal profile")
    st.plotly_chart(fig2,use_container_width=True)

st.markdown("""
> **Risk strategist's question:** Is the intervention proportionate to the evidence, or are we creating unnecessary friction?
""")

st.divider()
st.subheader("Challenge 4 · Stress-test the control")

scenario=st.radio(
    "What changes?",
    ["Normal environment","Loss severity doubles","Customer friction becomes more expensive"],
    horizontal=True
)
sloss,shold=loss,hold_good_cost
if scenario=="Loss severity doubles": sloss=loss*2
if scenario=="Customer friction becomes more expensive": shold=hold_good_cost*2.5

_,stress=evaluate(df,verify_t,review_t,hold_t,sloss,verify_cost,review_cost,shold)
x,y,z=st.columns(3)
x.metric("Risk capture",f"{stress['recall']:.1%}")
y.metric("False-positive rate",f"{stress['fpr']:.1%}")
z.metric("Net value",f"${stress['net']/1_000_000:.2f}M",
         delta=f"${(stress['net']-r['net'])/1_000_000:+.2f}M vs base")

st.write("The same model scores can justify a different operating policy when the cost of losses, customer friction, or operational capacity changes.")

st.divider()
st.subheader("Decision Science")
with st.expander("What is a decision threshold?"):
    st.write("A threshold is the point where a model score becomes a business action. Here, multiple thresholds create a ladder of increasingly strong interventions.")
with st.expander("Why not just HOLD every high-risk account?"):
    st.write("Because legitimate customers can look risky too. Stronger controls may reduce losses but can also create false positives, support contacts, delays, appeals, and lost business.")
with st.expander("What is risk capture / recall?"):
    st.write("Of all accounts that truly produce a harmful outcome, how many received an intervention?")
with st.expander("What is a false positive?"):
    st.write("A legitimate account that receives a risk intervention. False positives are one of the central customer-experience costs of risk decisioning.")
with st.expander("Why synthetic data?"):
    st.write("Every account, signal, score, threshold, outcome, and economic assumption in this lab is simulated. No proprietary employer data, confidential model parameters, customer data, or production policies are used.")

st.divider()
st.markdown("### Next: Challenger Model & Shadow Mode")
st.write("A new model appears to outperform the incumbent. Should Risk deploy it immediately, or inspect where the two models disagree first?")
st.caption("Portfolio project · Payments risk · Fraud risk · Decision science · Model-to-policy translation")
