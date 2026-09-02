# Risk Decisioning Lab

An interactive Python / Streamlit simulation showing how risk-model signals become production controls.

## Core question

**A model sees risk. What should the platform do about it?**

Visitors design a proportional intervention ladder:

- ALLOW
- VERIFY
- MANUAL REVIEW
- HOLD

They can then observe the impact on risk capture, false positives, customer friction, operating load, and simulated economics.

This deliberately differs from a credit-underwriting project. It is about **model-to-policy translation and post-onboarding risk controls**, not whether to extend credit.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Data disclosure

All accounts, scores, signals, outcomes, thresholds, and economics are synthetic. The project uses no proprietary employer data, customer information, confidential model parameters, or production policies.

## Roadmap

- Challenger model / shadow mode
- Incumbent vs challenger disagreement analysis
- Calibration
- Drift monitoring
- Emerging-risk investigation
- Segmented controls
