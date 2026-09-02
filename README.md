# Risk Decisioning Lab — Python / Streamlit

An interactive risk strategy simulation built in Python with Streamlit.

## What the user can do

- Tune a model intervention threshold
- See recall, precision, false-positive rate, review load, avoided loss, and net value
- Change the business economics and watch the optimal threshold move
- Compare the global policy with segment-level economics
- Inspect synthetic account-level decisions
- Learn the underlying decision-science concepts

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deploy with Streamlit Community Cloud

1. Create a GitHub repository named `risk-decisioning-lab`.
2. Upload:
   - `streamlit_app.py`
   - `requirements.txt`
   - `README.md`
3. Push to the `main` branch.
4. Sign in to Streamlit Community Cloud with GitHub.
5. Create an app and select your repository.
6. Use `streamlit_app.py` as the entrypoint.
7. Choose your public app subdomain and deploy.

## Why Streamlit

This project is simulation-heavy rather than content-heavy. Python makes it easier to add:

- segmented policies
- challenger models
- shadow-mode analysis
- calibration
- drift monitoring
- feature distributions
- anomaly investigation
- model comparison
- scenario simulation

## Data disclosure

All portfolio data is synthetic. No proprietary employer data, confidential model parameters, customer information, or production thresholds are used.
