# Histogram Fitter — Streamlit

This single-page Streamlit app allows you to load numeric data (paste or upload CSV), fit a variety of distributions from `scipy.stats`, visualize the fitted curves over a histogram, inspect fit parameters and quality metrics (MAE and Kolmogorov–Smirnov), and manually tune parameters with sliders.

Run locally:

```powershell
cd "c:\Users\souna\OneDrive - University of Waterloo\Desktop\NE111\Streamlit_Project\streamlit-website"
python -m pip install -r requirements.txt
streamlit run app.py
```

Features:
- Data input via CSV (first numeric column used) or pasted numbers
- At least 10 distributions available: Normal, Gamma, Weibull, Beta, Lognormal, Exponential, Chi-squared, Pareto, Rayleigh, Nakagami
- Visual overlay of fitted PDFs on the histogram
- Fit quality metrics: MAE (histogram vs pdf) and KS test statistic & p-value
- Manual slider tuning for distribution parameters

Synthetic data generator:
- Choose a distribution, set shape parameters (if applicable), `loc`, `scale`, and sample size.
- Generate synthetic samples with the chosen distribution and immediately fit distributions to those samples.
- A scatter plot of generated values vs measurement number is shown (like the class example).

Notes:
- If CSV contains multiple numeric columns, the first numeric column is used. You can paste data instead to choose custom values.
- Some distribution fits can fail or return unstable parameters depending on the data; the UI will show warnings when fits fail.
