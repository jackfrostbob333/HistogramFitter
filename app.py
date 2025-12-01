import io
from functools import partial
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

# app.py

import matplotlib.pyplot as plt
plt.style.use('dark_background')
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

st.set_page_config(layout="wide", page_title="Histogram Fitter")

# --- Utilities ---
def parse_text_data(text):
    if not text:
        return np.array([])
    try:
        # split on commas, whitespace, newlines
        parts = [p.strip() for p in text.replace(",", " ").split()]
        nums = [float(p) for p in parts if p != ""]
        return np.array(nums)
    except Exception:
        return np.array([])

def load_csv(file_buffer, column=None):
    try:
        df = pd.read_csv(file_buffer)
    except Exception:
        try:
            file_buffer.seek(0)
            df = pd.read_csv(file_buffer, sep=";")
        except Exception:
            return None, "Failed to parse CSV"
    if column is None:
        # try to auto-detect single numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            return None, "No numeric columns found"
        col = numeric_cols[0]
    else:
        col = column
        if col not in df.columns:
            return None, f"Column {col} not in CSV"
    data = df[col].dropna().values.astype(float)
    return data, None

def get_distribution_list():
    # Provide at least 10 distributions
    return {
        "Normal": stats.norm,
        "Exponential": stats.expon,
        "Gamma": stats.gamma,
        "Weibull (minimum)": stats.weibull_min,
        "Lognormal": stats.lognorm,
        "Beta": stats.beta,
        "Pareto": stats.pareto,
        "Uniform": stats.uniform,
        "Laplace": stats.laplace,
        "Logistic": stats.logistic,
        "Chi-squared": stats.chi2,
        "Student's t": stats.t,
    }

def fit_distribution(dist, data):
    try:
        params = dist.fit(data)
        return params, None
    except Exception as e:
        return None, str(e)

def params_dict_from_fit(dist, params):
    # scipy returns (*shape_args, loc, scale)
    shapes = getattr(dist, "shapes", None)
    result = {}
    if shapes:
        shape_names = [s.strip() for s in shapes.split(",")]
    else:
        shape_names = []
    n_shapes = len(shape_names)
    for i, name in enumerate(shape_names):
        result[name] = params[i]
    # loc and scale
    if len(params) >= 2:
        result["loc"] = params[-2]
        result["scale"] = params[-1]
    elif len(params) == 1:
        result["loc"] = params[-1]
    return result

def compute_metrics(dist, params, data, hist_bins=50):
    # histogram density
    hist_vals, bin_edges = np.histogram(data, bins=hist_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    args = tuple(params[:-2]) if len(params) >= 2 else tuple(params[:-1])
    loc = params[-2] if len(params) >= 2 else 0.0
    scale = params[-1] if len(params) >= 2 else 1.0
    try:
        pdf_vals = dist.pdf(bin_centers, *args, loc=loc, scale=scale)
    except Exception:
        # if pdf evaluation fails, return large errors
        return {"SSE": np.inf, "MAE": np.inf, "KS": np.inf}
    # Ensure pdf_vals shape matches
    if pdf_vals.shape != hist_vals.shape:
        pdf_vals = np.array(pdf_vals).reshape(hist_vals.shape)
    sse = np.sum((hist_vals - pdf_vals) ** 2)
    mae = np.mean(np.abs(hist_vals - pdf_vals))
    # Kolmogorov-Smirnov test
    try:
        cdf_func = lambda x: dist.cdf(x, *args, loc=loc, scale=scale)
        ks_stat, ks_p = stats.kstest(data, cdf_func)
    except Exception:
        ks_stat, ks_p = np.inf, 0.0
    return {"SSE": float(sse), "MAE": float(mae), "KS": float(ks_stat), "KS_p": float(ks_p)}

# --- App layout ---
st.title("Histogram Fitter")
st.markdown("Enter data manually or upload a CSV. Fit multiple distributions and compare fits. Use manual sliders to tweak parameters.")

# Left column: data input
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Data Input")

    # Text area for manual entry
    text = st.text_area("Enter numbers (comma, space or newline separated):", height=150)
    parsed = parse_text_data(text)

    # CSV upload
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    csv_col = None
    csv_data = None
    csv_error = None
    if uploaded_file is not None:
        # allow selecting column
        try:
            uploaded_file.seek(0)
            df_preview = pd.read_csv(uploaded_file, nrows=5)
            uploaded_file.seek(0)
            cols = df_preview.columns.tolist()
            csv_col = st.selectbox("Select column", options=cols)
            uploaded_file.seek(0)
            csv_data, csv_error = load_csv(uploaded_file, column=csv_col)
        except Exception as e:
            csv_data = None
            csv_error = str(e)

    # Option: sample data generator
    st.markdown("Or generate sample data")
    sample_choice = st.selectbox("Sample type", ["None", "Normal", "Gamma", "Weibull", "Mixture"])
    sample_n = st.number_input("Sample size", min_value=10, max_value=100000, value=500, step=10)
    if st.button("Generate sample"):
        rng = np.random.default_rng(12345)
        if sample_choice == "Normal":
            parsed = rng.normal(loc=0.0, scale=1.0, size=sample_n)
        elif sample_choice == "Gamma":
            parsed = rng.gamma(shape=2.0, scale=2.0, size=sample_n)
        elif sample_choice == "Weibull":
            parsed = rng.weibull(a=1.5, size=sample_n)
        elif sample_choice == "Mixture":
            a = rng.normal(0, 1, size=int(sample_n * 0.6))
            b = rng.normal(5, 1, size=sample_n - len(a))
            parsed = np.concatenate([a, b])
        elif sample_choice == "None":
            parsed = rng.random(size=sample_n)
        # remember generated sample so we can mark it on the main plot
        st.session_state['generated'] = parsed
        st.session_state['generated_choice'] = sample_choice

    # Final data selection
    data_source = st.radio("Data source", ("Manual entry", "CSV/uploaded", "Generated/Parsed"), index=2)
    if data_source == "Manual entry":
        data = parsed
    elif data_source == "CSV/uploaded":
        if csv_data is None:
            st.warning(f"CSV not loaded: {csv_error}" if csv_error else "Upload a CSV and select a column.")
            data = np.array([])
    elif data_source == "Generated/Parsed":
        if 'generated' in st.session_state:
            data = np.array(st.session_state['generated'])
        else:
            data = csv_data
    else:
        # prefer CSV if available, else manual parsed
        if csv_data is not None:
            data = csv_data
        else:
            data = parsed

    if data is None:
        data = np.array([])

    st.write(f"Data count: {len(data)}")
    if len(data) > 0:
        st.write("Preview:", np.round(data[:20], 5))

    st.subheader("Histogram settings")
    bins = st.slider("Number of bins", min_value=5, max_value=200, value=40)
    density = st.checkbox("Show density (PDF) histogram", value=True)

# Right column: fitting options and plot
with col2:
    st.subheader("Fitting Options")

    dists = get_distribution_list()
    dist_names = list(dists.keys())
    selected = st.multiselect("Select distributions to fit (choose 1 or more)", options=dist_names, default=["Normal"])

    st.write("After selecting distributions, click Fit Distributions.")
    fit_button = st.button("Fit Distributions")

    manual_mode = st.checkbox("Manual fitting mode (show sliders)", value=False)

    # Prepare plot area
    fig, ax = plt.subplots(figsize=(8, 5))

    # If no data, show message
    if len(data) == 0:
        st.info("No data available. Enter data or upload a CSV, then click Fit Distributions or generate sample data.")
        st.pyplot(fig)
    else:
        # draw histogram
        ax.hist(data, bins=bins, density=density, alpha=0.4, edgecolor="k")
        x_min, x_max = np.min(data), np.max(data)
        x_margin = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
        x = np.linspace(x_min - x_margin, x_max + x_margin, 1000)

        fit_results = {}
        colors = plt.cm.tab10.colors
        with st.spinner('Fitting distributions...'):
            for idx, name in enumerate(selected):
                dist = dists[name]
                params = None
                fit_err = None
                if fit_button and not manual_mode:
                    params, fit_err = fit_distribution(dist, data)
                    if params is None:
                        st.warning(f"Failed to fit {name}: {fit_err}")
                        continue
                elif manual_mode:
                    # use default fit to provide slider defaults
                    params, fit_err = fit_distribution(dist, data)
                    if params is None:
                        # fallback defaults
                        params = tuple()
                else:
                    # not fitting yet, skip
                    continue

                # Build param mapping
                # params is tuple: (*shapes, loc, scale) or similar
                pdict = params_dict_from_fit(dist, params)
                metrics = compute_metrics(dist, params, data, hist_bins=bins)
                fit_results[name] = {"dist": dist, "params": params, "pdict": pdict, "metrics": metrics, "color": colors[idx % len(colors)]}

        # Manual sliders: allow editing parameters for one selected distribution at a time
        manual_override = {}
        if manual_mode and selected:
            st.subheader("Manual parameter adjustment")
            manual_choice = st.selectbox("Choose distribution to adjust", options=selected)
            dist = dists[manual_choice]
            # get current fitted params if available
            fitted = fit_results.get(manual_choice)
            default_params = None
            if fitted:
                default_params = fitted["params"]
            else:
                p_try, _ = fit_distribution(dist, data)
                default_params = p_try if p_try is not None else ()
            shapes = getattr(dist, "shapes", None)
            sliders_values = []
            # parse shapes
            shape_names = [s.strip() for s in shapes.split(",")] if shapes else []
            # Create sliders for shape parameters
            for i, sname in enumerate(shape_names):
                default = float(default_params[i]) if i < len(default_params) else 1.0
                val = st.slider(f"{manual_choice} shape: {sname}", min_value=float(default*0.1 if default!=0 else -10.0), max_value=float(default*10 if default!=0 else 10.0), value=float(default), step=float(abs(default)*0.01 if abs(default)>0 else 0.01))
                sliders_values.append(val)
            # loc and scale
            # loc default
            if len(default_params) >= 2:
                default_loc = float(default_params[-2])
                default_scale = float(default_params[-1])
            elif len(default_params) == 1:
                default_loc = float(default_params[-1])
                default_scale = 1.0
            else:
                default_loc = 0.0
                default_scale = 1.0
            loc_val = st.slider(f"{manual_choice} loc", min_value=float(default_loc - abs(default_loc)*5 - 10), max_value=float(default_loc + abs(default_loc)*5 + 10), value=float(default_loc), step=max(abs(default_loc)*0.01, 0.01))
            scale_val = st.slider(f"{manual_choice} scale", min_value=1e-6, max_value=float(max(abs(default_scale)*10, 1.0)), value=float(default_scale), step=max(abs(default_scale)*0.01, 0.01))
            manual_override[manual_choice] = tuple(list(sliders_values) + [loc_val, scale_val])

        # Plot fitted PDFs
        for name, info in fit_results.items():
            dist = info["dist"]
            params = info["params"]
            # if manual override present for this dist, replace params
            if name in manual_override:
                params = manual_override[name]
            # unpack args
            if len(params) >= 2:
                args = tuple(params[:-2])
                loc = params[-2]
                scale = params[-1]
            elif len(params) == 1:
                args = ()
                loc = params[0]
                scale = 1.0
            else:
                args = ()
                loc = 0.0
                scale = 1.0
            try:
                y = dist.pdf(x, *args, loc=loc, scale=scale)
                ax.plot(x, y, label=name, color=info["color"], lw=2)
            except Exception:
                st.warning(f"Failed to evaluate PDF for {name} with params {params}")

        ax.legend()
        ax.set_xlabel("Value")
        ax.set_ylabel("Density" if density else "Count")
        # Overlay generated sample ticks (rug) and optional inset scatter if we have generated samples
        if st.session_state.get('generated') is not None:
            try:
                gdata = np.asarray(st.session_state['generated'])
                ymin, ymax = ax.get_ylim()
                y_tick = ymin + 0.02 * (ymax - ymin)
                ax.plot(gdata, np.full_like(gdata, y_tick), '|', color='white', markersize=8, alpha=0.9, label='generated samples')
                show_inset = st.checkbox('Show generated scatter inset', value=True)
                if show_inset:
                    try:
                        axins = inset_axes(ax, width='30%', height='30%', loc=1)
                        axins.plot(gdata, 'k.', markersize=2)
                        axins.set_xticks([])
                        axins.set_yticks([])
                        axins.set_title('Generated samples', color='white', fontsize=8)
                    except Exception:
                        pass
            except Exception:
                pass
        st.pyplot(fig)

        # Show detailed fit results
        if fit_results:
            st.subheader("📊 Fit Results & Quality Metrics")
            
            # Prepare downloadable summary
            summary_rows = []
            
            # Create expandable sections for each distribution
            for name, info in fit_results.items():
                params = info["params"]
                pdict = info["pdict"]
                metrics = info["metrics"]
                
                # if manual override applied, show those
                if name in manual_override:
                    overridden = manual_override[name]
                    pdict = params_dict_from_fit(info["dist"], overridden)
                    metrics = compute_metrics(info["dist"], overridden, data, hist_bins=bins)
                
                # Use expander for cleaner layout
                with st.expander(f"📈 **{name} Distribution**" + (" 🎛️ (Manual)" if name in manual_override else ""), expanded=True):
                    # Parameters section
                    st.markdown("##### Distribution Parameters")
                    param_cols = st.columns(min(len(pdict), 4))
                    for idx, (k, v) in enumerate(pdict.items()):
                        with param_cols[idx % len(param_cols)]:
                            st.metric(label=k.capitalize(), value=f"{v:.4f}")
                    
                    st.markdown("---")
                    
                    # Quality metrics section with color coding
                    st.markdown("##### Goodness of Fit Metrics")
                    metric_cols = st.columns(4)
                    
                    with metric_cols[0]:
                        sse_val = metrics.get("SSE", np.inf)
                        st.metric(
                            label="SSE",
                            value=f"{sse_val:.6f}" if sse_val != np.inf else "∞",
                            help="Sum of Squared Errors: Lower is better. Measures squared differences between histogram and fit."
                        )
                    
                    with metric_cols[1]:
                        mae_val = metrics.get("MAE", np.inf)
                        st.metric(
                            label="MAE",
                            value=f"{mae_val:.6f}" if mae_val != np.inf else "∞",
                            help="Mean Absolute Error: Lower is better. Average absolute difference between histogram and fit."
                        )
                    
                    with metric_cols[2]:
                        ks_val = metrics.get("KS", np.inf)
                        st.metric(
                            label="KS Statistic",
                            value=f"{ks_val:.6f}" if ks_val != np.inf else "∞",
                            help="Kolmogorov-Smirnov test statistic: Lower is better (< 0.05 typical). Measures max distance between empirical and theoretical CDFs."
                        )
                    
                    with metric_cols[3]:
                        ks_p = metrics.get("KS_p", 0.0)
                        ks_quality = "✅ Good" if ks_p > 0.05 else "⚠️ Poor" if ks_p > 0.01 else "❌ Bad"
                        st.metric(
                            label="KS p-value",
                            value=f"{ks_p:.4f}",
                            delta=ks_quality,
                            help="Kolmogorov-Smirnov p-value: Higher is better (> 0.05 means good fit). Tests if data could come from this distribution."
                        )
                    
                    # Overall fit quality indicator
                    if ks_p > 0.05:
                        st.success("✅ **Strong fit** — Data is statistically consistent with this distribution")
                    elif ks_p > 0.01:
                        st.warning("⚠️ **Moderate fit** — Some deviation from ideal distribution")
                    else:
                        st.error("❌ **Poor fit** — Data does not match this distribution well")
                
                # collect for CSV
                try:
                    pairs = []
                    for k, v in pdict.items():
                        pairs.append(f"{k}={v:.6f}")
                    summary_rows.append({
                        "distribution": name,
                        "params": "; ".join(pairs),
                        "SSE": metrics.get("SSE", None),
                        "MAE": metrics.get("MAE", None),
                        "KS_statistic": metrics.get("KS", None),
                        "KS_p_value": metrics.get("KS_p", None)
                    })
                except Exception:
                    pass
            
            # Download button
            if summary_rows:
                st.markdown("---")
                df_summary = pd.DataFrame(summary_rows)
                csv = df_summary.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Complete Results (CSV)",
                    data=csv,
                    file_name='histogram_fit_results.csv',
                    mime='text/csv',
                    help="Download all fitting parameters and quality metrics as a CSV file"
                )