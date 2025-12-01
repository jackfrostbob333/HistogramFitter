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
    # Try multiple parsing strategies for badly formatted CSVs
    parse_attempts = [
        {'sep': ',', 'engine': 'python'},
        {'sep': ';', 'engine': 'python'},
        {'sep': '\t', 'engine': 'python'},
        {'sep': '\\s+', 'engine': 'python'},  # whitespace
        {'sep': ',', 'engine': 'python', 'skipinitialspace': True},
        {'sep': None, 'engine': 'python', 'delim_whitespace': True},
    ]
    
    df = None
    for attempt in parse_attempts:
        try:
            file_buffer.seek(0)
            df = pd.read_csv(file_buffer, **attempt, on_bad_lines='skip', encoding='utf-8')
            if not df.empty and len(df.columns) > 0:
                break
        except Exception:
            try:
                file_buffer.seek(0)
                df = pd.read_csv(file_buffer, **attempt, on_bad_lines='skip', encoding='latin-1')
                if not df.empty and len(df.columns) > 0:
                    break
            except Exception:
                continue
    
    if df is None or df.empty:
        return None, "Failed to parse CSV with any common delimiter"
    
    # Check if columns were concatenated into a single string (parsing failure)
    if len(df.columns) == 1:
        col_name = str(df.columns[0])
        # If the column name contains delimiter characters, reparse with the correct delimiter
        if ';' in col_name:
            # Semicolon-separated values
            file_buffer.seek(0)
            try:
                df = pd.read_csv(file_buffer, sep=';', engine='python', on_bad_lines='skip', encoding='utf-8')
            except Exception:
                pass
        elif '\t' in col_name or '  ' in col_name or len(col_name.split()) > 2:
            # Tab or whitespace delimiter
            file_buffer.seek(0)
            try:
                df = pd.read_csv(file_buffer, sep='\t', engine='python', on_bad_lines='skip', encoding='utf-8')
            except Exception:
                try:
                    file_buffer.seek(0)
                    df = pd.read_csv(file_buffer, sep='\s+', engine='python', on_bad_lines='skip', encoding='utf-8')
                except Exception:
                    pass
    
    # Clean column names (strip whitespace, remove quotes, remove special characters)
    df.columns = df.columns.astype(str).str.strip()
    df.columns = df.columns.str.replace(r'^"""(.*)"""$', r'\1', regex=True)
    df.columns = df.columns.str.replace(r'^"(.*)"$', r'\1', regex=True)
    df.columns = df.columns.str.strip()
    
    # Remove completely empty columns (from ugly_csv empty_columns)
    df = df.dropna(axis=1, how='all')
    df = df.loc[:, (df != '').any(axis=0)]  # Remove columns with only empty strings
    
    # Remove completely empty rows (from ugly_csv empty_rows)
    df = df.dropna(axis=0, how='all')
    df = df.loc[(df != '').any(axis=1)]  # Remove rows with only empty strings
    
    # Handle duplicate schema rows - remove rows where values look like column names
    if len(df) > 0:
        # Check if first few rows might be duplicate headers
        potential_header_rows = []
        for idx in range(min(5, len(df))):
            row_values = df.iloc[idx].astype(str).str.strip()
            # Remove quotes from row values for comparison
            row_values = row_values.str.replace(r'^"""(.*)"""$', r'\1', regex=True)
            row_values = row_values.str.replace(r'^"(.*)"$', r'\1', regex=True)
            row_values = row_values.str.strip().str.lower()
            col_names = df.columns.astype(str).str.strip().str.lower()
            # If row contains column names, mark for removal
            if any(val in col_names.tolist() for val in row_values if val and val != 'nan'):
                potential_header_rows.append(idx)
        if potential_header_rows:
            df = df.drop(potential_header_rows).reset_index(drop=True)
    
    # Clean cells from ugly_csv artefacts
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str)
            # Remove triple quotes (both single and repeated)
            df[col] = df[col].str.replace(r'^"""(.*)"""$', r'\1', regex=True)
            df[col] = df[col].str.replace(r'^"(.*)"$', r'\1', regex=True)
            # Strip whitespace after quote removal
            df[col] = df[col].str.strip()
            # Replace NaN-like artefacts: #RIF!, #N/A, #NULL!, ----, ////, etc.
            nan_patterns = ['#RIF!', '#N/A', '#NULL!', '#DIV/0!', '#VALUE!', '#REF!']
            for pattern in nan_patterns:
                df[col] = df[col].replace(pattern, np.nan)
            # Replace repeated dashes and slashes
            df[col] = df[col].replace(r'^-+$', np.nan, regex=True)
            df[col] = df[col].replace(r'^/+$', np.nan, regex=True)
            df[col] = df[col].replace(r'^\.*$', np.nan, regex=True)
            # Replace empty-looking strings (including ones with only spaces)
            df[col] = df[col].replace(r'^\s*$', np.nan, regex=True)
            # Handle weird tuple-like strings from satellite_artefacts
            df[col] = df[col].replace(r"^\('.*'\).*", np.nan, regex=True)
    
    if column is None:
        # try to auto-detect single numeric column
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            # Try to coerce text columns to numeric
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception:
                    pass
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) == 0:
                return None, "No numeric columns found or could be converted"
        col = numeric_cols[0]
    else:
        col = column
        if col not in df.columns:
            return None, f"Column '{col}' not in CSV. Available columns: {', '.join(df.columns)}"
        # Try to coerce column to numeric if it isn't already
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                return None, f"Column '{col}' could not be converted to numeric"
    
    # Extract data and remove NaN values
    data = df[col].dropna().values
    if len(data) == 0:
        return None, f"Column '{col}' contains no valid numeric data after cleaning"
    
    try:
        data = data.astype(float)
    except Exception:
        return None, f"Column '{col}' contains non-numeric values that could not be converted"
    
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

# Sidebar: data source selection and relevant inputs
with st.sidebar:
    st.header("📊 Data Input")
    
    # Data source selection
    data_source = st.radio(
        "Select data source:",
        ("Manual entry", "Upload CSV/TSV", "Generate sample"),
        index=2
    )
    
    st.markdown("---")
    
    # Initialize variables
    parsed = np.array([])
    csv_data = None
    csv_error = None
    
    # Show relevant options based on data source
    if data_source == "Manual entry":
        st.subheader("✍️ Manual Entry")
        text = st.text_area("Enter numbers (comma, space or newline separated):", height=200)
        parsed = parse_text_data(text)
        data = parsed
        
    elif data_source == "Upload CSV/TSV":
        st.subheader("📁 File Upload")
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "tsv", "txt"])
        
        if uploaded_file is not None:
            # allow selecting column
            try:
                # choose a sensible preview read depending on filename
                fname = getattr(uploaded_file, 'name', '') or ''
                fname_lower = fname.lower()
                uploaded_file.seek(0)
                if fname_lower.endswith('.tsv') or fname_lower.endswith('.txt'):
                    # try tab-separated preview first
                    try:
                        df_preview = pd.read_csv(uploaded_file, sep='\t', nrows=5)
                    except Exception:
                        uploaded_file.seek(0)
                        df_preview = pd.read_csv(uploaded_file, nrows=5, engine='python', on_bad_lines='skip')
                else:
                    # try comma first, then fallback to tab or python engine
                    try:
                        df_preview = pd.read_csv(uploaded_file, nrows=5)
                    except Exception:
                        uploaded_file.seek(0)
                        try:
                            df_preview = pd.read_csv(uploaded_file, sep='\t', nrows=5)
                        except Exception:
                            uploaded_file.seek(0)
                            df_preview = pd.read_csv(uploaded_file, nrows=5, engine='python', on_bad_lines='skip')

                uploaded_file.seek(0)
                # If preview produced a single column whose header contains multiple names
                # (e.g. whitespace/tab separated header), try reparsing using whitespace/tab separators
                if df_preview.shape[1] == 1:
                    col_name = str(df_preview.columns[0])
                    try:
                        # Check if the single column name contains delimiter characters
                        if ';' in col_name:
                            # Semicolon-separated
                            uploaded_file.seek(0)
                            df_preview = pd.read_csv(uploaded_file, sep=';', nrows=5, engine='python', on_bad_lines='skip')
                        elif '\t' in col_name or '  ' in col_name or len(col_name.split()) > 2:
                            # Tab or whitespace-separated
                            # peek at the first chunk to detect delimiter hints
                            uploaded_file.seek(0)
                            raw = uploaded_file.read(2048)
                            # uploaded_file.read returns bytes for Streamlit UploadedFile
                            if isinstance(raw, (bytes, bytearray)):
                                try:
                                    first_chunk = raw.decode('utf-8', errors='ignore')
                                except Exception:
                                    first_chunk = str(raw)
                            else:
                                first_chunk = str(raw)
                            uploaded_file.seek(0)
                            # prefer tab if present
                            if '\t' in first_chunk:
                                try:
                                    df_preview = pd.read_csv(uploaded_file, sep='\t', nrows=5, engine='python')
                                except Exception:
                                    uploaded_file.seek(0)
                                    df_preview = pd.read_csv(uploaded_file, sep=r"\s+", nrows=5, engine='python', on_bad_lines='skip')
                            # otherwise if whitespace-separated header appears
                            elif ' ' in first_chunk:
                                try:
                                    df_preview = pd.read_csv(uploaded_file, sep=r"\s+", nrows=5, engine='python', on_bad_lines='skip')
                                except Exception:
                                    pass
                            uploaded_file.seek(0)
                    except Exception:
                        try:
                            uploaded_file.seek(0)
                        except Exception:
                            pass
                cols = df_preview.columns.tolist()
                csv_col = st.selectbox("Select column", options=cols)
                uploaded_file.seek(0)
                csv_data, csv_error = load_csv(uploaded_file, column=csv_col)
                
                if csv_data is not None:
                    data = csv_data
                    st.success(f"✅ Loaded {len(data)} values")
                else:
                    st.error(f"❌ {csv_error}")
                    data = np.array([])
            except Exception as e:
                csv_data = None
                csv_error = str(e)
                st.error(f"❌ {csv_error}")
                data = np.array([])
        else:
            st.info("Upload a CSV or TSV file to begin")
            data = np.array([])
            
    else:  # Generate sample
        st.subheader("🎲 Sample Generator")
        sample_choice = st.selectbox(
            "Distribution type", 
            ["Normal", "Exponential", "Gamma", "Weibull", "Beta", "Lognormal", 
             "Uniform", "Chi-squared", "Student's t", "Pareto", "Bimodal (mixture)"]
        )
        sample_n = st.number_input("Sample size", min_value=10, max_value=100000, value=500, step=10)
        
        # Distribution-specific parameters
        if sample_choice == "Normal":
            mean = st.slider("Mean (μ)", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
            std = st.slider("Std Dev (σ)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        elif sample_choice == "Exponential":
            scale = st.slider("Scale (1/λ)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        elif sample_choice == "Gamma":
            shape = st.slider("Shape (k)", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
            scale = st.slider("Scale (θ)", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
        elif sample_choice == "Weibull":
            shape = st.slider("Shape (a)", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
            scale = st.slider("Scale", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        elif sample_choice == "Beta":
            alpha = st.slider("Alpha (α)", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
            beta = st.slider("Beta (β)", min_value=0.1, max_value=10.0, value=5.0, step=0.1)
        elif sample_choice == "Lognormal":
            mean = st.slider("Mean (log)", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
            sigma = st.slider("Sigma (log)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
        elif sample_choice == "Uniform":
            low = st.slider("Lower bound", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
            high = st.slider("Upper bound", min_value=low + 0.1, max_value=20.0, value=10.0, step=0.1)
        elif sample_choice == "Chi-squared":
            df = st.slider("Degrees of freedom", min_value=1, max_value=30, value=5, step=1)
        elif sample_choice == "Student's t":
            df = st.slider("Degrees of freedom", min_value=1, max_value=30, value=10, step=1)
        elif sample_choice == "Pareto":
            shape = st.slider("Shape (α)", min_value=0.5, max_value=5.0, value=2.5, step=0.1)
        elif sample_choice == "Bimodal (mixture)":
            mean1 = st.slider("Mean 1", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
            mean2 = st.slider("Mean 2", min_value=-10.0, max_value=10.0, value=5.0, step=0.1)
            std1 = st.slider("Std Dev 1", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
            std2 = st.slider("Std Dev 2", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
            mix_ratio = st.slider("Mix ratio (% from dist 1)", min_value=0, max_value=100, value=60, step=5)
        
        random_seed = st.number_input("Random seed", min_value=0, max_value=99999, value=12345, step=1)
        
        if st.button("Generate Sample", type="primary"):
            rng = np.random.default_rng(random_seed)
            
            if sample_choice == "Normal":
                parsed = rng.normal(loc=mean, scale=std, size=sample_n)
            elif sample_choice == "Exponential":
                parsed = rng.exponential(scale=scale, size=sample_n)
            elif sample_choice == "Gamma":
                parsed = rng.gamma(shape=shape, scale=scale, size=sample_n)
            elif sample_choice == "Weibull":
                parsed = scale * rng.weibull(a=shape, size=sample_n)
            elif sample_choice == "Beta":
                parsed = rng.beta(a=alpha, b=beta, size=sample_n)
            elif sample_choice == "Lognormal":
                parsed = rng.lognormal(mean=mean, sigma=sigma, size=sample_n)
            elif sample_choice == "Uniform":
                parsed = rng.uniform(low=low, high=high, size=sample_n)
            elif sample_choice == "Chi-squared":
                parsed = rng.chisquare(df=df, size=sample_n)
            elif sample_choice == "Student's t":
                parsed = rng.standard_t(df=df, size=sample_n)
            elif sample_choice == "Pareto":
                parsed = (rng.pareto(a=shape, size=sample_n) + 1)
            elif sample_choice == "Bimodal (mixture)":
                n1 = int(sample_n * mix_ratio / 100)
                n2 = sample_n - n1
                a = rng.normal(mean1, std1, size=n1)
                b = rng.normal(mean2, std2, size=n2)
                parsed = np.concatenate([a, b])
                rng.shuffle(parsed)
            
            # remember generated sample so we can mark it on the main plot
            st.session_state['generated'] = parsed
            st.session_state['generated_choice'] = sample_choice
            st.success(f"✅ Generated {len(parsed)} values from {sample_choice} distribution")
        
        if 'generated' in st.session_state:
            data = np.asarray(st.session_state['generated'])
        else:
            data = np.array([])
    
    st.markdown("---")
    st.subheader("📈 Histogram Settings")
    bins = st.slider("Number of bins", min_value=5, max_value=200, value=40)
    density = st.checkbox("Show density (PDF) histogram", value=True)

# Left column: data preview and info
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Data Preview")
    # Ensure data is always defined
    if data is None:
        data = np.array([])

    # Display data info
    if len(data) > 0:
        st.metric("Data Count", len(data))
        st.write("**First 20 values:**")
        st.write(np.round(data[:20], 5))
        
        # Basic statistics
        st.markdown("**Statistics:**")
        stats_col1, stats_col2 = st.columns(2)
        with stats_col1:
            st.metric("Mean", f"{np.mean(data):.4f}")
            st.metric("Min", f"{np.min(data):.4f}")
        with stats_col2:
            st.metric("Std Dev", f"{np.std(data):.4f}")
            st.metric("Max", f"{np.max(data):.4f}")
    else:
        st.info("No data loaded. Select a data source in the sidebar.")

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
                show_inset = st.checkbox('Show generated scatter inset', value=False)
                if show_inset:
                    try:
                        axins = inset_axes(ax, width='30%', height='30%', loc='upper left')
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