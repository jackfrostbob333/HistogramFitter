# Histogram Fitter — Streamlit

A comprehensive single-page Streamlit app for statistical distribution fitting with enhanced data input flexibility and robust parsing capabilities.

## Quick Start

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

## Features

### Data Input Options (Sidebar)
- **Manual Entry**: Paste numbers directly (comma/space separated)
- **Upload CSV/TSV**: Supports multiple delimiters (comma, semicolon, tab, whitespace)
  - Handles badly formatted CSVs with:
    - Triple-quoted values (`"""value"""`)
    - Error codes (#VALUE!, #REF!, #N/A, #NULL!, #DIV/0!, #RIF!)
    - Empty columns/rows
    - Duplicate header rows
    - Mixed delimiters
    - Dash/slash artifacts (-----, /////, .....)
    - Extra whitespace and quotes
  - File types: .csv, .tsv, .txt
  - Column selection from preview
- **Generate Samples**: Create synthetic data from 11 distributions with custom parameters
  - Normal, Exponential, Gamma, Weibull, Beta, Lognormal, Uniform
  - Chi-squared, Student's t, Pareto, Bimodal mixture
  - Configurable shape, scale, location parameters
  - Sample size and random seed control

### Distribution Fitting
- **12 Distributions Available**: Normal, Exponential, Gamma, Weibull (minimum), Lognormal, Beta, Pareto, Uniform, Laplace, Logistic, Chi-squared, Student's t
- **Automatic Fitting**: Uses scipy.stats maximum likelihood estimation
- **Manual Parameter Tuning**: Slider controls for fine-tuning fit parameters

### Visualization
- Histogram with overlaid fitted distribution PDF
- Optional scatter plot (index vs value) with inset positioning
- Dark theme styling

### Quality Metrics
- **SSE** (Sum of Squared Errors): Histogram vs PDF
- **MAE** (Mean Absolute Error): Average deviation
- **KS Statistic**: Kolmogorov-Smirnov test statistic
- **KS p-value**: Statistical significance of fit
- Color-coded metric cards with expandable details

## Robust CSV Parsing

The app automatically detects and handles:
- **Delimiters**: Comma, semicolon, tab, whitespace
- **Encodings**: UTF-8, Latin-1 fallback
- **Ugly CSV artifacts**: Removes noise and error codes
- **Smart detection**: Retries parsing if initial delimiter detection fails

## Notes

- For CSV uploads with multiple numeric columns, select the desired column from the preview
- Some distributions may fail to fit certain datasets; warnings will be displayed
- Manual parameter adjustment is available when automatic fitting succeeds
- Generated samples use scipy.stats distributions for ground truth comparison
