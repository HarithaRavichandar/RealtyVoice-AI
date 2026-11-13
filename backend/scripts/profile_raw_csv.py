import pandas as pd, json, re, sys
from pathlib import Path
csv_path = Path(r'H:\RealtyVoice-AI\data\raw\house_prices.csv')
if not csv_path.exists():
    print('CSV file not found', file=sys.stderr)
    sys.exit(1)
# Read CSV
df = pd.read_csv(csv_path, dtype=str)
# Basic stats
row_count = len(df)
col_count = len(df.columns)
col_names = list(df.columns)
missing_per_col = df.isna().sum().to_dict()
duplicate_rows = int(df.duplicated().sum())
# Unique cities from location column (assuming column name "location")
unique_cities = []
if 'location' in df.columns:
    unique_cities = df['location'].dropna().unique().tolist()
# BHK distribution from Title column
bhk_dist = {}
if 'Title' in df.columns:
    for title in df['Title'].dropna():
        m = re.search(r'(\d+)\s*BHK', title, re.IGNORECASE)
        if m:
            bhk = int(m.group(1))
            bhk_dist[bhk] = bhk_dist.get(bhk,0)+1
# Price cleaning function
def clean_price(val):
    if pd.isna(val):
        return None
    s = str(val).replace(',', '').replace(' ', '')
    # handle units
    if 'Lac' in val:
        try:
            return float(s.replace('Lac','')) * 1e5
        except:
            return None
    if 'Cr' in val:
        try:
            return float(s.replace('Cr','')) * 1e7
        except:
            return None
    try:
        return float(s)
    except:
        return None
price_series = df['Amount(in rupees)'].apply(clean_price).dropna().astype(float) if 'Amount(in rupees)' in df.columns else pd.Series([], dtype=float)
price_min = price_series.min() if not price_series.empty else None
price_max = price_series.max() if not price_series.empty else None
price_median = price_series.median() if not price_series.empty else None
# Area fields sampling
area_fields = ['Carpet Area','Super Area','Plot Area']
area_samples = {col: df[col].dropna().head(3).tolist() for col in area_fields if col in df.columns}
# Representative rows (exclude sensitive columns like Title, Description)
rep_cols = [c for c in df.columns if c not in ('Title','Description')]
representative = df[rep_cols].head(10).to_dict(orient='records')
report = {
    'row_count': row_count,
    'col_count': col_count,
    'col_names': col_names,
    'missing_per_col': missing_per_col,
    'duplicate_rows': duplicate_rows,
    'unique_cities_total': len(unique_cities),
    'unique_cities_sample': unique_cities[:10],
    'bhk_distribution': bhk_dist,
    'price_min': price_min,
    'price_max': price_max,
    'price_median': price_median,
    'area_samples': area_samples,
    'representative_rows': representative,
}
print(json.dumps(report, indent=2, default=str))

