import pandas as pd
from aerthos_quant.api.supabase import fetch_table

def fetch_carbon_data():
    raw = fetch_table("carbon")
    df = pd.DataFrame(sorted(raw, key=lambda x: x['Date']))
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=False)
    return df