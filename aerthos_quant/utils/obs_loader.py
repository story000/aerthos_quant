# aerthos_quant/utils/ecb_loader.py

import pandas as pd
import requests
import numpy as np
from io import StringIO
from aerthos_quant.api.supabase import insert_data

ECB_URL = "https://data-api.ecb.europa.eu/service/data/FM/M.U2.EUR.RT.MM.EURIBOR3MD_.HSTA?format=csvdata"

def fetch_ecb_euribor_data() -> pd.DataFrame:
    response = requests.get(ECB_URL)
    data = StringIO(response.text)
    df = pd.read_csv(data)

    df = df[['TIME_PERIOD', 'OBS_VALUE']]
    df.rename(columns={'TIME_PERIOD': 'Month', 'OBS_VALUE': 'OBS_Value'}, inplace=True)
    
    return df

def load_to_supabase(table_name: str = "OBS"):
    df = fetch_ecb_euribor_data()
    insert_data(table_name, df.to_dict(orient='records'))
    print(f"✅ Successfully inserted {len(df)} rows into {table_name}.")
