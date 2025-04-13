import http.client
import json
import pandas as pd
from datetime import datetime
import os
import numpy as np
from dotenv import load_dotenv
from aerthos_quant.api.supabase import insert_data, fetch_table

load_dotenv()

capital_url = "api-capital.backend-capital.com"


def connect_capital():
    conn = http.client.HTTPSConnection(capital_url)
    payload = json.dumps({
    "identifier": os.getenv("CAPITAL_IDENTIFIER"),
    "password": os.getenv("CAPITAL_PASSWORD")
    })

    headers = {
    'X-CAP-API-KEY': os.getenv("CAPITAL_API_KEY"),
    'Content-Type': 'application/json'
    }

    conn.request("POST", "/api/v1/session", payload, headers)
    res = conn.getresponse()

    cst = res.getheader('CST')
    x_security_token = res.getheader('X-SECURITY-TOKEN')
    return cst, x_security_token


def get_day_price(cst, x_security_token, symbol, start_date, end_date):
    conn = http.client.HTTPSConnection(capital_url)
    payload = ''
    headers = {
    'X-SECURITY-TOKEN': x_security_token,
    'CST': cst
    }

    max_days = (datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%S") - datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%S")).days + 1
    request = f"/api/v1/prices/{symbol}?resolution=DAY&max={max_days}&from={start_date}&to={end_date}"
    conn.request("GET", request, payload, headers)
    res = conn.getresponse()
    data = res.read()
    return data.decode("utf-8")

def fetch_ecb_euribor_data(last_date: str) -> pd.DataFrame:
    cst, x_security_token = connect_capital()
    data = get_day_price(cst, x_security_token, "ECFZ2025", last_date, datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
    data_json = json.loads(data)
    if 'prices' not in data_json:
        return None
    df = pd.DataFrame(data_json['prices'])
    df['Date'] = pd.to_datetime(df['snapshotTime']).dt.strftime('%Y-%m-%d')
    df['Open'] = df['openPrice'].apply(lambda x: x['bid'])
    df['Close'] = df['closePrice'].apply(lambda x: x['bid'])    
    df['High'] = df['highPrice'].apply(lambda x: x['bid'])
    df['Low'] = df['lowPrice'].apply(lambda x: x['bid'])
    df['Vol'] = df['lastTradedVolume']
    df['pct_change'] = (df["Close"] / df["Close"].shift(1) - 1)
    df.drop(columns=['openPrice', 'closePrice', 'highPrice', 'lowPrice', 'lastTradedVolume', 'snapshotTime','snapshotTimeUTC'], inplace=True) 
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%m/%d/%Y')

    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%m/%d/%Y')
    df.rename(columns={'Close': 'Price'}, inplace=True)
    
    return df

def load_to_supabase(table_name: str = "carbon"):
    old_df = fetch_table(table_name)
    last_date = (datetime.strptime(old_df[-1]['Date'], '%Y-%m-%d') + pd.Timedelta(hours=6)).strftime('%Y-%m-%dT%H:%M:%S') 
    df = fetch_ecb_euribor_data(last_date)
    if df is not None:
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        insert_data(table_name, df.to_dict(orient='records'))
        print(f"✅ Successfully inserted {len(df)} rows into {table_name}.")
    else:
        print(f"❌ No new data to insert into {table_name}.")
    

