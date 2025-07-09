import pandas as pd
import requests
from datetime import datetime, timedelta

TIINGO_API_KEY = '5c30f6c2e27d1f902ace1d777b29691a610df388'
BASE_URL = 'https://api.tiingo.com/tiingo/daily/{}/prices'

# Example ADR tickers and local tickers (to be expanded)
ADR_PAIRS = [
    # (ADR, Local)
    ('BABA', '9988.HK'),   # Alibaba US/HK
    ('JD', '9618.HK'),     # JD.com US/HK
    ('ITUB', 'ITUB4.SA'),  # Itau Unibanco US/Brazil
    ('PBR', 'PETR4.SA'),   # Petrobras US/Brazil
    ('TME', '1698.HK'),    # Tencent Music US/HK (example)
    # Add more pairs for Mexico, Brazil, etc.
]

START_DATE = '2022-01-01'
END_DATE = datetime.today().strftime('%Y-%m-%d')

headers = {'Content-Type': 'application/json'}

def fetch_tiingo(ticker, start_date, end_date):
    url = BASE_URL.format(ticker.lower()) + f'?startDate={start_date}&endDate={end_date}&token={TIINGO_API_KEY}'
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        df = pd.DataFrame(r.json())
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        return df
    else:
        print(f"Error fetching {ticker}: {r.status_code}")
        return pd.DataFrame()

def get_adr_local_data(pairs, start_date, end_date):
    data = {}
    for adr, local in pairs:
        print(f"Fetching {adr} and {local}")
        adr_df = fetch_tiingo(adr, start_date, end_date)
        local_df = fetch_tiingo(local, start_date, end_date)
        if not adr_df.empty and not local_df.empty:
            merged = pd.merge(adr_df, local_df, left_index=True, right_index=True, suffixes=(f'_{adr}', f'_{local}'))
            data[(adr, local)] = merged
    return data

if __name__ == "__main__":
    data = get_adr_local_data(ADR_PAIRS, START_DATE, END_DATE)
    # Save to disk for further processing
    for (adr, local), df in data.items():
        df.to_csv(f"adr_pair_{adr}_{local}.csv")
    print("Data fetching complete.")