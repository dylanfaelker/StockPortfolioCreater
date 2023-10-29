import pandas as pd
import numpy as np

def get_price_history_multi(tickers, start_date, end_date, all_ticker_data):
    combined_price_history = pd.DataFrame()
    for ticker in tickers:
        combined_price_history[ticker] = all_ticker_data[ticker].loc[start_date:end_date]
    combined_price_history.dropna()
    return combined_price_history

def get_price_history(ticker, start_date, end_date, all_ticker_data):
    price_history = pd.DataFrame()
    price_history['Close'] = all_ticker_data[ticker].loc[start_date:end_date]
    return price_history

def populate_next_month(all_ticker_data, portfolio, start_date, end_date):

    stocks = portfolio['Ticker']

    price_per_stock_df = pd.DataFrame()
    all_ticker_data_at_point = {x:all_ticker_data[x] for x in stocks if x in all_ticker_data}
    for ticker in all_ticker_data_at_point.keys():
        all_ticker_data_at_point[ticker] = all_ticker_data_at_point[ticker].loc[start_date:end_date]
        all_ticker_data_at_point[ticker].name = ticker
        price_per_stock_df = pd.concat([price_per_stock_df, all_ticker_data_at_point[ticker]], axis=1)
    
    shares_per_stock_df = price_per_stock_df.copy()
    i = 0
    for ticker in shares_per_stock_df.columns:
        shares_per_stock_df[ticker] = np.where(True, portfolio['Shares'].iloc[i], shares_per_stock_df[ticker])
        i += 1

    value_per_stock_df = shares_per_stock_df * price_per_stock_df

    value_per_stock_df['Total Value'] = value_per_stock_df.sum(axis=1)
    
    return value_per_stock_df[['Total Value']]