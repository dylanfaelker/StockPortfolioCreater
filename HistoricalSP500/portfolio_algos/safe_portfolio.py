import pandas as pd
import numpy as np

from utils import get_price_history_multi
from utils import get_price_history
from utils import populate_next_month

# Determines which ticker is the rickiest based on standard deviation and beta
def safest(tickers_data):
    smallest3_std = tickers_data.nsmallest(3, ['STD'])
    smallest_beta = smallest3_std.nlargest(1, ['Beta'])
    smallest_beta.reset_index(inplace=True)
    return smallest_beta

# produces a dataframe of the 10 stocks in the portfolio and their data
def other_9(tickers, correlator):

    correlation =  pd.DataFrame({'Ticker': [],
                                 'Price': [],
                                 'Beta': [],
                                 'STD': [],
                                 'Returns': [],
                                 'Corr': []})
    
    returns = pd.DataFrame(correlator['Returns'].iloc[0])
    returns.columns = ['Risky']
    
    for index in range(len(tickers.index)):
        if tickers['Ticker'].iloc[index] != correlator['Ticker'].iloc[0]:
            returns = returns[['Risky']]
            stock_returns = tickers['Returns'].iloc[index]
            returns = pd.concat([returns, stock_returns], join='inner', axis=1)
            correlation = pd.concat([correlation, pd.DataFrame({'Ticker': [tickers['Ticker'].iloc[index]],
                                                                'Price': [tickers['Price'].iloc[index]],
                                                                'Beta': [tickers['Beta'].iloc[index]],
                                                                'STD': [tickers['STD'].iloc[index]],
                                                                'Returns': [tickers['Returns'].iloc[index]],
                                                                'Corr': [returns.corr().iat[0,1]]})])
    
    most_correlated_20 = correlation.nsmallest(20, 'Corr')
    risky_9 = most_correlated_20.nsmallest(9, 'STD')
    final = pd.concat([correlator, risky_9])
    
    final.reset_index(inplace=True)
    final = final[['Ticker', 'Price', 'Beta', 'STD', 'Returns', 'Corr']]
    
    return final
       
# Takes in a list of 10 tickers and produces weightings for a portfolio to maximise risk level
def weightings(tickers, availibleCash, start_date, end_date, all_ticker_data):

    #Creates the initial weight distribution, which is not set in stone
    weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    
    #Creates a final dataframe to output
    FinalPortfolio = tickers
    
    #adds weights to the dataframe
    FinalPortfolio['weights'] = pd.Series(weights)

    # Fixes the Price Column
    FinalPortfolio['Price'] = FinalPortfolio['Price'].apply(lambda x: x.Close)
    
    #Calculates number of shares of bought of each stock
    FinalPortfolio['Shares'] = (FinalPortfolio.weights * availibleCash) / FinalPortfolio.Price
    
    #Creates column for the value of each stock within the portfolio
    FinalPortfolio['Value'] = FinalPortfolio.weights * availibleCash
    
    #Creates a new colummn for the weights at the end of the dataframe with the numbers in terms of %
    FinalPortfolio['Weight'] = FinalPortfolio.weights * 100
    
    #Formatting
    FinalPortfolio = FinalPortfolio[['Ticker', 'Price', 'Shares', 'Value', 'Weight']]
    FinalPortfolio.columns = ['Ticker', 'Price', 'Shares', 'Value', 'Weight']
    
    #returns a final portfolio with the purchasing data for the stock being baught
    return(FinalPortfolio)

def produce_portfolio(all_ticker_data, start_date, end_date, eom_date, stocks, funds, daily_market_returns):

    # Gets data for current SP500 constituents and ensures there is data till the EOM
    all_ticker_data_at_point = {x:all_ticker_data[x] for x in stocks if x in all_ticker_data 
                                                                        and pd.DataFrame(all_ticker_data[x]).index[-1] >= eom_date
                                                                        and pd.DataFrame(all_ticker_data[x]).index[0] < end_date
    }
    
    # sets data to be within the past few years
    for ticker in list(all_ticker_data_at_point.keys()).copy():
        all_ticker_data_at_point[ticker] = all_ticker_data_at_point[ticker].loc[start_date:end_date]


    daily_market_returns_at_point = daily_market_returns.loc[start_date:end_date]

    input_data = pd.DataFrame({'Ticker': [],
                               'Price': [],
                               'Beta': [],
                               'STD': [],
                               'Returns': []})
    for ticker in all_ticker_data_at_point:

        prices = pd.DataFrame(all_ticker_data_at_point[ticker])
        returns = prices.pct_change()

        ####### Price #############
        price = prices.iloc[-1]

        ######## Beta #############
        returns['Market'] = daily_market_returns_at_point['Close']
        beta = (returns.cov() / daily_market_returns_at_point['Close'].var()).iat[1,0]

        ######### STD #############
        std = prices.pct_change().std()

        input_data = pd.concat([input_data, pd.DataFrame({'Ticker': [ticker], 'Price': [price], 'Beta': [beta], 'STD': [std.Close], 'Returns': [prices['Close'].pct_change()]})], ignore_index=True)

    safe1 = safest(input_data)

    final10 = other_9(input_data, safe1)

    FinalPortfolio = weightings(final10, funds, start_date, end_date, all_ticker_data)

    return FinalPortfolio