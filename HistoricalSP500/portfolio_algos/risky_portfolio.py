import pandas as pd
import numpy as np

from utils import get_price_history_multi
from utils import get_price_history
from utils import populate_next_month

# Determines which ticker is the rickiest based on standard deviation and beta
def riskiest(tickers_data):
    largest3_std = tickers_data.nlargest(3, ['STD'])
    largest_beta = largest3_std.nlargest(1, ['Beta'])
    largest_beta.reset_index(inplace=True)
    return largest_beta

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
    
    most_correlated_20 = correlation.nlargest(20, 'Corr')
    risky_9 = most_correlated_20.nlargest(9, 'STD')
    final = pd.concat([correlator, risky_9])
    
    final.reset_index(inplace=True)
    final = final[['Ticker', 'Price', 'Beta', 'STD', 'Returns', 'Corr']]
    
    return final
       
# Takes in a list of 10 tickers and produces weightings for a portfolio to maximise risk level
def weightings(tickers, availibleCash, start_date, end_date, all_ticker_data):

    #Creates list of tickers in the order of risk level 
    tickers10 = tickers['Ticker'].iloc[0:10].tolist()

    #Creates the initial weight distribution, which is not set in stone
    weights = [0.35, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]

    #Creates a list to store tickers except the second to forth most risky stocks
    restTickers = tickers10[4:10]
    restTickers.insert(0, tickers10[0])

    #Creates a list for the weights for each of the stocks in RestTickers (these will not changed) 
    restweights = weights[4:10]
    restweights.insert(0, weights[0])

    #Gets data from yf for those tickers, and stores closing prices in a dataframe
    restprices = get_price_history_multi(restTickers, start_date, end_date, all_ticker_data)

    #Calculates the value of each of the 7 stocks based on each of their weights
    for ticker in restTickers:
        restprices[ticker+'_SharesPurchased'] = 0
        restprices[ticker+'_SharesPurchased'].iloc[0] = availibleCash * restweights[restTickers.index(ticker)] / restprices[ticker].iloc[-1]

    #Creates a column in the prices dataframe for the unchanging tickers with for the portfolio value overtime
    restprices['Portfolio_Value'] = 0
    restprices['Portfolio_Value'].iloc[0] = availibleCash * 0.65

    #Calculates the total portfolio value over time
    for x in range(1,len(restprices.index)):
        for ticker in restTickers:
            restprices['Portfolio_Value'].iloc[x] += restprices[ticker+'_SharesPurchased'].iloc[0] * restprices[ticker].iloc[x]

    #Creates a list of the tickers of the second, third and forth most risky stocks
    Two_Three_Four_Tickers = [tickers10[1], tickers10[2], tickers10[3]]

    #Gets closing prices from yahoo finance for each of these stocks
    prices = get_price_history_multi(Two_Three_Four_Tickers, start_date, end_date, all_ticker_data)

    #Sets the starting value of each stock to be $100000
    #We will later find a portion of the values when determining the riskest weightings for these stocks
    for ticker in Two_Three_Four_Tickers:
        prices[ticker+'_SharesPurchased']= 0
        prices[ticker+'_Value']=0
        prices[ticker+'_Value'].iloc[0]=availibleCash
        prices[ticker+'_SharesPurchased'].iloc[0] =  availibleCash / prices[ticker].iloc[0]
        
    #Calculates value over time for each stock
    for x in range(1,len(prices.index)):
        for ticker in Two_Three_Four_Tickers:
            prices[ticker+'_Value'].iloc[x] = prices[ticker+'_SharesPurchased'].iloc[0] * prices[ticker].iloc[x]

    #Combines the two dataframes into one that will contain all value and pricing data overtime
    combined = pd.concat([restprices,prices],join='inner', axis=1)

    #Creates a new dataframe to store standard deviations and their weightings
    totalstd = pd.DataFrame(index=range(0, 21))
    totalstd['Standard_Deviation'] = 0
    totalstd['Weight_3'] = 0

    #Creates a dataframe to store the value of each of the stocks that have a dynamic weightings
    value_2 = combined[tickers10[1]+"_Value"]
    value_3 = combined[tickers10[2]+"_Value"]
    value_4 = combined[tickers10[3]+"_Value"]

    #Loops through all possible ways to ditribute $20000 in portions of $1000
    for x in totalstd.index:
        
        #creates a dataframe to store the weightings of the first stock and the standard deviation that goes with it
        weight34 = pd.DataFrame(index=range(0,21-x))
        weight34['Standard_Deviation'] = 0

        #loops through all ways to distribute the remaining cash not used by the first stock to the second and third stock
        for y in range(0,21-x):
            #Calculates the value and standard deviation of the portfolio for the current weightings
            total_values = restprices.Portfolio_Value + (value_2 * (5+x)/100) + (value_3 * (5+y)/100) + (value_4 * (25-x-y)/100)
            portfolio = pd.DataFrame(total_values)
            returns = pd.DataFrame(portfolio.pct_change())

            #Adds the standard deviation to a dataframe
            weight34['Standard_Deviation'].iloc[y] = returns.std()[0]

        #Calculates the biggest standard devaition of the dataframe
        #  which changes how cash was distributed betweeen the second and third stock
        std1 = weight34['Standard_Deviation'].max()

        # print(tickers10[1])
        # print(tickers10[2])
        # print(tickers10[3])
        # print(value_2)
        # print(value_3)
        # print(value_4)
        # print(restprices.Portfolio_Value)
        # print(str(x) + ', ' + str(y))
        # print(total_values)
        # print(portfolio)
        # print(returns)
        # print(weight34)
        
        #Adds the max standard deviation and it's weights to another dataframe which 
        #  holds standard deviations for different amounts cash in the first stock
        weight3 = weight34.index[weight34['Standard_Deviation']==std1].tolist()[0]
        totalstd['Weight_3'].iloc[x] = weight3
        totalstd['Standard_Deviation'].iloc[x] = std1

    ## Determines the weighting for the stock with the biggest standard deviation
    weight2 = totalstd[totalstd.Standard_Deviation == totalstd['Standard_Deviation'].max()].index[0]
    weight3 = totalstd.loc[weight2, 'Weight_3']
    
    #Updates the weights in the original weights dataframe
    weights[1]=(5 + weight2)/100
    weights[2]=(5 + weight3)/100
    weights[3]=(25 - weight2 - weight3)/100  
    
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

    risky1 = riskiest(input_data)

    final10 = other_9(input_data, risky1)

    FinalPortfolio = weightings(final10, funds, start_date, end_date, all_ticker_data)

    return FinalPortfolio