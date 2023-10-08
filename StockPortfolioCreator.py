import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from datetime import timedelta
import concurrent.futures as cf

#Takes in a dataframe of tickers and filters out ones that are duplicates, not traded in the US 
#  or have an average volume less than 10000 for Jul 2 to Oct 22 2021. It produces a dataframe
#  with all valid tickers and finance data to go with them.
def filtering(Tickers):
    
    #Creates a new dataframe to store valid tickers and their financial data
    Valid_Tickers =  pd.DataFrame({'Tickers': [],
                                   'Price': [],
                                  'Beta': [],
                                  'STD': [],
                                  'Returns': []})
    
    #makes sure there are no duplicates
    for index in range(len(Tickers.index)):
        if Tickers.iloc[index,0] in Tickers.iloc[index+1:]:
            Tickers.drop([index])
            
    number = 0
    
    #Threading
    with cf.ThreadPoolExecutor() as executor:
        
        #creates a thread for each Ticker to gets it's history data
        datarow = [executor.submit(filtering_thread, Tickers.iloc[index,0]) for index in range(len(Tickers.index))]
        
        #Adds each ticker's data to the dataframe
        for row in cf.as_completed(datarow):
            Valid_Tickers = Valid_Tickers.append(row.result())
            number+=1
            print(number)
    
    #Formats the data
    Valid_Tickers.reset_index(inplace=True)
    Valid_Tickers = Valid_Tickers[['Tickers', 'Price', 'Beta', 'STD', 'Returns']]
    
    #returns the dataframe with all the data
    return (Valid_Tickers)

#Takes in a Ticker, filters it to ensure it is traded in the US and as enough volume.
#  It then grabs the finacial data and returns a dataframe with a single row 
#  containing the ticker and the financial data
def filtering_thread(Ticker):
    
    #Gets data for filtering
    stock = yf.Ticker(Ticker)
    stock_hist = stock.history(start=data_start, end=data_end, interval='1d')
    
    #gets the data for the last 3 months to check volume
    volume_hist = stock_hist.iloc[-66:-1]
    
    #grabs stock info
    info = stock.info

    #Checks if stock is traded in the US
    if 'market' in info and info['market'] == 'us_market' and not(volume_hist.empty):

        #Checks if the daily volume is at least 10000
        total_sum = stock_hist.Volume.sum(axis=0)
        average = total_sum/(len (stock_hist))
        if average >= 10000:

            #Gets monthly histru for that time
            monthly_hist=stock_hist.resample('MS').first()
            prices = pd.DataFrame(monthly_hist['Close'])
            monthly_returns = prices.pct_change()

            #creates a dataframe for just the daily closing price 
            #  and another one for just daily returns
            daily_price = pd.DataFrame(stock_hist['Close'])
            daily_returns = daily_price.pct_change()

            ####### Price #############

            #Closing price for the last day availible (Nov 26 when run for competition)
            price = stock_hist['Close'].iloc[-1]

            ######## Beta #############

            #Adds markets daily returns to the dataframe
            daily_returns['Market'] = daily_market_returns['Close']

            #Calculates beta
            beta = daily_returns.cov() / daily_market_returns['Close'].var()

            ######### STD #############

            #calculated standard deviation
            std = prices.pct_change().std()

            #returns a dataframe with the tickers price, beta, std and a dataframe for it's returns
            return pd.DataFrame({'Tickers': [Ticker],
                               'Price': [price],
                              'Beta': [beta.iat[1,0]],
                              'STD': [std.Close],
                              'Returns': [stock_hist['Close'].pct_change()]})

#Takes in a set of tickers and determines which ticker is the 
#  rickiest based on standard deviation and beta
def riskiest (Tickers):
    
    # gets the 3 tickers with the highest standard deviation
    largest3_std = Tickers.nlargest(3, ['STD'])
    
    # gets the highest beta value from the 3 tickers with the highest standard deviation
    largest_beta = largest3_std.nlargest(1, ['Beta'])
    
    #returns the riskiest stock
    return (largest_beta)

#Tickers is the dataframe with all the stock and their data (beta, std, mcap, returns)
#Corr is the stock that is being used to get correlation (riskiest). It should be a dataframe
#  similar to Tickers but with only one row.
#  It produces a dataframe of 10 stocks and their data which will be the stock in Corr 
#  and 9 other most correlated and risky stocks.
def other_9(Tickers, Corr):
    
    #creates a dataframe to store values
    Correlation =  pd.DataFrame({'Tickers': [],
                                 'Price': [],
                                 'Beta': [],
                                 'STD': [],
                                 'Returns': [],
                                 'Corr': []})
    
    #gets returns for the riskiest stock
    returns = Corr.iloc[0,4]
    
    #loops through the tickers
    for index in range(len(Tickers.index)):
        
        #makes sure it doesn't get correaltion with itself
        if not(Tickers.iloc[index, 0] == Corr.iloc[0,0]):
            
            #gets the returns for new stock being checked for correlation
            stock_returns = Tickers.iloc[index, 4]
            
            #combines the monthly returns for the risky and other stock in one dataframe
            returns = pd.concat([returns, stock_returns], join='inner', axis=1)
            returns.columns = ['Risky', 'Other']
            
            #adds correlation data to the main dataframe with all stocks and data
            Correlation = Correlation.append(pd.DataFrame({'Tickers': [Tickers.iloc[index,0]],
                                                            'Price': [Tickers.iloc[index,1]],
                                                            'Beta': [Tickers.iloc[index,2]],
                                                            'STD': [Tickers.iloc[index,3]],
                                                            'Returns': [Tickers.iloc[index,4]],
                                                            'Corr': [returns.corr().iat[0,1]]}))
            
            #removes the returns for the other stock
            returns = returns[['Risky']]
            
    #Gets the top 20 most correlated stocks
    most_correlated_20 = Correlation.nlargest(20, 'Corr')
    
    #Gets the top 9 riskiest stocks from the most correlated
    risky_9 = most_correlated_20.nlargest(9, 'STD')
    
    #Combines the dataframe into one final stock with the risky stock
    final = Corr.append(risky_9)
    
    #formatting
    final.reset_index(inplace=True)
    final = final[['Tickers', 'Price', 'Beta', 'STD', 'Returns', 'Corr']]
    
    #returns top 10 stocks
    return(final)
            
 #Takes in a list of 10 tickers and produces weightings for a portfolio to maximise risk level
def weightings(Tickers, availibleCash): 
    #Creates list of tickers in the order of risk level 
    Tickers10 = Tickers['Tickers'].iloc[0:10].tolist()

    #Creates the initial weight distribution, which is not set in stone
    weights = [0.35, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]

    #Creates a list to store Tickers except the second to forth most risky stocks
    RestTickers = Tickers10[4:10]
    RestTickers.insert(0, Tickers10[0])

    #Gets data from yf for those tickers, and stores closing pirces in a dataframe
    data = yf.download(RestTickers,start= "2010-01-01", end= datetime.today(),period='1d')
    restprices = pd.DataFrame(data['Close'])
    restprices = restprices.dropna()
    restprices.columns = RestTickers

    #Creates a list for the weights for each of the stocks in RestTickers (these will not changed) 
    restweights = weights[4:10]
    restweights.insert(0, weights[0])

    #Creates a column in the prices dataframe for the unchanging tickers with for the portfolio value overtime
    restprices['Portfolio_Value'] = 0
    restprices['Portfolio_Value'].iloc[0]=100000*0.65 #sets first row to be the total value of these 7 stocks which is $65000

    #Calculates the value of each of the 7 stocks based on each of their weights
    for ticker in RestTickers:
        restprices[ticker+'_SharesPurchased']= 0
        restprices[ticker+'_SharesPurchased'].iloc[0] = 100000 * restweights[RestTickers.index(ticker)] / restprices[ticker][0]

    #Calculates the total portfolio value over time
    for x in range(1,len(restprices.index)):
        for ticker in RestTickers:
            restprices['Portfolio_Value'].iloc[x] += restprices[ticker+'_SharesPurchased'].iloc[0] * restprices[ticker].iloc[x]

    #Creates a list of the Tickers of the second, third and forth most risky stocks
    Two_Three_Four_Tickers = [Tickers10[1], Tickers10[2], Tickers10[3]]

    #Gets closing prices from yahoo finance for each of these stocks
    data1 = yf.download(Two_Three_Four_Tickers,start= "2010-01-01", end= datetime.today(),period='1d')
    prices = pd.DataFrame(data1['Close'])
    prices = prices.dropna()
    prices.columns = Two_Three_Four_Tickers

    #Sets the starting value of each stock to be $100000
    #We will later find a portion of the values when determining the riskest weightings for these stocks
    for ticker in Two_Three_Four_Tickers:
        prices[ticker+'_SharesPurchased']= 0
        prices[ticker+'_Value']=0
        prices[ticker+'_Value'].iloc[0]=100000
        prices[ticker+'_SharesPurchased'].iloc[0] =  100000 / prices[ticker][0]
        
    #Calculates value over time for each stock
    for x in range(1,len(prices.index)):
        for ticker in Two_Three_Four_Tickers:
            prices[ticker+'_Value'].iloc[x] = prices[ticker+'_SharesPurchased'].iloc[0] * prices[ticker][x]

    #Combines the two dataframes into one that will contain all value and pricing data overtime
    combined = pd.concat([restprices,prices],join='inner', axis=1)

    #Creates a new dataframe to store standard deviations and their weightings
    totalstd = pd.DataFrame(index=range(0, 21))
    totalstd['Standard_Deviation'] = ''
    totalstd['Weight_2'] = ''

    #Creates a dataframe to store the value of each of the stocks that have a dynamic weightings
    value_2= combined[Tickers10[1]+"_Value"]
    value_3= combined[Tickers10[2]+"_Value"]
    value_4= combined[Tickers10[3]+"_Value"]

    #Loops through all possible ways to ditribute $20000 in portions of $1000
    for x in totalstd.index:
        
        #creates a dataframe to store the weightings of the first stock and the standard deviation that goes with it
        weight23 = pd.DataFrame(index=range(0,21-x))
        weight23['Standard_Deviation'] = ''
        weight23['weight1'] = x

        #loops through all ways to distribute the remaining cash not used by the first stock to the second and third stock
        for y in range(0,21-x):
            #Calculates the value and standard deviation of the portfolio for the current weightings
            total_values= combined.Portfolio_Value + (value_2 *(5+x)/100) + (value_3 * (5+y)/100) + (value_4 * (25-x-y)/100)
            portfolio = pd.DataFrame(total_values)
            returns = pd.DataFrame(portfolio.pct_change())
            
            #Adds the standard deviation to a dataframe
            weight23['Standard_Deviation'].iloc[y] = returns.std()[0]

        #Calculates the biggest standard devaition of the dataframe
        #  which changes how cash was distributed betweeen the second and third stock
        std1 = weight23['Standard_Deviation'].max()
        
        #Adds the max standard deviation and it's weights to another dataframe which 
        #  holds standard deviations for different amounts cash in the first stock
        weight2 = weight23.index[weight23['Standard_Deviation']==std1].tolist().pop(0)
        totalstd['Weight_2'].iloc[x] = weight2
        totalstd['Standard_Deviation'] = std1

    ## Determines the weighting for the stock with the biggest standard deviation
    diff = totalstd[totalstd.Standard_Deviation == totalstd['Standard_Deviation'].max()].index[0]
    
    #Updates the weights in the original weights dataframe
    weights[1]=(5 + diff)/100
    weights[2]=(5 + totalstd.iloc[diff].Weight_2)/100
    weights[3]=(25 - diff - totalstd.iloc[diff].Weight_2)/100  
    
    #Creates a final dataframe to output
    FinalPortfolio = Tickers
    
    #adds weights to the dataframe
    weights = pd.Series(weights)
    FinalPortfolio['weights'] = weights
    
    #Calculates number of shares of bought of each stock
    FinalPortfolio['Shares'] = (FinalPortfolio.weights * availibleCash) / FinalPortfolio.Price
    
    #Creates column for the value of each stock within the portfolio
    FinalPortfolio['Value'] = FinalPortfolio.Price * FinalPortfolio.Shares
    
    #Creates a new colummn for the weights at the end of the dataframe with the numbers in terms of %
    FinalPortfolio['Weight'] = FinalPortfolio.weights * 100
    
    #Formatting
    FinalPortfolio = FinalPortfolio[['Tickers', 'Price', 'Shares', 'Value', 'Weight']]
    FinalPortfolio.columns = ['Ticker', 'Price', 'Shares', 'Value', 'Weight']
    
    #returns a final portfolio with the purchasing data for the stock being baught
    return(FinalPortfolio)

# Returns a dataframe with the inputted tickers and their current prices
def grab_tickers(Tickers):
    
    #Creates a new dataframe to store tickers and there price
    Valid_Tickers =  pd.DataFrame({'Ticker': [],
                                   'Price': []})
            
    number = 0;
    
    #Threading
    with cf.ThreadPoolExecutor() as executor:
        
        #creates a thread for each Ticker to gets it's history data
        datarow = [executor.submit(price_thread, Tickers.iloc[index,0]) for index in range(len(Tickers.index))]
        
        #Adds each ticker's data to the dataframe
        for row in cf.as_completed(datarow):
            Valid_Tickers = Valid_Tickers.append(row.result())
            number+=1
            print(number)
    
    #Formats the data
    Valid_Tickers.reset_index(inplace=True)
    Valid_Tickers = Valid_Tickers[['Ticker', 'Price']]
    
    
    
    #returns the dataframe with all the data
    return (Valid_Tickers)

# Takes in a Ticker and finds its most recent closing price
def price_thread(Ticker):
    
    #Gets data for filtering
    
    today = datetime.today()
    yesterday = today - timedelta(days=7)

    stock = yf.Ticker(Ticker)
    stock_hist = stock.history(start=yesterday, end=today, interval='1d')
    print(stock_hist)
    close_today = stock_hist.iloc[-1, 3]
    
    return pd.DataFrame({'Ticker': [Ticker],
                         'Price': [close_today]})

# Determines the current value of the portfolio
def calc_funds():
    
    # Grabs the past data from the csv
    past_data = pd.read_csv('C:\\Users\\faelk\\PersonalProjects\\PortfolioGenerator\\autorunning_script - Copy\\Portfolio_History.csv')
    
    # Creates a dataframe of past tickers most recent share count sorted alphabetically by ticker
    prev_shares = past_data.iloc[1: , :]
    prev_shares = prev_shares.set_index('Ticker')
    prev_shares.sort_index(inplace=True) 
    prev_shares.reset_index(inplace=True)
    prev_shares = prev_shares.iloc[: , -1:]
    
    # Creates a dataframe of past tickers sorted alphabetically
    prev_tickers = past_data.iloc[1: , 1:2]
    prev_tickers = prev_tickers.set_index('Ticker')
    prev_tickers.sort_index(inplace=True) 
    prev_tickers.reset_index(inplace=True)
    prev_tickers = prev_tickers[['Ticker']]
    
    # Optimises the reading of ticker price
    prev_tickers['Shares'] = prev_shares
    prev_tickers = prev_tickers.dropna()
    prev_tickers = prev_tickers.reset_index()
    prev_tickers = prev_tickers[['Ticker']]
    
    prev_shares = prev_shares.dropna()
    prev_shares = prev_shares.reset_index()
    prev_shares = prev_shares.iloc[: , -1:]
    
    # Grabs the closing price for today and sorts alphabetically by ticker
    prices = grab_tickers(prev_tickers)
    prices = prices.set_index('Ticker')
    prices.sort_index(inplace=True) 
    prices = prices.reset_index()
    
    # Merges data and calculates total value of portfolio
    prev = prices
    prev['Shares'] = prev_shares
    prev['Value'] = prev.Price * prev.Shares
    availible_funds = prev.Value.sum(axis=0)
    
    if availible_funds == 0:
        return(100000)
    else:
        return(availible_funds)

# Updates the Portfolio_History.csv and saves yesterdays data in Yesterday_Portfolio_History.csv
def rewriting_csv(FinalPortfolio):
    
    today = datetime.today()
    yesterday = today - timedelta(days=7)
    sp500_stock = yf.Ticker('^GSPC')
    sp500_hist = sp500_stock.history(start=yesterday, end=today, interval='1d')
    sp500_today = sp500_hist.iloc[-1, 3]
    multiplier = 26.561906231801917
    sp500_today*=multiplier
    
    # Adding total value to the top of the table
    # Shares will be the total value in dollars of the total row
    today_value = FinalPortfolio.Value.sum(axis=0)
    value_row = []
    value_row.insert(0, {'Ticker': 'SP500', 'Price': today_value, 'Shares': today_value, 'Value': today_value, 'Weight': 100.0})
    value_row.insert(0, {'Ticker': 'Total Value', 'Price': sp500_today, 'Shares': sp500_today, 'Value': sp500_today, 'Weight': 100.0})
    today_data = pd.concat([pd.DataFrame(value_row), FinalPortfolio], ignore_index=True)

    # reformatting columns
    today_data = today_data[['Ticker', 'Shares']]
    today = datetime.today()
    today_data.columns = ['Ticker', today.strftime("%x")]

    # Grabbing past data
    past_data = pd.read_csv('C:\\Users\\faelk\\PersonalProjects\\PortfolioGenerator\\autorunning_script - Copy\\Portfolio_History.csv')
    
    # Saving yesterdays data incase of error
    saving = past_data.iloc[:,1:]
    saving.to_csv('C:\\Users\\faelk\\PersonalProjects\\PortfolioGenerator\\autorunning_script - Copy\\Yesterday_Portfolio_History.csv')
    
    # removing extra index column
    past_data = past_data.iloc[: , 1:]
    full_data = past_data.merge(today_data, left_on='Ticker', right_on='Ticker', how='outer')

    # Ordering the shares alphabetically
    reorder_tickets = full_data.iloc[1:]
    reorder_tickets = reorder_tickets.set_index('Ticker')
    reorder_tickets.sort_index(inplace=True) 
    reorder_tickets = reorder_tickets.reset_index()

    # Adding ordered share data back to dataframe
    full_data.iloc[1:] = reorder_tickets

    # Updating csv file
    full_data.to_csv('C:\\Users\\faelk\\PersonalProjects\\PortfolioGenerator\\autorunning_script - Copy\\Portfolio_History.csv')
    print(full_data)

# Returns a dataframe of all tickers in the SP500
def get_sp500():
    try:
        # Grabs the SP500 tickers from wikipedia
        wikipedia = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        sp500 = wikipedia[0]
        sp500_symbols = pd.DataFrame(sp500['Symbol'].values.tolist())
        sp500_symbols.columns = ['Tickers']

        # Saves current SP500 tickers incase of future issues
        sp500_symbols.to_csv('S&P500.csv')
    except:
        # Grabs yesterdays data instead
        print('ERROR in reading current SP500 companies')
        print('Using last successful read instead')
        sp500_symbols = pd.read_csv('S&P500.csv')
        sp500_symbols = sp500_symbols.iloc[:,1:]
        sp500_symbols.columns = [['Tickers']]
        
    return(sp500_symbols)

##################### The "main" area ##############################################################################################

#Reads in the csv file 
Tickers = get_sp500()

print("Tickers grabbed")

#Sets constants for the data being collected
data_end = datetime.today()
data_start = data_end - timedelta(days=3653) #10 years

print("Date range set")

#Grabs market data for S&P500
market_index = yf.Ticker('^GSPC')
market_hist = market_index.history(start=data_start, end=data_end, interval='1d')
market_hist = pd.DataFrame(market_hist['Close'])
daily_market_returns = market_hist.pct_change()

print("Market data grabbed")

#Filters the stocks and gets their data
Tickers = filtering(Tickers)

print("Tickets filtered")

#Determines the riskiest stock
riskiest = riskiest(Tickers)

print("Riskiest chosen")

#Chooses the other 9 stocks
final10 = other_9(Tickers, riskiest)

print("Riskiest 10 chosen")

#Gets the weighting for the 10 stocks
FinalPortfolio = weightings(final10, calc_funds())

print("Portfolio weightings decided")
print(FinalPortfolio)

#Calculates the total weights and value to prove the porfolio is valid
totalValue = FinalPortfolio.Value.sum(axis=0)
totalWeight = FinalPortfolio.Weight.sum(axis=0)
print('Total value is: $' + str(totalValue))
print('Total weight is: ' + str(totalWeight) + '%')

#Creates a dataframe for the .csv file and creates the csv
rewriting_csv(FinalPortfolio)

print("File created")
print(FinalPortfolio)

####################################################################################################################################