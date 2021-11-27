# StockPortfolioCreater
Creates an investment portfolio of ten S&amp;P500 stocks that is considered risky.

### 1. Strategy for filtering the stocks within the csv file:
 * Given a dataframe of tickers, this function first creates a new empty dataframe including the name of the ticker, the price of it, beta, standard deviation, the market cap and its returns
 * In the filtering stages
     - It first checks if there are any duplicates
     - It then checks to confirm the the stock is traded on US markets
     - Finally it confirms the daily volume from Jul 2 2021 to October 22 2021 is at least 10000
 * In order to get the beta calculations, we get the necessary data needed of the market (S&P500) which was obtained ouside of the function and stored in a dataframe
 * Then, with the help of threading, a for loop that goes through the entirety of the received dataframe and it ...
  - Gets the yfinance data for each stock in the dataframe 
  - Calculates the price, beta, standard deviation and returns to a single row dataframe and adds said dataframe to the main dataframe that was created at the beginning of the function
 * It then returns the final dataframe after escaping the threads

### 2. Strategy for choosing the 1 sinlge riskiest stock

* We gathered 3 stocks with the highest standard deviation on monthly returns to be the riskiest stocks. From these 3 stocks, we calculated the riskiest stock to be the one with the highest beta value.
* Stocks with high standard deviation are considered risky as they are stocks with high volatility and great fluctuations with prices. In addition, stocks with high beta values are considered risky as they are more volatile when compared to the overall market. 
* Since standard deviation and beta are both measures of riskiness and we wanted to take both into consideration. We did so by narrowing down the stocks to those that are risky in terms of standard deviation and then the one that is riskiest in terms of beta
* There are some limitations with this approach which we will touch on later that has to do with as unlucky set of data.

### 3. Discussion for our decision in choosing the stocks to include
* After determining the riskest stock among all the qualified stocks in the given csv file, we now want to decide what other stocks we should include in our portfolio in order to achieve a high-risk level. We will use the characteristics of this riskest stock as our guide to choosing the remaining stocks. The key idea of our approach is to **minimize the effect of diversification**.
* The following steps explain in detail our code and why we do this. 
>1. Firstly, we filter out 20 stocks that are most correlated to the riskest stock determined before. The purpose is to make sure the stocks in our final portfolio are highly correlated, and thus less diversified. As the riskest stock is expected to fluctuate a lot, we expect the rest of the stocks in the portfolio to change in a similar way, so that the total fluctuation is larger. 
>2. Now, we filter out the 9 riskest stocks from the list of 20 correlated stocks obtained in the previous step. To determine their risk level, we mainly look at their standard deviations, as this metric measures fluctuations. We only pick the 9 riskest stocks, because the minimum number of stocks we need to have is 10. To minimize the effect of diversification, we want minimum stocks in our portfolio. 

* Discussions:
    * After designing step 1, we recognized that solely looking at the correlation will not guarantee a high-risk level. To illustrate, consider the case where stock A and stock B have a similar change pattern, say correlation is 0.9, yet, while stock A fluctuates dramatically, the extent to which stock B prices fluctuate can be minimal. Then, adding stock B to the stock A portfolio can decrease the risk level if the flucatuations in stock B can't make up for the risk lost due to diversification. Therefore, we will also consider the risk level of each individual stock, which is step 2. 
    * Finally, we will include a total of 10 stocks in our portfolio  (i.e., 1 riskest stock we determine before + 9 stocks picked in this process), with the property that, every stock is highly correlated to one single stock (the riskest one) and in a high-risk level. 

### 4. Strategy for deciding the weights
* Firstly, we assign 35% to the riskest stock. 35% is the maximum we can give to one single stock. We choose to do so because we aim to achieve a high risk. By distributing the most weight to the riskest stock, we want our final portfolio to behave more like the riskiest and thus fluctuate more. 
* Secondly, we assign 5% to all of the rest 9 stocks. 5% is the minimum we can give to every single stock. 
* Finally, we distribute the other 20% to only the second third and fourth stocks in our list in the way that the new weight distribution gives us the highest total standard deviation. To do this, we use 2 for loops to interate between between all the distributions, calculate the standard deviation of the entire portfolio and choose the combination that maximises standard deviation. We don't give more weightings to the last 6 stocks because they are less risky and doing so will create more diversification.
* One limitation of this program is the fact that we can't iterate through all the ways to distribute the final $\$20000 between all 9 other stocks. Doing so creates exponetial blow up and trying to do it for all 9 would take 66 years. The most we can iterate though ina resonable time is 3 which takes about 0.6 seconds. Once you start doing 4, it takes a couple minutes and anything more than that will take too long. For this reason, we are only iterating between distribution for 3 stocks. Those 3 stocks are the 3 with the highest standard deviation since they are the most likely to increase riskiness level by being given more weight.

### 5. Next Steps:
    
Our project evaluated the risk of stocks based on high standard deviation, high beta value and low diversification. However there are other factors that could have helped to determine the risk level of stocks. These include calculating R-squared (coefficient of determination), Value at Risk (VaR) and/or market capitalization. 

The coefficient of determination displays the percentage of a fund or security's movements based on movements in a benchmark index (for example the Standard & Poor's 500 index). This value helps determine how likely a stock would drop if it's benchmark index dropped. A stock with a high R squared value measured against the riskiest stock from the portfolio can be considered risky. Value at Risk provides a worst-case scenario analysis where it calculates the percent of loss based on a time period and confidence level. It measures the risk of loss for investments. A stock with a high VaR value would be risky as the probability of losing that investment is high. Market cap is the total value of a company’s stocks. It is calculated by multiplying the number of outstanding shares with the current price of each share. A company with a small market cap is deemed more risky than a company with a large market cap. This is because companies with small market caps tend to be young companies with more uncertainties and high volatility. Incorporating these additional risk factors into our project could provide a better portfolio of risky stocks as there would be more factors evaluated with each stock. 

When it comes to limitations to our project, the steps that we take to come up with the riskiest stocks may not produce the best results. For example, when we consider 20 stocks that are highly correlated with the riskiest stock, we would not consider stocks that are risky (according to the beta value and standard deviation) but not correlated with the riskiest stock. One way we can fix the issue is by dynamically setting a minimum value for the correlation coefficient and then gather the stocks that fit the requirement. In the case that there are less than 9 stocks, we would decrease the minimum value of the correlation coefficient until we reach the required number of stocks.

Another limitation comes from the narrowing down of stocks using different values. There can be unlucky set of data where you do not even come close to ending up with a risky set of stocks. Take for example a set where stock A has a STD (standard devaition) of 20 and a beta of 1, stock B has a STD of 8 and a beta of 1.01, and stock C has a STD of 7 and a beta of 1.1. If these were the 3 stocks with the highest STD when trying to find the riskiest, clearly the top riskiest is stock A. But since we choose our riskiest stock off of beta from the top 3 STDs, our algorithm would produce stock C as the riskiest. If we were to fix this issue, we could create a points system that assigns certain amounts of points for each measure of risk. In the above scenario, stock A would get alot of points for it's STD and the other stocks wouldn't be able to get enough from beta to catch up. Perfecting the points values could take awhile but it would eliminate the possibility of the above situation.
