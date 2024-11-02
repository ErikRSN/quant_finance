# Standard library imports
import datetime as dt
import warnings

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import pandas_ta
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import yfinance as yf
import os

# Configuration
warnings.filterwarnings('ignore')


#   Download/Load SP500 stocks prices data.


sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')

symbols_list = sp500['Symbol'].unique().tolist()


end_date = '2023-09-27'

start_date = pd.to_datetime(end_date)-pd.DateOffset(365*8)


# Replace your download code with this:
CSV_PATH = 'sp500_data.csv'

if os.path.exists(CSV_PATH):
    print("Loading data from CSV...")
    df = pd.read_csv(CSV_PATH)
    df['date'] = pd.to_datetime(df['date'])  # Convert date back to datetime
    df.set_index(['date', 'ticker'], inplace=True)
else:
    print("Downloading data...")
    df = yf.download(tickers=symbols_list,
                    start=start_date,
                    end=end_date).stack()
    df.index.names = ['date', 'ticker']
    df.columns = df.columns.str.lower()
    # Save to CSV
    df.to_csv(CSV_PATH)

print(df.head())



df.index.names = ['date', 'ticker']

df.columns = df.columns.str.lower()

df


### Calculate features and technical indicators for each stock.

df['garman_klass_vol'] = ((np.log(df['high'])-np.log(df['low']))**2)/2-(2*np.log(2)-1)*((np.log(df['adj close'])-np.log(df['open']))**2)

df['rsi'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))

df['bb_low'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,0])
                                                          
df['bb_mid'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,1])
                                                          
df['bb_high'] = df.groupby(level=1)['adj close'].transform(lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:,2])

def compute_atr(stock_data):
    atr = pandas_ta.atr(high=stock_data['high'],
                        low=stock_data['low'],
                        close=stock_data['close'],
                        length=14)
    return atr.sub(atr.mean()).div(atr.std())

df['atr'] = df.groupby(level=1, group_keys=False).apply(compute_atr)

def compute_macd(close):
    macd = pandas_ta.macd(close=close, length=20).iloc[:,0]
    return macd.sub(macd.mean()).div(macd.std())

df['macd'] = df.groupby(level=1, group_keys=False)['adj close'].apply(compute_macd)

df['dollar_volume'] = (df['adj close']*df['volume'])/1e6

df


print(df.head())


# Aggregate to monthly level and filter top 150 most liquid stocks for each month.
# To reduce training time and experiment with features and strategies, we convert the business-daily data to month-end frequency.


# ### Visualize the data  

# # Select a specific stock for plotting
# stock = 'AAPL'  # Replace with the stock symbol you want to plot
# stock_data = df.xs(stock, level=1)

# # Plot Garman-Klass Volatility
# plt.figure(figsize=(14, 7))
# plt.plot(stock_data.index, stock_data['garman_klass_vol'], label='Garman-Klass Volatility')
# plt.title(f'{stock} Garman-Klass Volatility')
# plt.xlabel('Date')
# plt.ylabel('Volatility')
# plt.legend()
# plt.show()