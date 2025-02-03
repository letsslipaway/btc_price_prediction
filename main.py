import os
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from dotenv import load_dotenv
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import ta
import requests
from fredapi import Fred

# loading env variables
load_dotenv()
tg_token = ${{shared.TG_TOKEN}}
fred_key = ${{shared.FRED_TOKEN}}

model = joblib.load("BTC_xgb.pkl")


def load_data(ticker='BTC-USD', period='1mo'):
    df = yf.download(ticker, period=period)
    df = df.droplevel(level=1, axis=1)
    df.dropna(inplace=True)
    return df

def add_proportional_indicators(
    df,
    rsi_window=14,
    stoch_window=14,
    stoch_smooth=3,
    wpr_lbp=14,
    bb_window=20,
    bb_std=2,
    macd_fast=12,
    macd_slow=26,
    macd_sign=9,
    mfi_window=14,
    cci_window=20,
    atr_window=14,
    ):

    # 1. RSI, Stoch, Williams %R, Bollinger %B, MACD(proportional)
    df['rsi'] = ta.momentum.RSIIndicator(close=df['Close'], window=rsi_window).rsi()

    stoch = ta.momentum.StochasticOscillator(
        high=df['High'], low=df['Low'], close=df['Close'],
        window=stoch_window, smooth_window=stoch_smooth
    )
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    df['williams_r'] = ta.momentum.WilliamsRIndicator(
        high=df['High'], low=df['Low'], close=df['Close'], lbp=wpr_lbp
    ).williams_r()

    bb = ta.volatility.BollingerBands(
        close=df['Close'], window=bb_window, window_dev=bb_std
    )
    df['bb_b'] = bb.bollinger_pband()

    macd_obj = ta.trend.MACD(
        close=df['Close'], window_slow=macd_slow, window_fast=macd_fast, window_sign=macd_sign
    )
    df['macd_diff'] = macd_obj.macd_diff()
    df['macd_prop'] = df['macd_diff'] / df['Close'] * 100
    df.drop(columns=['macd_diff'], inplace=True)

    # 2. MFI (0..100)
    mfi = ta.volume.MFIIndicator(
        high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], window=mfi_window
    )
    df['mfi'] = mfi.money_flow_index()

    # 3. CCI (±200+)
    cci = ta.trend.CCIIndicator(
        high=df['High'], low=df['Low'], close=df['Close'], window=cci_window
    )
    df['cci'] = cci.cci()

    # 4. ATR (relative)
    atr = ta.volatility.AverageTrueRange(
        high=df['High'], low=df['Low'], close=df['Close'], window=atr_window
    )
    df['atr'] = atr.average_true_range()
    df['atr_rel'] = df['atr'] / df['Close'] * 100
    df.drop(columns=['atr'], inplace=True)

    return df

def load_fear_greed_index(limit=90, date_format='world'):
    url = "https://api.alternative.me/fng/"
    params = {
        'limit': limit,
        'format': 'json',
        'date_format': date_format
    }

    response = requests.get(url, params=params)
    data = response.json()

    records = data.get('data', [])

    df_fg = pd.DataFrame(records)

    df_fg['timestamp'] = pd.to_datetime(df_fg['timestamp'], format='%d-%m-%Y')
    df_fg.set_index('timestamp', inplace=True)
    df_fg.drop(columns=['time_until_update', 'value_classification'], inplace=True)
    df_fg.rename(columns={'value': 'fear_greed'}, inplace=True)
    df_fg['fear_greed'] = df_fg['fear_greed'].astype(float)

    return df_fg

def add_fear_greed_index(df, df_fg):
    df = df.copy()
    df = df.join(df_fg, how='inner')
    return df

def add_dollar_index_feature(df):
    start_date = df.index.min().strftime('%Y-%m-%d')
    end_date = df.index.max().strftime('%Y-%m-%d')

    dxy_data = yf.download('DX-Y.NYB', start=start_date, end=end_date)
    dxy_data = dxy_data.droplevel(level=1, axis=1)
    dxy = dxy_data[['Close']]

    dxy.rename(columns={'Close': 'dollar_index'}, inplace=True)
    dxy = dxy.reindex(df.index)
    dxy.ffill(inplace=True)
    df = df.join(dxy, how='inner')

    return df

def add_macro_indicators(df_prices):
    start_date = df_prices.index.min().strftime('%Y-%m-%d')
    end_date = df_prices.index.max().strftime('%Y-%m-%d')

    df_prices = df_prices.copy()
    if not isinstance(df_prices.index, pd.DatetimeIndex):
        df_prices.index = pd.to_datetime(df_prices.index)

    macro_series = {
        'CPI':      'CPIAUCSL',  # Consumer Price Index
        'FedFunds': 'FEDFUNDS',  # Effective Federal Funds Rate
    }

    fred = Fred(api_key=fred_key)

    last_values = {}

    for col_name, fred_code in macro_series.items():
        try:
            temp = fred.get_series(fred_code, start_date, end_date).dropna()
            if not temp.empty:
                last_value = temp.iloc[-1]
                last_values[col_name] = last_value
            else:
                last_values[col_name] = None
                print(f"No data for {fred_code}")
        except Exception as e:
            print(f"Error when loading {fred_code}: {e}")
            last_values[col_name] = None

    for col, value in last_values.items():
        df_prices[col] = value

    return df_prices
    
def make_sample(df):
    df.drop(columns=['Close','Open','High','Low','Volume'], inplace=True)
    sample = df.iloc[-1]
    return sample



# start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [["/about", "/predict"]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    await update.message.reply_text("Choose what you want:", reply_markup=reply_markup)

# about command
async def about(update: Update, context: ContextTypes.DEFAULT_TYPE):
    info = """
    About:
This model predicts the future price movement of BTC (Bitcoin) over the next 10 days, with a focus on identifying whether it will increase by 5% without falling below 4%. The predictions are based on a machine learning model that analyzes historical data and current market conditions. The accuracy of the predictions is around 55-60%, depending on the volatility and other factors influencing the market.
The prediction refers to the next 10 days, as per UTC time.

Disclaimer:
This model was created for research purposes and is not intended as financial advice or as a financial tool. It should not be used for making investment decisions. The predictions are based on historical data and may not reflect future market conditions.
    """
    await update.message.reply_text(info)

# predict command
async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    df = load_data(ticker='BTC-USD', period='3mo')
    df = add_proportional_indicators(df)
    df = add_fear_greed_index(df, load_fear_greed_index())
    df = add_dollar_index_feature(df)
    df = add_macro_indicators(df)
    sample = make_sample(df)

    probs = model.predict_proba(sample.values.reshape(1, -1))
    prob_class1 = probs[0, 1]
    tau = 0.65
    if prob_class1 >= tau:
        result = f'The model predicts with a confidence of {prob_class1:.2f} that BTC may increase by 5% and not fall below 4% within the next 10 days.'
    else:
        result = 'The model predicts that BTC will not increase by 5% or fall below 4% within the next 10 days.'
    await update.message.reply_text(result)

# bot launch
def main():
    app = ApplicationBuilder().token(tg_token).build()

    # Добавляем команды
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("about", about))
    app.add_handler(CommandHandler("predict", predict))

    print("Bot is ON")
    app.run_polling()

if __name__ == "__main__":
    main()
