# btc_price_prediction
Simple ML model, which predicts BTC price movements

# Bitcoin Price Prediction Project

## Project Goal
The objective of this project is to predict whether the price of Bitcoin (BTC-USD) will increase by a specified percentage (e.g., +5%) within a given time horizon, with a probability greater than 50%. This prediction aims to support a potentially stable profit strategy by minimizing false signals.

## Dataset
Our core dataset consists of historical BTC-USD price data sourced from **Yandex Finance**. This dataset provides OHLCV data (Open, High, Low, Close, Volume) that serves as the foundation for feature engineering.

## Features and Indicators
We engineered a comprehensive set of **proportional technical indicators** to ensure model stability across varying price levels:
- **RSI (Relative Strength Index)**
- **Stochastic Oscillator** (Stoch K, Stoch D)
- **Williams %R**
- **Bollinger Bands %B**
- **MACD** (proportional form)
- **ADX** with **DI+** and **DI-**
- **MFI** (Money Flow Index)
- **SSI** (Sentiment Strength Index, if applicable)
- **ATR** (Average True Range, relative)

Additionally, we incorporated other features such as:
- **Fear & Greed Index**
- **Dollar Index (DXY)**
- **Macro Indicators**: Consumer Price Index, Federal Funds Rate

These features provide scale-invariant information and capture underlying market dynamics.

## Methodology Overview
1. **Initial Model Comparison**:  
   Train and evaluate basic models (Logistic Regression, Random Forest, XGBoost) on raw data to select the most stable algorithm. XGBoost consistently showed better stability and performance.

2. **Feature Engineering**:  
   Add core proportional technical indicators, supplement with Dollar Index, macro indicators, and the Fear & Greed Index despite limited historical coverage.

3. **Cross-Validation and Hyperparameter Tuning**:  
   Use time-aware cross-validation (TimeSeriesSplit) on the training set to fine-tune XGBoost hyperparameters, focusing on maximizing precision while maintaining acceptable recall.

4. **Threshold Optimization**:  
   Lower the classification threshold to 0.2 after observing left-skewed probability distributions, thereby improving recall while keeping precision above 50%.

5. **Final Evaluation**:  
   Evaluate the optimized XGBoost model on the untouched test set with the selected threshold, achieving a precision of 56% and recall of 42%.

## How to Run

1. Clone the repository:
   ```bash
   !git clone https://github.com/letsslipaway/btc_price_prediction.git
   cd btc_price_prediction
   
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

3. Run the Jupyter Notebook
   ```bash
   !jupyter notebook BTC_price_prediction.ipynb
