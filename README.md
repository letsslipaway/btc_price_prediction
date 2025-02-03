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

### Option 1: Run in Google Colab

You can run the project in Google Colab by clicking the link below:
[Open in Google Colab](https://colab.research.google.com/github/letsslipaway/btc_price_prediction/blob/main/model/BTC_price_prediction.ipynb)

The notebook will open, and you can start running the cells. The first cell contains the installation of dependencies. Simply execute each cell in order.

### Option 2: Run Locally

If you prefer to run the project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/letsslipaway/btc_price_prediction.git
   cd btc_price_prediction/model
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Jupyter Notebook (if you don't have it yet):**
   ```bash
   pip install jupyter
   ```

4. **Run Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

   This will open a web page in your browser. In the interface, navigate to the `BTC_price_prediction.ipynb` file and click on it to open.
