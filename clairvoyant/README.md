# Clairvoyant v4 - XRP Price Forecaster

![Clairvoyant Banner](assets/clairvoyant-banner.jpg)

Clairvoyant is a sophisticated, multi-modal crypto price forecasting platform specifically designed for XRP. It leverages a comprehensive suite of data sources, advanced analytical techniques, and an ensemble of machine learning and deep learning models to predict future price movements with enhanced accuracy and quantified confidence.

## Table of Contents

- [Project Name and Description](#project-name-and-description)
- [Technology Stack](#technology-stack)
- [Project Architecture](#project-architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Running the Application](#running-the-application)
- [Key Features](#key-features)
- [Development Workflow](#development-workflow)
- [Performance Results](#performance-results)
- [License](#license)

## Project Clairvoyant

Clairvoyant is an advanced cryptocurrency price prediction system focusing on XRP. It integrates real-time market data from major exchanges (Binance, OKX), on-chain analytics from the XRP Ledger, news sentiment analysis, and fundamental indicators like Kimchi Premium and Order Blocks. The platform employs an ensemble of Deep Learning (GRU, LSTM, CNN-LSTM) and Machine Learning (LGBM, XGBoost) models, trained and validated using a robust rolling walk-forward cross-validation strategy. It provides future price forecasts complete with confidence intervals, offering traders a powerful tool for informed decision-making.

## Technology Stack

The Clairvoyant project is built using a modern Python-based technology stack, leveraging powerful libraries for data science, machine learning, and financial data processing.

-   **Languages:** Python 3.x
-   **Core Libraries:**
    -   `ccxt`: Cryptocurrency exchange integration (Binance, OKX, Bithumb).
    -   `pandas`, `numpy`: Efficient data manipulation and numerical operations.
    -   `tensorflow`, `keras`: Deep Learning framework for GRU, LSTM, and CNN-LSTM models.
    -   `lightgbm`, `xgboost`, `scikit-learn`: Machine Learning algorithms (LGBM, XGBoost, Ridge Regression for meta-modeling).
    -   `optuna`: Advanced hyperparameter optimization for model tuning.
    -   `nltk`, `afinn`: Natural Language Toolkit and sentiment analysis library for news processing.
    -   `aiohttp`, `feedparser`, `requests`: Asynchronous and synchronous HTTP requests for data fetching.
    -   `matplotlib`, `seaborn`: Data visualization and plotting.
    -   `ta`: Technical Analysis indicators.
    -   `websockets`: For potential future real-time data streaming integration.
    -   `python-dotenv`: Management of environment variables for secure configuration.
    -   `joblib`: Efficient serialization and deserialization of Python objects (for model persistence).
-   **Development Tools:** Git, Visual Studio Code (with Python extensions).

## Project Architecture

The Clairvoyant system operates on a modular, pipeline-driven architecture designed for robust data processing, model training, and prediction.

```mermaid
graph TD
    subgraph Data Sources
        A[Exchange APIs (Binance, OKX, Bithumb)] -- Market Data (OHLCV, Funding, OI) --> B(Data Ingestion)
        C[RSS Feeds (Crypto News)] -- Article Text --> D(Sentiment Analysis)
        E[XRPScan.com / XRPL] -- On-Chain Data (Whale Activity) --> F(On-Chain Analysis)
        G[.env Config] -- API Keys & Parameters --> H(Configuration Manager)
    end

    subgraph Clairvoyant Processing Pipeline
        B -- Preprocessed Market Data --> I(Feature Engineering)
        D -- Sentiment Scores --> I
        F -- On-Chain Metrics --> I
        I -- Enriched & Aligned Features --> J(Data Preparation)
        J -- Sequenced Data (Time-Series) --> K(Rolling Walk-Forward Validation)
        J -- Tabular Data --> K
        K -- Model Evaluation & Hyperparameter Tuning (Optuna) --> L(Ensemble Model Training)
        L -- Trained Base Models (GRU, LSTM, CNN-LSTM, LGBM, XGB) --> M(Meta-Model Stacking)
        M -- Final Ensemble Prediction --> N(Prediction Generation)
    end

    subgraph Outputs & Logging
        N -- Future Price Forecasts & CI --> O(predictions/ folder)
        L -- Saved Models & Scalers --> P(models/ folder)
        D -- Sentiment Plots --> Q(sentiment/ folder)
        K -- Training Metrics & TensorBoard Logs --> R(logs/ folder)
    end

    H --> B
    H --> F
    H --> K
```

## Project Structure

The project follows a clear and organized directory structure to separate different components and outputs.

```
.
.
├── .env                    # Environment variables (API keys, configuration parameters)
├── .gitignore              # Git ignore rules
├── clairvoyant.py          # Main entry point with rolling walk-forward validation and ensemble training
├── requirements.txt        # Python package dependencies
├── terminal_styles.py      # ANSI color and formatting utilities for CLI output
├── xrpscan.py              # XRP Ledger whale activity detector and on-chain analyzer
├── assets/                 # Static resources (banners, images)
├── lexicon/                # Sentiment analysis resources
│   └── crypto_lexicon.py   # Crypto-specific word sentiment dictionary
├── logs/                   # Training logs and TensorBoard event files
│   ├── optuna_global/      # Persistent Optuna hyperparameter optimization database
│   │   └── eos.db          # Eternal study database for self-improving optimization
│   └── run_date_time.pkl   # Timestamped training run logs and metrics
├── models/                 # Trained base models, meta-models, and feature scalers
│   ├── gru_model_*.h5      # Trained Bi-directional GRU models per horizon
│   ├── lstm_model_*.h5     # Trained Bi-directional LSTM models per horizon
│   ├── cnn_lstm_model_*.h5 # Trained CNN-LSTM models per horizon
│   ├── lgbm_model_*.pkl    # Trained LightGBM models per horizon
│   ├── xgb_model_*.pkl     # Trained XGBoost models per horizon
│   ├── meta_model_*.pkl    # Trained Ridge meta-models per horizon
│   └── scaler_*.pkl        # Feature scalers for normalization per horizon
├── predictions/            # Forecast outputs, prediction history, and backtest charts
│   ├── predictions.csv     # Complete prediction history and metadata
│   ├── future_forecast_1.png     # Latest forecast (generated on current run)
│   ├── future_forecast_2.png     # Previous forecast (from last run, now overlaid with actual price)
│   └──  future_forecast_3.png    # Forecast from 2 runs ago
├── sentiment/                    # Cached sentiment scores and trend visualizations
│   ├── sentiment_cache.json      # Cached sentiment scores by date
│   └── sentiment_trend_*.png     # Historical sentiment trend charts per ticker
└── LICENSE                       # MIT License documentation

```

## Getting Started

Follow these instructions to set up and run the Clairvoyant project.

### Prerequisites

*   **Python 3.8+:** Download and install from [python.org](https://www.python.org/downloads/).
*   **Git:** Install Git from [git-scm.com](https://git-scm.com/downloads).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [YOUR_REPOSITORY_URL_HERE]
    cd clairvoyant
    ```
    *(Note: Replace `[YOUR_REPOSITORY_URL_HERE]` with the actual clone URL)*

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    # On Windows:
    .\.venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

1.  **Create a `.env` file:** Copy the provided example environment file:
    ```bash
    copy .env.example .env  # On Windows
    cp .env.example .env    # On macOS/Linux
    ```

2.  **Edit `.env`:** Open the newly created `.env` file and populate it with your API keys and desired configuration settings.
    *   `OKX_API_KEY`, `OKX_SECRET_KEY`, `OKX_PASSPHRASE` (for OKX exchange data).
    *   `Binance_API_Key`, `Binance_Secret_Key` (for Binance exchange data).
    *   `EXCHANGERATE_HOST_KEY` (for Bithumb Korean Won conversion).
    *   Adjust `TICKER`, `TIMEFRAME`, `PREDICTION_HORIZONS`, `OPTIMIZE_HYPERPARAMETERS`, etc., as needed.

### Running the Application

The `clairvoyant2.py` script is the primary entry point for training and generating predictions.

*   **To train models:**
    ```bash
    python clairvoyant2.py --train
    ```
    This will run the full rolling walk-forward validation and train the final ensemble models. This process can be lengthy if hyperparameter optimization is enabled.

*   **To generate future predictions:**
    ```bash
    python clairvoyant2.py --predict
    ```
    This will load the latest trained models and generate future price forecasts based on current market conditions.

*   **To run in continuous training mode:**
    ```bash
    python clairvoyant2.py --training-mode
    ```
    This will continuously retrain the models.

## Key Features

The Clairvoyant platform offers a robust set of features designed to provide comprehensive and actionable insights for XRP price forecasting:

*   **Multi-Modal Data Ingestion:** Gathers data from Binance, OKX, Bithumb (Korean market data for Kimchi Premium), RSS news feeds, and the XRP Ledger for on-chain whale analytics.
*   **Advanced Feature Engineering:** Incorporates a wide array of technical indicators (RSI, MACD, Bollinger Bands, ATR, etc.), sentiment scores, derived leverage metrics (funding rates, open interest), and sophisticated ICT-style Institutional Order Block detection.
*   **Omniscient Whale Audit:** Dynamically scans the XRP Ledger for significant whale transaction activity, identifying net flows, buy/sell intensity, and market dominance from top accounts.
*   **Ensemble Modeling:** Utilizes an ensemble of powerful deep learning models (Bi-directional GRU, Bi-directional LSTM, CNN-LSTM) and machine learning models (LightGBM, XGBoost), meta-stacked with Ridge Regression for enhanced predictive power.
*   **Rolling Walk-Forward Validation:** Employs a rigorous time-series cross-validation strategy that simulates real-world trading conditions, providing a more reliable assessment of model performance over time.
*   **Dynamic Accuracy Metrics:** Beyond standard RMSE/MAE, it includes "Directional Accuracy" (predicting the correct price movement) and "Dynamic Half-Distance Accuracy" (prediction is correct if the actual price falls within 50% of the predicted price move).
*   **Confidence Interval Quantification:** Generates future predictions with adaptive confidence intervals, providing a measure of prediction certainty to aid risk management.
*   **TensorBoard Integration:** Logs extensive market context, training metrics, prediction quality, and model performance visualizations for detailed analysis and monitoring.

### How to Use (High-Level User Guide)

1.  **Prepare your environment:** Ensure Python and Git are installed, and clone the repository.
2.  **Install dependencies:** Use `pip install -r requirements.txt`.
3.  **Configure API keys:** Set up your `.env` file with necessary exchange and external service API keys.
4.  **Train the models:** Run `python clairvoyant2.py --train`. This will build and validate the ensemble models across multiple historical folds. Be aware that this is a time-intensive process, especially with hyperparameter optimization enabled.
5.  **Generate forecasts:** Once training is complete, run `python clairvoyant2.py --predict` to get the latest future price predictions.
6.  **Monitor progress & analyze results:** Use TensorBoard to visualize training progress, model performance, and market context by navigating to the `logs/` directory.

## Development Workflow

The project encourages an iterative development approach focused on enhancing model robustness, prediction accuracy, and feature engineering.

*   **Feature Branches:** Develop new features or significant changes in dedicated branches.
*   **Configuration:** Utilize the `.env` file for managing sensitive API keys and adjustable parameters.
*   **Logging:** Training runs generate timestamped directories in `logs/` for easy tracking and debugging.
*   **Model Persistence:** Trained models and scalers are automatically saved to the `models/` directory for reusability.
*   **Testing:** *(Note: Automated testing framework is not explicitly defined in the provided context, but assumed to be part of good practice.)*

## Performance Results

The Clairvoyant project emphasizes rigorous evaluation using a rolling walk-forward validation strategy. This approach provides a realistic assessment of how the models would perform over unseen data. Performance metrics are meticulously tracked and logged to TensorBoard for detailed analysis.

| Metric                        | Description                                                                                                                                                                                                                                                                                                          |
| :---------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Average RMSE**              | Measures the average magnitude of the errors in predictions, indicating how far off predictions are from actual values (in USDT terms). Lower values are better.                                                                                                                                                         |
| **Average MAE**               | Provides the average absolute difference between predicted and actual values, offering another perspective on prediction accuracy, less sensitive to large outliers than RMSE. Lower values are better.                                                                                                                |
| **Average Directional Accuracy** | The percentage of times the model correctly predicted the direction of the price movement (up or down) relative to the price at the time of prediction. Higher values are better.                                                                                                                                        |
| **Average Dynamic Accuracy**  | A unique metric that considers a prediction "correct" if the actual price falls within 50% of the predicted price movement's magnitude. This provides insight into whether the model captures the general scale of the move. Higher values are better.                                                                  |

### Methodology and Environment

The performance evaluation is conducted using `TimeSeriesSplit` cross-validation with 5 folds. Each fold involves hyperparameter optimization (when enabled) and training across an ensemble of 5 base models (GRU, LSTM, CNN-LSTM, LGBM, XGBoost), followed by a Ridge meta-model. The final performance is aggregated across these folds.

The computational environment for the last recorded training run showed significant processing times, with each subsequent fold taking longer due to the increasing size of the training datasets. Optimization trials for deep learning models can take minutes to hours per trial, accumulating into substantial overall training durations.

### Visualizations

The project generates several visualizations to aid performance analysis:

*   **Backtest Plots:** Found in `predictions/` (e.g., `backtest_6h.png`), these charts visualize actual vs. predicted price paths over the test sets of each fold.
*   **Sentiment Trend Plots:** Located in `sentiment/` (e.g., `sentiment_trend_XRP.png`), showing the historical trend of news sentiment.
*   **TensorBoard:** A rich visualization dashboard (`logs/run_YYYYMMDD_HHMMSS`) that provides detailed graphs for all logged metrics (loss, MAE, various accuracies, feature importances, market context, etc.). To view: `tensorboard --logdir=logs`

### Recommendations for Performance Improvements

*   **Hardware Acceleration:** Utilize GPUs (e.g., NVIDIA CUDA-enabled GPUs) for TensorFlow/Keras deep learning models to drastically reduce training times. Cloud GPU instances (AWS, GCP, Azure, Google Colab) are highly recommended.
*   **Hyperparameter Optimization Strategy:**
    *   Reduce `OPTUNA_TRIALS` in `.env` for faster but less exhaustive searches.
    *   Disable `OPTIMIZE_HYPERPARAMETERS` (`False` in `.env`) for quick model training runs (e.g., for testing new features), sacrificing potential performance gains from fine-tuned parameters.
*   **Cross-Validation Folds:** Adjust the `n_splits` parameter in `TimeSeriesSplit` (currently 5) to balance between robust evaluation and training time. Fewer splits lead to faster runs but potentially less reliable performance estimates.

### Conclusion

Clairvoyant offers a robust framework for XRP price forecasting, with a strong emphasis on comprehensive data integration and rigorous model validation. While the training process can be time-intensive, particularly with hyperparameter optimization, the detailed performance metrics and visualizations provide valuable insights into model behavior and potential future price movements. The combination of diverse data sources and ensemble modeling aims to capture complex market dynamics for improved predictive accuracy.

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2025 Kevin ₿ourn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

--- use following context as filler to take snippets or sections from to beef up the above Readme context ---
# CLAIRVOYANT v4 - THE ORACLE

<img src="/assets/clairvoyant-banner.jpg" alt="Clairvoyant Banner">

<p align="center">
  <a href="https://www.python.org/" alt="Badge: Python 3.10+"><img src="https://img.shields.io/badge/-3.10%2B-grey?style=flat-square&logo=python&logoColor=white&labelColor=%233776AB"></a>
  <a href="https://opensource.org/licenses/MIT" alt="License: MIT"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-blue?style=flat-square"></a>
  <a href="https://www.tensorflow.org/" alt="Badge: TensorFlow"><img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white"></a>
  <a href="https://optuna.org/" alt="Badge: Optuna"><img src="https://img.shields.io/badge/Optuna-800080?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTI4IDEyOCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cGF0aCBkPSJtMTI3LjQgNjMuN2MwIDMzLjEtMjYuOCA1OS45LTU5LjkgNTkuOXMtNTkuOS0yNi44LTU5LjktNTkuOSAyNi44LTU5LjkgNTkuOS01OS45IDU5LjkgMjYuOCA1OS45IDU5Ljl6IiBmaWxsPSIjODAwMDgwIj48L3BhdGg+PHBhdGggZD0ibTEwMS45IDY5Yy4zLS44LjUtMS42LjUtMi41cy0uMi0xLjctLjUtMi41bC0xLjcgMS45Yy4xLjYuMiAxLjIuMiAxLjhzLS4xIDEuMy0uMiAxLjlsMS43IDEuOXptLTUgOC43Yy41LTEuMS44LTIuMy44LTMuNnMtLjMtMi41LS44LTMuNmwtMS42IDEuOWMuMy45LjUgMS45LjUgMy4xIDAgMS4yLS4yIDIuMi0uNSAzLjFsMS42IDEuOXptLTQwLjYtMTUuMWMyLjgtMi40IDQuNC01LjkgNC40LTkuN3MtMS42LTcuMy00LjQtOS43bC0xLjYgMi4xYy0yLjUgMi00IDQuOS00IDcuNnMyLjIgNi4xIDUgOC4ybDEuNi0yLjF6bS00LjYgOC44YzMtMi45IDQuOC02LjkgNC44LTExLjNzLTEuOC04LjQtNC44LTExLjNsLTEuNiAyYy0yLjYgMi41LTQuMiA1L'-'OCA0LjIgOS4zcy45IDUuNyAyLjYgOC4ybDEuNi0yLjJ6bTIzLjQtMjMuNGMtLjQtLjMtLjktLjUtMS40LS41cy0xIC4yLTEuNC41bC0yLjQgMi40Yy0uNC40LS40IDEgMCAxLjQuNC40IDEgLjQgMS40IDBsMi40LTIuNGMuNC0uMy40LTEgMC0xLjR6bTQuOSA1Yy0uNC0uMy0uOS0uNS0xLjQtLjVzLTEgLjItMS40LjVsLTIuNCAyLjRjLS40LjQtLjQgMSAwIDEuNC40LjQgMSAuNCAxLjQgMGwyLjQtMi40Yy40LS40LjQtMS4xIDAtMS41em0tOS45IDExLjhjLS41IDAtLjkuMS0xLjQuMmwtMi42IDIuMmMtLjkuOC0xLjUgMi0xLjUgMy4zcy42IDIuNSAxLjUgMy4zbDIuNiAyLjJjLjQuMiAxIC4yIDEuNCAwIC45LS44IDEuNS0yIDEuNS0zLjNzLS42LTIuNS0xLjV'--'My4zbC0yLjYtMi4yYy0uNC0uMy0uOS0uMy0xLjQtLjN6bS0yNC42LTEyYy0uOC4zLTEuNS44LTEuOSAxcLTEuMyAxLjMtMS4zIDMuNCAwIDQuOC41LjYgMS4xIDEuMSAxLjkgMS41IDEuMS41IDIuNC41IDMuNSAwbDIuMy0yLjFjLTEuNS0uOC0yLjgtMi4yLTMuNS0zLjktLjctMS43LS43LTMuNSAwLTUuMmwzLjUtMy45Yy45LS45IDIuMS0xLjUgMy40LTEuNWgtLjFjLS40IDAtLjkuMS0xLjQgMy4xLTIuMi42LTMuNyAxLjgtNC41IDMuN3ptLTI5LjEtMTQuOWMtLjQgMC0uOS4xLTEuMi40bC0yLjYgMi42Yy0uMy4zLS4zLjkgMCAxLjIuMy4zLjkuMyAxLjIgMGwyLjYtMi42Yy4zLS4zLjMtLjkgMC0xLjJzLS44LS40LTEuMi0uNHptNC45LTVjLS40IDAtLjkuMS0xLjIuNGwtMi42IDIuNmMtLjMuMy0uMy45IDAgMS4yLjMuMy45LjMuMS4yIDBsMi42LTIuNmMtLjMtLjMtLjMtLjkgMC0xLjItLjQtLjQtLjktLjQtMS4yLS40em0yMC4xIDEuOWMtMS45LTEuOS00LjQtMi45LTYuOS0yLjloLS4xYy4zIDAgLjYuMS45LjJsMS44IDEuNGMuNy41IDEuMyAxLjIgMS43IDIuMWgxLjVjMS40IDAgMi44LS41IDMuOS0xLjZjLS45LTEtMi0xLjgtMy4yLTIuMnptLTQuOSA1Yy0xLjkgMS45LTQuNCAyLjktNi45IDIuOWgtLjFjLjMgMCAuNi0uMS45LS4ybDEuOC0xLjRjLjctLjUgMS4zLTEuMiAxLjctMi4xaDEuNWMxLjQgMCAyLjcuNSAzLjggMS42LTEgMS0yLjEgMS44LTMuMyAyLjJ6bS0zMy4yLTIzLjVjLTMuMy0uMS02LjYgMS4yLTkgMy41LTIuMyAyLjMtMy41IDUuMy0zLjUgOC43IDAgMy4zIDEuMiA2LjQgMy41IDguNyAyLjMgMi4zIDUuMyAzLjUgOC43IDMuNWgxYzUuNC4xIDEwLjktMi4xIDE1LjQtNi4zbC0xLjYtMS42Yy0zLjggMy42LTkuMyA1LjQtMTMuOCA1LjVoLS45Yy0yLjkgMC01LjctMS03LjktMi45cy0zLjUtNC42LTMuNS03LjYgMS01LjcgMi45LTcuOSAzLjUtNC42IDcuNi00LjZoLjljNC41LS4xIDkuMyAxLjggMTMuMSA1LjNsMS42LTEuNmMtNC4zLTQtOS41LTYuMi0xNC42LTYuM2gtMXoiIGZpbGw9IiNmZmYiIHBhY2luZyI9IjAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDIgMikiIHN0cm9rZS1kYXNoYXJyYXk9Im5vbmUiIHN0cm9rZS1taXRlcmxpbWl0PSIxMCIgc3Ryb2tlLXdpZHRoPSIxIj48L3BhdGg+PC9zdmc+"></a>
  <a href="https://github.com/ccxt/ccxt" alt="Badge: CCXT"><img src="https://img.shields.io/badge/-CCXT-black?style=flat-square&logo=x&logoColor=white&labelColor=black"></a>
  <a href="https://xrpscan.com/" alt="Badge: XRPScan"><img src="https://img.shields.io/badge/XRPScan-black?style=flat-square&logo=xrp&logoColor=white"></a>
  <a href="https://www.paypal.com/paypalme/kb90fund" alt="Badge: Fund my dev"><img src="https://img.shields.io/badge/-support_me-blue?style=flat-square&logo=paypal&logoColor=white"></a>
</p>

**Clairvoyant** is an AI-powered crypto oracle that delivers high-accuracy price forecasts by synthesizing a massive range of market, sentiment, on-chain, and leverage data. 

Version 4 is a quantum leap forward, integrating institutional-grade features like an **ICT Order Block Detector**, real-time **Kimchi Premium** analysis, and authenticated **exchange leverage data**. Combined with a self-improving **Optuna** optimization engine and a sophisticated stacking ensemble AI, Clairvoyant provides a multi-dimensional view of the market, complete with adaptive confidence intervals that tell you not just *what* it thinks, but *how sure* it is.

This is not just another technical analysis script. This is your personal AI market analyst.

---

## Table of Contents

- [Live Showcase](#live-showcase)
- [Why Clairvoyant?](#why-clairvoyant)
- [Core Features](#core-features)
- [How It Works: The Data Engine](#how-it-works-the-data-engine)
- [How It Works: The AI Architecture](#how-it-works-the-ai-architecture)
- [Getting Started](#getting-started)
- [How to Run](#how-to-run)
- [Understanding the Output](#understanding-the-output)
- [Hyperparameter Optimization](#hyperparameter-optimization-with-optuna)
- [Technology Stack](#technology-stack)
- [Disclaimer](#disclaimer)

---

## Live Showcase

Clairvoyant doesn't just predict; it provides a complete, visual narrative of its market analysis.

### Multi-Horizon Forecasting
The primary output is a single, unified chart showing predictions for multiple future time horizons. This allows you to see the expected price trajectory and potential turning points at a glance.

<p align="center">
  <img width="80%" src="/assets/future_forecast_readme_2.png" alt="Clairvoyant Multi-Horizon Forecast">
</p>

### Automated Back-Verification
A key feature of Clairvoyant is its ability to learn from the past. At the start of each run, it automatically loads its *previous* forecast and overlays the *actual price path* that occurred. This provides instant, visual feedback on the model's past performance and builds confidence in its future predictions.

*(The white line shows the actual price movement after the forecast was made)*
<p align="center">
  <img width="80%" src="https://i.imgur.com/example-verification.png" alt="Clairvoyant Forecast Verification">
  *Note: This is a representative image. Your plots will be generated in the `/predictions` directory.*
</p>

### Sentiment Analysis
The model continuously scrapes and analyzes news sentiment, plotting the trend to give you a feel for the market's mood.

<p align="center">
  <img width="80%" src="/assets/sentiment_trend_XRP_readme.png" alt="Clairvoyant Sentiment Trend Analysis">
</p>

---

## Why Clairvoyant?

- **Institutional-Grade Insights, Zero Hassle:** Clairvoyant brings sophisticated concepts like institutional order blocks, exchange leverage, and arbitrage gaps (Kimchi Premium) into a simple, automated script. You get the data edge without the complexity.
- **Truly Self-Improving AI:** With its persistent Optuna study database, the AI gets smarter with every single optimization run. It continuously builds on its past "experience" to find better models, forever.
- **Adaptive Confidence:** The model doesn't give you a single, reckless price target. It provides a dynamic confidence score and a price *range*. In stable markets, this range tightens for precision; in volatile markets, it widens to signal caution.
- **Total Transparency:** The core logic is contained in a single, well-documented Python file. The output is verbose, and TensorBoard integration gives you a god-mode view into every aspect of the model's "brain."
- **Plug-and-Play (Almost):** While some features require API keys (for live exchange data), the core forecasting engine can run without them. Configuration is centralized in a single `.env` file for easy tweaking.

---

## Core Features

<details>
<summary><strong>🤖 Stacking Ensemble AI</strong></summary>

- **Two-Layer Architecture**: A diverse set of five base models (Bi-GRU, Bi-LSTM, CNN-LSTM, LightGBM, XGBoost) capture different patterns in the data. Their predictions are then fed into a `Ridge` meta-model, which intelligently weights their inputs to produce a single, robust final forecast.
- **Best of Both Worlds**: Combines the power of deep learning models (which excel at temporal sequences) with gradient-boosted trees (which are masters of structured, tabular data).
</details>

<details>
<summary><strong>📈 Dynamic On-Chain Whale Tracking</strong></summary>

- **Real-Time & Directional**: The script queries the XRP Ledger directly to find the largest accounts. It then scans their recent transactions to determine not just *if* they're active, but whether they are **accumulating (buying)** or **distributing (selling)**.
- **Beyond Static Metrics**: This is a live, dynamic analysis of market-moving behavior, providing a true insight into whale intent.
</details>

<details>
<summary><strong>🏛️ Institutional Order Block Detector</strong></summary>

- **ICT-Based Logic**: Implements concepts from *Inner Circle Trader* methodology to identify high-probability support and resistance zones (order blocks) where institutional players are likely to have pending orders.
- **Smart & Practical**: It focuses only on recent, unmitigated zones within a practical distance of the current price, providing the two most critical levels: the nearest support below and the nearest resistance above.
</p>

<details>
<summary><strong>⚖️ Live Exchange Leverage & Funding Data</strong></summary>

- **Authenticated Data**: If configured, Clairvoyant connects directly to your OKX and Binance accounts to fetch real-time Open Interest (OI) and Funding Rates.
- **Leverage-Driven Features**: This data is used to engineer critical features like the 24h OI change, the ratio of OI on different exchanges, and the weekly OI Z-score, providing a clear view of market speculation and potential volatility.
</details>

<details>
<summary><strong>🇰🇷 Real-Time Kimchi Premium Analysis</strong></summary>

- **Arbitrage Signal**: Fetches live XRP prices from both the global market (USDT) and the Korean market (Bithumb, in KRW).
- **Live Conversion**: Uses a real-time currency conversion API to accurately calculate the "Kimchi Premium"—the price gap between the two markets, which is often a powerful indicator of retail sentiment and market direction.
</ar-premium)
- **Volume Dominance**: Also calculates the share of Korean volume vs. global volume.
</details>

<details>
<summary><strong>🧠 Persistent Hyperparameter Optimization (Optuna)</strong></summary>

- **Eternal Study Database**: Optimization results are saved to a central database. Every subsequent run loads the *entire history* of previous trials, meaning the system continuously compounds its "knowledge" to find better model parameters.
- **Automated & Integrated**: Simply set a flag in the `.env` file, and the script will automatically run an optimization study for every model and every prediction horizon.
</details>

<details>
<summary><strong>📊 Advanced Terminal UI & TensorBoard Logging</strong></summary>

- **Trader-Friendly CLI**: The terminal output is beautifully formatted, with color-coded metrics, progress bars, and prediction boxes that make the information easy to digest at a glance.
- **Deep Dive with TensorBoard**: Clairvoyant logs a rich set of data to TensorBoard, including market conditions, model performance, prediction quality, backtest charts, and feature importance plots, giving you an unparalleled view into the model's performance.
</details>

---

## How It Works: The Data Engine

Clairvoyant's power comes from its five distinct data pillars, which are engineered into a rich feature set.

<p align="center">
  <em>Data Pillars → Feature Engineering → Stacking Ensemble → Prediction</em>
</p>

1.  **Market Data Pillar**: Fetches thousands of historical OHLCV data points from Binance to build a baseline of price action and calculate technical indicators.
2.  **Sentiment Data Pillar**: Scrapes and analyzes news articles from over 14 financial RSS feeds using a crypto-specific sentiment lexicon to gauge market mood.
3.  **On-Chain Data Pillar**: Dynamically identifies the top whale accounts on the XRP Ledger and analyzes their transactions to determine directional buying or selling pressure.
4.  **Leverage Data Pillar**: Fetches live Open Interest and Funding Rates from major exchanges (OKX, Binance) to measure market speculation and the cost of leverage.
5.  **Korean Premium Pillar**: Fetches live price data from the Korean exchange Bithumb to calculate the Kimchi Premium, a key indicator of retail sentiment.

---

## How It Works: The AI Architecture

1.  **Feature Engineering**: The raw data from the pillars is transformed into hundreds of features, including dozens of technical indicators, market regime detection, and the calculated on-chain, leverage, and sentiment scores.
2.  **Base Models**: The feature set is fed to a diverse group of five "specialist" models:
    -   `Bi-GRU`, `Bi-LSTM`, `CNN-LSTM`: Deep learning models that excel at finding temporal patterns in sequence data.
    -   `LightGBM`, `XGBoost`: Gradient-boosted tree models that are highly effective on the structured, tabular features.
3.  **Meta-Model (The Ensemble)**: The predictions from these five base models are then passed to a final `Ridge` regression model. This "meta-model" learns how to best weight the opinions of the specialists to produce a single, highly accurate, and reliable final forecast.
4.  **Adaptive Confidence Interval**: Instead of using a simple standard deviation, Clairvoyant calculates its confidence by looking at the *degree of agreement* among the base models after filtering for outliers. If the models are in strong consensus, the confidence interval tightens. If they disagree, the interval widens, providing a natural, built-in risk indicator.

---

## Getting Started

### 1. Prerequisites
- Python 3.10+
- Git

### 2. Installation
Clone the repository and install the required dependencies.
```bash
git clone https://github.com/your-username/clairvoyant.git
cd clairvoyant
pip install -r requirements.txt
```

### 3. Configuration (`.env` file)
Create a file named `.env` in the project root directory. This is where you'll put all your settings and API keys.

**Copy this template into your `.env` file and fill it out:**

```dotenv
# === MAIN CONFIGURATION ===
TICKER="XRP/USDT"
TIMEFRAME="1h"
PREDICTION_HORIZONS="6,12,24"
DATA_LIMIT=5000
SEQUENCE_LENGTH=60

# === TRAINING & OPTIMIZATION ===
OPTIMIZE_HYPERPARAMETERS="False"
OPTUNA_TRIALS=50
DL_EPOCHS=50
CV_EPOCHS_DL=50
CV_PATIENCE=8

# === EXCHANGE API KEYS (OPTIONAL BUT RECOMMENDED) ===
# Required for live Open Interest and Funding Rate data
OKX_API_KEY=""
OKX_SECRET_KEY=""
OKX_PASSPHRASE=""
Binance_API_Key=""
Binance_Secret_Key=""

# === CURRENCY CONVERSION API KEY (OPTIONAL BUT RECOMMENDED) ===
# Required for live Kimchi Premium calculation
# Get a free key from https://exchangerate.host/
EXCHANGERATE_HOST_KEY=""
```

#### Configuration Details
| Parameter | Description | Default | Recommendation / Notes |
|---|---|---|---|
| **MAIN CONFIGURATION** | | | |
| `TICKER` | The cryptocurrency pair to trade (from Binance). | `XRP/USDT` | Any valid `ccxt` pair. On-chain analysis only works for `XRP`. |
| `TIMEFRAME` | The candle timeframe for the data. | `1h` | Standard `ccxt` timeframes (`30m`, `4h`, `1d`). |
| `PREDICTION_HORIZONS` | Comma-separated list of future hours to predict. | `6,12,24` | The script trains a separate model for each horizon. |
| `DATA_LIMIT` | Number of historical candles to fetch for training. | `5000` | More data can improve accuracy but increases training time. |
| `SEQUENCE_LENGTH` | Number of past time steps for deep learning models. | `60` | Affects memory usage. Change only if you understand LSTMs. |
| **TRAINING & OPTIMIZATION** | | | |
| `OPTIMIZE_HYPERPARAMETERS` | Set to `True` to run the Optuna optimization study. | `False` | **Crucial!** See the [Optuna section](#hyperparameter-optimization-with-optuna) below. |
| `OPTUNA_TRIALS` | Number of trials to run *per model, per horizon*. | `50` | Active only if the above is `True`. |
| `DL_EPOCHS` | Number of epochs for the final deep learning model training. | `50` | Higher values can lead to better models but risk overfitting. |
| `CV_EPOCHS_DL` | Epochs used *during* cross-validation/optimization. | `50` | Lower than `DL_EPOCHS` for faster optimization runs. |
| `CV_PATIENCE` | Number of epochs with no improvement before early stopping. | `8` | Prevents wasting time on non-converging models. |
| **API KEYS** | | | |
| `OKX_*` / `Binance_*` | API credentials for OKX and Binance. | `""` | **Required** for the Leverage Data Pillar (OI, Funding Rates). The script will run without them but with reduced accuracy. |
| `EXCHANGERATE_HOST_KEY`| Free API key for currency conversion. | `""` | **Required** for the Korean Premium Pillar. Get one at `exchangerate.host`. |

---

## How to Run

Clairvoyant uses command-line arguments to determine its mode of operation.

### Default Mode (Train & Predict)
If you run the script with no arguments, it will automatically train all models and then generate a future prediction.
```bash
python clairvoyant.py
```

### Train-Only Mode
Use the `--train` flag to only train and save the models without making a final prediction. This is useful for scheduled training jobs.
```bash
python clairvoyant.py --train
```

### Predict-Only Mode
Use the `--predict` flag to load existing trained models and generate a new forecast. This is very fast and is the recommended way to get quick updates.
```bash
python clairvoyant.py --predict
```

### Continuous Training Mode
Use the `--training-mode` flag to run the training pipeline in an infinite loop. This is for dedicated machines where you want the model to be constantly re-training on the latest data.
```bash
python clairvoyant.py --training-mode
# Press CTRL+C to stop
```

---

## Understanding the Output

### Terminal Output
The script provides a rich, color-coded output in the terminal, guiding you through each phase of the process, from data fetching to the final prediction summary.

### Project Directory Structure
The script will generate the following directories:

```
clairvoyant/
├── predictions/          # Prediction logs (predictions.csv) and forecast charts
│   ├── predictions.csv
│   ├── future_forecast_1.png   (Latest forecast)
│   ├── future_forecast_2.png   (Previous forecast, now with actuals)
│   └── future_forecast_3.png   (Forecast from 2 runs ago)
├── models/               # All trained model files (.pkl) and scalers
├── logs/                 # Timestamped logs for TensorBoard and Optuna
│   ├── optuna_global/
│   │   └── eternal_optuna_studies.db  # The persistent Optuna database
│   └── run_YYYYMMDD_HHMMSS/        # Rich logs for each specific run
└── sentiment/            # Cached sentiment data and trend plots
```

### Using TensorBoard
For a deep analysis of model performance, run TensorBoard and point it to the `logs` directory.
```bash
tensorboard --logdir logs
```
This will launch a web server (usually at `http://localhost:6006`) where you can explore everything from market conditions and feature importance to detailed backtest performance charts.

---

## Hyperparameter Optimization with Optuna

Clairvoyant's most powerful feature is its ability to self-improve via **Optuna**.

- **How it Works**: When `OPTIMIZE_HYPERPARAMETERS` is set to `True`, the script runs a study to find the best possible settings (hyperparameters) for each of the 5 base models and for each prediction horizon.
- **Persistent Intelligence**: The results are saved in `/logs/optuna_global/eternal_optuna_studies.db`. Every new optimization run loads this history, meaning the system learns from all past trials and doesn't start from scratch. It continuously refines its understanding of what makes a good model.

### Calculating the Number of Trials
The total number of optimization trials is: **(Number of Models) x (Number of Prediction Horizons) x (`OPTUNA_TRIALS`)**
- With default settings (`PREDICTION_HORIZONS="6,12,24"` and `OPTUNA_TRIALS=50`), this is: `5 models * 3 horizons * 50 trials = 750` total runs. This can take a significant amount of time.

### Recommended Workflow
1.  **First Run**: For your very first run, set `OPTIMIZE_HYPERPARAMETERS=True` and `OPTUNA_TRIALS=100` (or more, if you have time). This will build a strong baseline of high-quality models.
2.  **Daily Predictions**: For daily use, set `OPTIMIZE_HYPERPARAMETERS=False` and run with `python clairvoyant.py --predict`. This will be very fast as it just uses the best-found parameters.
3.  **Periodic Re-Optimization**: Once a week or month, run another optimization study (e.g., with `OPTUNA_TRIALS=25`). Because the study is persistent, even a small number of new trials will help the models adapt to new market dynamics and contribute to the long-term intelligence of the system.

---

## Technology Stack
- **AI & Machine Learning**: TensorFlow, Scikit-learn, Optuna, XGBoost, LightGBM
- **Data & Analysis**: Pandas, NumPy, TA (Technical Analysis)
- **Data Fetching**: CCXT, aiohttp, feedparser, requests
- **NLP & Sentiment**: NLTK, Afinn

---

## Disclaimer
This project and its predictions are for educational and informational purposes only. Cryptocurrency markets are extremely volatile. Always conduct your own research and do not consider this as financial advice. Past performance is not indicative of future results. You are solely responsible for any financial decisions you make.