import os
import sys

# 1. HARDWARE OPTIMIZATION (Must be absolute first)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'      # Hide the K4100M
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'       # Boost the i7-4800MQ
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        # Silence the "No GPU" screaming

# 2. STANDARD & ASYNC UTILITIES
import time
import json
import logging
import asyncio
import re
import io
import argparse
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple

# 3. DATA & NETWORK
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
import requests
import aiohttp
import websockets
import feedparser
import ccxt
import joblib

# 4. CRYPTO & UI TOOLS
from terminal_styles import TerminalColors, TerminalStyle
from xrpscan import IntelligentWhaleScanner
from lexicon.crypto_lexicon import CRYPTO_LEXICON
import nltk
from afinn import Afinn

# 5. TECHNICAL ANALYSIS (Grouped for speed)
import ta.momentum as mom
import ta.trend as trd
import ta.volatility as vol
import ta.volume as vlm

# 6. MACHINE LEARNING (The Heavy Lifters)
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans

# Boosting
from lightgbm import LGBMRegressor
import xgboost as xgb

# TensorFlow / Keras (Loading this last saves startup hangs)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    GRU, LSTM, Dense, Dropout, Input, 
    Bidirectional, Conv1D, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback

# Authenticated exchanges for funding/OI (using .env credentials)
okx_api_key = os.getenv('OKX_API_KEY')
okx_secret_key = os.getenv('OKX_SECRET_KEY')
okx_password = os.getenv('OKX_PASSPHRASE')

okx_exchange = None
if all([okx_api_key, okx_secret_key, okx_password]):
    try:
        okx_exchange = ccxt.okx({
            'apiKey': okx_api_key,
            'secret': okx_secret_key,
            'password': okx_password,
            'enableRateLimit': True,  # FIX: Prevents many OKX rate/signing bugs
            'options': {'defaultType': 'swap'},
        })
        okx_exchange.load_markets()  # This line often triggers the bug if ccxt version bad
        TerminalStyle.success("OKX authenticated markets loaded")
    except Exception as e:
        TerminalStyle.warning(f"OKX initialization failed: {e}")
        okx_exchange = None
else:
    TerminalStyle.warning("OKX API credentials incomplete or missing — skipping authenticated features (funding/OI zeroed)")

# Binance
binance_api_key = os.getenv('Binance_API_Key')
binance_secret_key = os.getenv('Binance_Secret_Key')



binance_exchange = None
if binance_api_key and binance_secret_key:
    try:
        binance_exchange = ccxt.binance({
            'apiKey': binance_api_key,
            'secret': binance_secret_key,
            'enableRateLimit': True,  # Recommended for stability
            'options': {'defaultType': 'future'},
        })
        binance_exchange.load_markets()
        TerminalStyle.success("Binance authenticated markets loaded")
    except Exception as e:
        TerminalStyle.warning(f"Binance initialization failed: {e}")
        binance_exchange = None
else:
    TerminalStyle.warning("Binance API credentials not found — skipping authenticated Binance")

# Public Bithumb for XRPKRW data (always public)
bithumb_exchange = ccxt.bithumb()
try:
    bithumb_exchange.load_markets()
    TerminalStyle.success("BITHUMB markets loaded (XRPKRW)")
except Exception as e:
    TerminalStyle.warning(f"Bithumb market load warning: {e}")

# -----------------------------
# Public fallback exchanges (used later for OHLCV / basic data when authenticated fails)
# -----------------------------
public_binance = ccxt.binance({'enableRateLimit': True})
public_okx = ccxt.okx({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})

try:
    public_binance.load_markets()
    TerminalStyle.info("Public Binance fallback loaded successfully")
except Exception as e:
    TerminalStyle.warning(f"Public Binance fallback load failed: {e}")

try:
    public_okx.load_markets()
    TerminalStyle.info("Public OKX fallback loaded successfully")
except Exception as e:
    TerminalStyle.warning(f"Public OKX fallback load failed: {e}")

# Final safety: Log overall exchange status
if okx_exchange is None:
    TerminalStyle.warning("OKX authenticated unavailable — funding/OI features will be zeroed")
if binance_exchange is None:
    TerminalStyle.warning("Binance authenticated unavailable — some features limited to public data")

# -----------------------------
# Optional: Delay or conditional load_markets for authenticated exchanges (if bug persists)
# Uncomment if ccxt version issue continues after upgrade
# if okx_exchange is not None:
#     try:
#         okx_exchange.load_markets()
#     except:
#         pass  # Already warned
# if binance_exchange is not None:
#     try:
#         binance_exchange.load_markets()
#     except:
#         pass
# -----------------------------

# Import Optuna for hyperparameter tuning
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    OPTUNA_AVAILABLE = False


def get_global_optuna_study(model_name: str, horizon: int, ticker: str):
    """
    ONE SINGLE ETERNAL OPTUNA STUDY PER MODEL+HORIZON
    Lives forever in logs/optuna_global/
    Every run merges into the same study → compounds intelligence forever
    """
    if not OPTUNA_AVAILABLE or optuna is None:
        return None

    global_db_dir = Path("logs/optuna_global")
    global_db_dir.mkdir(parents=True, exist_ok=True)
    
    # One eternal DB for ALL runs of this model+horizon
    db_path = global_db_dir / "eternal_optuna_studies.db"
    
    study_name = f"{ticker.split('/')[0]}_{horizon}h_{model_name.upper()}"
    
    return optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{db_path.resolve()}",
        direction="minimize",
        load_if_exists=True,        # ← merges with every past run automatically
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=20)
    )


# --- 0. CONFIGURATION & SETUP ---
load_dotenv()
MODEL_DIR = Path("models")

# CONFIDENCE INTERVAL CONFIGURATION
CI_METHOD = 'adaptive'

def get_env_int(key, default):
    """Gets an integer from an environment variable, stripping non-numeric chars."""
    value = os.getenv(key, str(default))
    # Remove non-digit characters so "50s" becomes "50"
    sanitized_value = re.sub(r'\D', '', value)
    try:
        return int(sanitized_value)
    except ValueError:
        return default

# --- TRAINING CONFIGURATION ---
# Parameters are now loaded from the .env file.
# Default values are provided as a fallback.
TRAINING_CONFIG = {
    # adjust in .env
    "TICKER": os.getenv("TICKER", "XRP/USDT"),
    # adjust in .env
    "TIMEFRAME": os.getenv("TIMEFRAME", "1h"),
    # adjust in .env - Comma-separated list of integers
    "PREDICTION_HORIZONS": [int(h) for h in os.getenv("PREDICTION_HORIZONS", "12,24").split(',')],
    # adjust in .env
    "SEQUENCE_LENGTH": get_env_int("SEQUENCE_LENGTH", 168),
    # adjust in .env
    "DATA_LIMIT": get_env_int("DATA_LIMIT", 3000),
    # adjust in .env
    "CV_EPOCHS_DL": get_env_int("CV_EPOCHS_DL", 100),
    # adjust in .env
    "CV_PATIENCE": get_env_int("CV_PATIENCE", 12),
    # adjust in .env
    "DL_EPOCHS": get_env_int("DL_EPOCHS", 100),
    # adjust in .env - Must be "True" or "False"
    "OPTIMIZE_HYPERPARAMETERS": os.getenv("OPTIMIZE_HYPERPARAMETERS", "false").lower() in ('true', '1', 't'),
    # adjust in .env
    "OPTUNA_TRIALS": get_env_int("OPTUNA_TRIALS", 25),
    "DIRECTIONAL_MOVE_THRESHOLD_PCT": float(os.getenv("DIRECTIONAL_MOVE_THRESHOLD_PCT", "0.1")),
    "CONFIDENCE_FILTER_THRESHOLD": int(os.getenv("CONFIDENCE_FILTER_THRESHOLD", "66")),
    "WHALE_THRESHOLD": int(os.getenv("WHALE_THRESHOLD", "100000")),

    # --- Model-specific parameters (automatically loaded from Optuna or fallback defaults) ---
    # NOTE: These are FALLBACK DEFAULTS only. When Optuna studies exist with best params,
    # those are used instead. This dict is only used when no Optuna study has been completed yet.
    # To force reset and use these defaults, delete logs/optuna_global/eternal_optuna_studies.db
    "MODEL_PARAMS": {
        "rf": {"n_estimators": 100, "max_depth": 20, "min_samples_split": 5, "random_state": 42, "n_jobs": -1},
        "lgbm": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 10, "random_state": 42, "verbose": -1, "n_jobs": -1},
        "xgb": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 10, "random_state": 42, "tree_method": "hist", "n_jobs": -1},
        "gbm": {"n_estimators": 150, "learning_rate": 0.05, "max_depth": 8, "random_state": 42},
    }
}


def load_best_optuna_params(model_name: str, horizon: int, ticker: str) -> dict:
    """
    Loads the best hyperparameters from Optuna for a given model+horizon+ticker.
    Returns empty dict if no study exists or study has no trials.
    
    This ensures that when OPTIMIZE_HYPERPARAMETERS=False, we still use
    the best params found in previous runs (stored in eternal_optuna_studies.db).
    """
    if not OPTUNA_AVAILABLE or optuna is None:
        return {}
    
    try:
        global_db_dir = Path("logs/optuna_global")
        db_path = global_db_dir / "eternal_optuna_studies.db"
        
        if not db_path.exists():
            return {}  # No Optuna DB yet
        
        study_name = f"{ticker.split('/')[0]}_{horizon}h_{model_name.upper()}"
        
        # Load the study without creating a new one
        study = optuna.load_study(
            study_name=study_name,
            storage=f"sqlite:///{db_path.resolve()}"
        )
        
        # Return best params if trials exist
        if study.best_trial is not None:
            return study.best_params
        else:
            return {}
    except Exception as e:
        # Study doesn't exist or other error - silently return empty
        return {}

# Set up custom logging formatter
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors"""
    
    FORMATS = {
        logging.DEBUG: f"{TerminalColors.DIM}%(message)s{TerminalColors.ENDC}",
        logging.INFO: f"{TerminalColors.BLUE}● %(message)s{TerminalColors.ENDC}",
        logging.WARNING: f"{TerminalColors.ORANGE}⚠ %(message)s{TerminalColors.ENDC}",
        logging.ERROR: f"{TerminalColors.RED}✗ %(message)s{TerminalColors.ENDC}",
        logging.CRITICAL: f"{TerminalColors.RED}{TerminalColors.BOLD}✗ %(message)s{TerminalColors.ENDC}"
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

logger = logging.getLogger('clairvoyant')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter())
logger.addHandler(handler)

# One-time download for NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    TerminalStyle.info("Downloading required NLTK data (punkt)...")
    nltk.download('punkt', quiet=True)
    TerminalStyle.success("NLTK data downloaded")

class UltimateOrderBlockDetector:
    """
    Practical ICT Order Block Detector for 1h XRP/USDT
    - Focuses on recent zones (last 300 candles)
    - Filters far-away zones (>20% from price)
    - Always finds nearest support below and resistance above
    - Counts real touches
    """
    def __init__(self, lookback_candles=300, max_distance_pct=20.0, swing_length=7, min_impulse_pct=3.0):
        self.lookback_candles = lookback_candles
        self.max_distance_pct = max_distance_pct
        self.swing_length = swing_length
        self.min_impulse_pct = min_impulse_pct

    def detect_and_add_features(self, df):
        df = df.copy()
        original_index = df.index
        df = df.reset_index(drop=True)
        
        # Only analyze recent data
        start_idx = max(0, len(df) - self.lookback_candles)
        df_recent = df.iloc[start_idx:].copy()
        df_recent = df_recent.reset_index(drop=True)
        
        # Detect swings in recent data
        df_recent['swing_high'] = self._detect_swings(df_recent['high'], True)
        df_recent['swing_low'] = self._detect_swings(df_recent['low'], False)
        
        # Detect OBs
        bullish_obs = self._detect_bullish_obs(df_recent)
        bearish_obs = self._detect_bearish_obs(df_recent)
        
        # Adjust indices back
        for ob in bullish_obs + bearish_obs:
            ob['index'] += start_idx
        
        # Filter active + in-range zones
        active_zones = self._filter_active_zones(df, bullish_obs + bearish_obs)
        
        total_detected = len(bullish_obs) + len(bearish_obs)
        TerminalStyle.info(f"Detected {len(bullish_obs)} bullish OBs, {len(bearish_obs)} bearish OBs → {len(active_zones)} active (last {self.lookback_candles} candles)")
        
        # Add features
        df = self._add_ml_features(df, active_zones)
        df.index = original_index
        return df

    def _detect_swings(self, series, is_high):
        swings = np.full(len(series), False)
        for i in range(self.swing_length, len(series) - self.swing_length):
            window = series[i-self.swing_length:i+self.swing_length+1]
            if (is_high and series[i] == window.max()) or (not is_high and series[i] == window.min()):
                swings[i] = True
        return swings

    def _detect_bullish_obs(self, df):
        obs = []
        swings = df[df['swing_low']].index
        for idx in swings:
            if idx < 10: continue
            for j in range(idx-1, max(idx-25, -1), -1):
                candle = df.iloc[j]
                if candle['close'] < candle['open']:  # Bearish candle
                    future_high = df.iloc[idx:idx+12]['high'].max()
                    impulse = (future_high - candle['low']) / candle['low'] * 100
                    if impulse >= self.min_impulse_pct:
                        obs.append({
                            'index': j,
                            'high': candle['high'],
                            'low': candle['low'],
                            'mid': (candle['high'] + candle['low']) / 2,
                            'strength': impulse,
                            'type': 'bullish'
                        })
                        break
        return obs

    def _detect_bearish_obs(self, df):
        obs = []
        swings = df[df['swing_high']].index
        for idx in swings:
            if idx < 10: continue
            for j in range(idx-1, max(idx-25, -1), -1):
                candle = df.iloc[j]
                if candle['close'] > candle['open']:  # Bullish candle
                    future_low = df.iloc[idx:idx+12]['low'].min()
                    impulse = (candle['high'] - future_low) / candle['high'] * 100
                    if impulse >= self.min_impulse_pct:
                        obs.append({
                            'index': j,
                            'high': candle['high'],
                            'low': candle['low'],
                            'mid': (candle['high'] + candle['low']) / 2,
                            'strength': impulse,
                            'type': 'bearish'
                        })
                        break
        return obs

    def _filter_active_zones(self, df, all_obs):
        current_price = df['close'].iloc[-1]
        active = []
        
        for ob in all_obs:
            future_df = df[df.index > ob['index']]
            violated = (ob['type'] == 'bullish' and future_df['close'].min() < ob['low']) or \
                       (ob['type'] == 'bearish' and future_df['close'].max() > ob['high'])
            if violated: continue
            
            dist_pct = abs(current_price - ob['mid']) / current_price * 100
            if dist_pct > self.max_distance_pct: continue
            
            ob['dist_pct'] = dist_pct
            ob['touches'] = self._count_touches(df, ob)
            active.append(ob)
        
        # Sort by distance
        active = sorted(active, key=lambda x: x['dist_pct'])
        return active[:8]  # Top 8 nearest

    def _count_touches(self, df, ob):
        future_df = df[df.index > ob['index']]
        touches = 0
        for _, row in future_df.iterrows():
            if ob['type'] == 'bullish':
                if row['low'] <= ob['high'] and row['close'] > ob['low']:
                    touches += 1
            else:
                if row['high'] >= ob['low'] and row['close'] < ob['high']:
                    touches += 1
        return min(touches, 10)

    def _add_ml_features(self, df, active_zones):
        current_price = df['close'].iloc[-1]
        
        # Defaults
        df['ob_nearest_support_dist'] = 999.0
        df['ob_nearest_resistance_dist'] = 999.0
        df['ob_support_touches'] = 0
        df['ob_resistance_touches'] = 0
        df['ob_support_strength'] = 0.0
        df['ob_resistance_strength'] = 0.0
        
        support = None
        resistance = None
        
        for zone in active_zones:
            if zone['type'] == 'bullish' and zone['mid'] < current_price:
                if support is None or zone['dist_pct'] < support['dist_pct']:
                    support = zone
            elif zone['type'] == 'bearish' and zone['mid'] > current_price:
                if resistance is None or zone['dist_pct'] < resistance['dist_pct']:
                    resistance = zone
        
        if support:
            df['ob_nearest_support_dist'] = support['dist_pct']
            df['ob_support_touches'] = support['touches']
            df['ob_support_strength'] = support['strength']
        
        if resistance:
            df['ob_nearest_resistance_dist'] = resistance['dist_pct']
            df['ob_resistance_touches'] = resistance['touches']
            df['ob_resistance_strength'] = resistance['strength']
        
        # --- NEW POWER BIAS LOGIC ---
        # Calculate raw 'power' for the nearest levels
        # Power = Strength (impulse %) weighted by the number of successful touches
        support_power = (support['strength'] * (1 + support['touches']/10)) if support else 0
        resistance_power = (resistance['strength'] * (1 + resistance['touches']/10)) if resistance else 0

        # Feature: Power Differential (Positive = Bullish Wall, Negative = Bearish Wall)
        df['ob_power_differential'] = support_power - resistance_power

        # Feature: Wall Density (Total active zones in the 20% range)
        df['ob_zone_density'] = len(active_zones)
        
        s_level = f"{support['low']:.4f}" if support else "None"
        r_level = f"{resistance['high']:.4f}" if resistance else "None"
        TerminalStyle.info(f"Nearest Support: {s_level} | Nearest Resistance: {r_level} | Power Differential: {df['ob_power_differential'].iloc[-1]:+.1f}")
        
        TerminalStyle.success("Ultimate Order Block features added")
        return df


# --- 1. DYNAMIC ON-CHAIN ANALYSIS ENGINE ---



def fetch_funding_rates_authenticated():
    """Fetch current funding rates using authenticated OKX + Binance"""
    funding_rates = {}
    sources = []

    # OKX
    if okx_exchange:
        try:
            funding = okx_exchange.fetch_funding_rate('XRP-USDT-SWAP')
            funding_rates['okx'] = funding['fundingRate']
            sources.append('OKX')
        except Exception as e:
            TerminalStyle.error(f"OKX funding fetch failed: {e}")
            TerminalStyle.error(f"OKX raw response on failure: {e.args}")

    # Binance
    try:
        funding = binance_exchange.fetch_funding_rate('XRPUSDT')
        funding_rates['binance'] = funding['fundingRate']
        sources.append('Binance')
    except Exception as e:
        TerminalStyle.error(f"Binance funding fetch failed: {e}")
        TerminalStyle.error(f"Binance raw response on failure: {e.args}")

    if funding_rates:
        avg = sum(funding_rates.values()) / len(funding_rates)
        TerminalStyle.info(f"Funding avg: {avg} from {', '.join(sources)}")
        return avg, ', '.join(sources)
    TerminalStyle.info("No funding data fetched")
    return 0.0, "None"

def fetch_current_oi_authenticated():
    """Fetch current OI using authenticated OKX + Binance"""
    oi_total = 0
    oi_okx = 0
    sources = []

    # OKX - prioritize oiUsd (standard for OKX perps, often returned as string)
    if okx_exchange:
        try:
            oi_data = okx_exchange.fetch_open_interest('XRP-USDT-SWAP')
            # Handle string values from OKX
            oi_value_str = (oi_data.get('oiUsd') or 
                            oi_data.get('oi') or 
                            oi_data.get('openInterestAmount') or 
                            oi_data.get('openInterest') or '0')
            oi_value = float(oi_value_str)
            if oi_value == 0:
                TerminalStyle.info(f"OKX raw OI response (debug): {oi_data}")
            oi_total += oi_value
            oi_okx = oi_value
            sources.append('OKX')
            TerminalStyle.info(f"OKX OI: {oi_value}")
        except Exception as e:
            TerminalStyle.error(f"OKX OI fetch failed: {e}")
            TerminalStyle.error(f"OKX raw response on failure: {e.args}")

    # Binance
    try:
        oi_data = binance_exchange.fetch_open_interest('XRPUSDT')
        oi_value = oi_data.get('openInterestAmount', oi_data.get('openInterest', 0))
        oi_total += oi_value
        sources.append('Binance')
        TerminalStyle.info(f"Binance OI: {oi_value}")
    except Exception as e:
        TerminalStyle.error(f"Binance OI fetch failed: {e}")
        TerminalStyle.error(f"Binance raw response on failure: {e.args}")

    TerminalStyle.info(f"Total OI: {oi_total} from {', '.join(sources)}")
    return oi_total, oi_okx, ', '.join(sources)


def fetch_bithumb_krw_data(df_main):
    """Fetch REAL Bithumb XRPKRW OHLCV, convert properly to USD, add Kimchi premium"""
    try:
        # Fetch real XRP/KRW from Bithumb only
        ohlcv = bithumb_exchange.fetch_ohlcv('XRP/KRW', timeframe=TRAINING_CONFIG["TIMEFRAME"], limit=TRAINING_CONFIG["DATA_LIMIT"] + 100)
        if not ohlcv:
            TerminalStyle.warning("No Bithumb XRPKRW data fetched")
            return df_main
        
        df_krw = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df_krw['timestamp'] = pd.to_datetime(df_krw['timestamp'], unit='ms', utc=True)
        df_krw.set_index('timestamp', inplace=True)
        
        # Live KRW/USD from exchangerate.host (free key required)
        api_key = os.getenv('EXCHANGERATE_HOST_KEY')
        krw_usd_rate = 0.00073  # Fallback (Dec 2025 approx)
        if api_key:
            try:
                url = f"https://api.exchangerate.host/live?access_key={api_key}&source=USD&currencies=KRW"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success') and 'quotes' in data and 'USDKRW' in data['quotes']:
                        usd_to_krw = data['quotes']['USDKRW']
                        krw_usd_rate = 1 / usd_to_krw
                        TerminalStyle.info(f"Live KRW/USD rate (exchangerate.host): {krw_usd_rate:.6f}")
                    else:
                        TerminalStyle.warning("exchangerate.host API error in response")
            except Exception as e:
                TerminalStyle.warning(f"KRW/USD API failed: {e} — using fallback")
        else:
            TerminalStyle.warning("No EXCHANGERATE_HOST_KEY in .env — using fallback rate")
        
        # Convert Bithumb KRW prices and volume to USD
        df_krw['price_usd'] = df_krw['close'] * krw_usd_rate
        df_krw['volume_usd'] = df_krw['volume'] * df_krw['close'] * krw_usd_rate  # Approx USD volume
        
        # Align timestamps with main df (nearest match)
        df_krw = df_krw.reindex(df_main.index, method='nearest')
        
        # Add to main df
        df_main['bithumb_price_usd'] = df_krw['price_usd']
        df_main['bithumb_volume_usd'] = df_krw['volume_usd']
        
        # Kimchi premium: Korean price vs global USDT price
        df_main['kimchi_premium_pct'] = (df_main['bithumb_price_usd'] / df_main['close']) - 1
        
        # Korean volume dominance
        usdt_volume_usd = df_main['volume'] * df_main['close']
        total_volume_usd = df_main['bithumb_volume_usd'] + usdt_volume_usd
        df_main['korean_volume_share'] = df_main['bithumb_volume_usd'] / (total_volume_usd + 1e-8)
        
        avg_premium = df_main['kimchi_premium_pct'].mean() * 100
        TerminalStyle.info(f"Bithumb XRPKRW data added: {len(df_krw)} rows, avg Kimchi premium: {avg_premium:.1f}%")
        TerminalStyle.success("Real Korean (Bithumb XRPKRW) features added")
        
    except Exception as e:
        TerminalStyle.error(f"Bithumb Korean data failed: {e}")
        TerminalStyle.warning("Korean features skipped")
    
    return df_main



# --- 2. ENHANCED SENTIMENT ANALYSIS ENGINE ---

def calculate_sentiment(text: str, afinn: Afinn) -> float:
    """
    Calculates a Context-Aware 'Neural' sentiment score.
    Includes N-gram matching, Negation flipping, and Emphasis boosting.
    """
    total_score = 0.0
    relevant_items = 0
    
    # Import lexicons dynamically
    try:
        from lexicon import crypto_lexicon as _crypto_mod
        CRYPTO_LEXICON = getattr(_crypto_mod, 'CRYPTO_LEXICON', {})
        CRYPTO_NGRAMS = getattr(_crypto_mod, 'CRYPTO_NGRAMS', {})
    except Exception:
        CRYPTO_LEXICON = {}
        CRYPTO_NGRAMS = {}

    # Work on a lower-cased version for matching, keep original for emphasis
    text_lower = text.lower()
    
    # 1. N-Gram Matching (Greedy)
    # We iterate through N-grams and if found, add score and "consume" the text
    # so individual words aren't double-counted.
    for phrase, score in CRYPTO_NGRAMS.items():
        if phrase in text_lower:
            count = text_lower.count(phrase)
            total_score += (score * count)
            relevant_items += count
            # Remove the phrase from text_lower to prevent single-word matching
            # We replace with spaces to preserve word boundaries
            text_lower = text_lower.replace(phrase, " " * len(phrase))

    # 2. Word-level Analysis with Negation and Emphasis
    # We use the potentially modified 'text_lower' for word matching, 
    # but we need to map back to original text for CAPS check. 
    # For simplicity in this 'lightweight' neural map, we'll check CAPS on the token itself 
    # if it matches the modified lower string.
    
    words = re.findall(r"\b[\w']+\b", text_lower)
    # To check emphasis, we need the original words. This alignment is tricky after replacement.
    # heuristic: just check if the word exists in original text as ALL CAPS.
    
    negation_words = {'not', 'no', 'never', "don't", "doesnt", "didn't", "cant", "wont", "wouldnt", "isn't", "aren't"}
    
    i = 0
    while i < len(words):
        word_lower = words[i]
        
        # Skip if it was part of a removed n-gram (it would be gone or spaces)
        if not word_lower.strip(): 
            i += 1
            continue

        # Check for negation in previous word
        is_negated = False
        if i > 0 and words[i-1] in negation_words:
            is_negated = True
        
        # Determine base score
        score = 0.0
        found = False
        
        if word_lower in CRYPTO_LEXICON:
            score = CRYPTO_LEXICON[word_lower]
            found = True
        else:
            af_score = afinn.score(word_lower)
            if af_score != 0:
                score = af_score
                found = True
        
        if found:
            # Apply Negation
            if is_negated:
                # "Not bullish" (5.0) -> -5.0
                # "Not scam" (-4.5) -> +2.25 (Dampened positive)
                if score > 0:
                    score = -score
                else:
                    score = abs(score) * 0.5
            
            # Apply Emphasis (CAPS CHECK)
            # We look for the word in the original text. 
            # If the exact word exists in UPPERCASE, boost it.
            # This is a loose check but efficient.
            if word_lower.upper() in text:
                score *= 1.5
            
            total_score += score
            relevant_items += 1
            
        i += 1

    return total_score / relevant_items if relevant_items > 0 else 0.0

def analyze_sentiment(text: str) -> float:
    """
    Analyzes sentiment using AFINN + crypto lexicon.
    """
    chunks = [chunk.strip() for chunk in text.replace('.', ',').split(',') if chunk.strip()]
    if not chunks:
        return 0.0
    
    afinn = Afinn()
    chunk_scores = []
    
    for chunk in chunks:
        score = calculate_sentiment(chunk, afinn)
        chunk_scores.append(score)
        
    final_score = np.mean(chunk_scores)
    return np.clip(final_score / 5.0, -1.0, 1.0)

# --- ENHANCED RSS PARSING ---
async def fetch_feed(session, url):
    """
    Asynchronously fetches and parses a single RSS feed.
    """
    try:
        async with session.get(url, timeout=15) as response:
            if response.status == 200:
                content = await response.text()
                return feedparser.parse(content)
            logger.warning(f"Failed to fetch {url}, status: {response.status}")
            return None
    except asyncio.TimeoutError:
        logger.warning(f"Timeout error fetching RSS feed from {url}")
        return None
    except aiohttp.ClientError as e:
        logger.warning(f"Client error fetching {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error while fetching {url}: {e}")
        return None

async def scrape_financial_articles() -> List[dict]:
    """
    Fetches financial news articles from a predefined list of RSS feeds.
    """
    rss_urls = [
        'https://cointelegraph.com/rss',
        'https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml',
        'https://decrypt.co/feed',
        'https://crypto.news/feed/',
        'https://u.today/rss',
        'https://bitcoinist.com/category/ripple/feed/',
        'https://dailyhodl.com/ripple-and-xrp/feed/',
        'https://ambcrypto.com/tag/ripple/feed/',
        'https://www.newsbtc.com/feed/',
        'https://cryptobriefing.com/feed/',
        'https://beincrypto.com/feed/',
        'https://coingape.com/feed/',
        'https://cryptonews.com/news/feed/',
        'https://rsscrypto.com/feed/'
    ]
    articles = []
    processed_titles = set()
    twenty_four_hours_ago = pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=24)

    async with aiohttp.ClientSession(headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}) as session:
        tasks = [fetch_feed(session, url) for url in rss_urls]
        feeds = await asyncio.gather(*tasks)

    for feed in feeds:
        if not feed:
            continue
        for entry in feed.entries[:30]:
            title = entry.get('title', '')
            pub_date = entry.get('published', None)
            timestamp = pd.to_datetime(pub_date, utc=True, errors='coerce')
            if pd.isna(timestamp):
                # Fall back to published_parsed if available
                published_parsed = entry.get('published_parsed') or entry.get('updated_parsed')
                if published_parsed:
                    try:
                        timestamp = pd.to_datetime(time.strftime('%a, %d %b %Y %H:%M:%S %z', published_parsed), utc=True, errors='coerce')
                    except Exception:
                        timestamp = pd.NaT
                if pd.isna(timestamp):
                    timestamp = pd.Timestamp.now(tz='UTC')

            if title and title not in processed_titles and timestamp >= twenty_four_hours_ago:
                processed_titles.add(title)
                summary = entry.get('summary', entry.get('description', ''))
                full_text = f"{title}. {summary}"
                articles.append({'text': full_text, 'timestamp': timestamp, 'title': title})

    TerminalStyle.success(f"Found {len(articles)} unique articles from the last 24 hours")
    return articles

def analyze_news_sentiment(articles: List[dict], ticker_symbol: str) -> pd.DataFrame:
    """
    Analyzes and scores news articles for a specific ticker, applying weights.
    """
    all_scores = []
    now = pd.Timestamp.now(tz='UTC')
    ticker_lower = ticker_symbol.lower()

    for article in articles:
        text = article.get('text', '')

        if ticker_lower not in text.lower():
            continue

        score = analyze_sentiment(text)
        timestamp = article['timestamp']
        
        age_hours = (now - timestamp).total_seconds() / 3600
        
        if age_hours <= 3:
            weight = 1.5
        elif age_hours <= 6:
            weight = 1.2
        elif age_hours <= 12:
            weight = 1.0
        elif age_hours <= 24:
            weight = 0.8
        elif age_hours <= 48:
            weight = 0.6
        else:
            weight = 0.4

        if ticker_lower in article.get('title', '').lower():
            weight *= 1.2

        all_scores.append({
            'timestamp': timestamp,
            'sentiment': score,
            'weight': weight
        })

    if all_scores:
        df = pd.DataFrame(all_scores)
        df['weighted_sentiment'] = df['sentiment'] * df['weight']
        
        result = df.groupby('timestamp').agg({
            'weighted_sentiment': 'sum',
            'weight': 'sum'
        })
        
        result['news_sentiment'] = result['weighted_sentiment'] / result['weight']
        result['news_sentiment'] = np.clip(result['news_sentiment'], -1, 1)
        return result[['news_sentiment']].sort_index()
    else:
        return pd.DataFrame()

# --- 3. DATA LOADING & SENTIMENT INTEGRATION ---
def load_data(ticker='XRP/USDT', timeframe='1h', limit=1000):
    """
    Loads historical OHLCV data from Binance with error handling and retries.
    """
    exchange = ccxt.binance({'enableRateLimit': True})
    ms_per_unit = exchange.parse_timeframe(timeframe) * 1000
    all_ohlcv = []
    since = exchange.milliseconds() - (limit * ms_per_unit)
    TerminalStyle.info(f"Fetching ~{limit} data points for {ticker} ({timeframe})")
    
    retries = 0
    max_retries = 5
    seen_timestamps = set()  # OPTIMIZED: Track unique timestamps
    
    while len(all_ohlcv) < limit and retries < max_retries:
        try:
            ohlcv = exchange.fetch_ohlcv(ticker, timeframe, since=since, limit=1000)
            if not ohlcv:
                logger.warning("No more OHLCV data available from exchange")
                break
            since = ohlcv[-1][0] + 1
            
            # OPTIMIZED: Single-pass deduplication
            for candle in ohlcv:
                ts = candle[0]
                if ts not in seen_timestamps:
                    seen_timestamps.add(ts)
                    all_ohlcv.append(candle)
            
            retries = 0
            TerminalStyle.progress(len(all_ohlcv), limit, "Fetching data")
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            retries += 1
            logger.warning(f"Network issue (attempt {retries}/{max_retries}): {e}")
            time.sleep(5 * retries)
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error: {e}. Aborting")
            break
        except Exception as e:
            retries += 1
            logger.error(f"Unexpected error (attempt {retries}/{max_retries}): {e}")
            time.sleep(5 * retries)
    
    TerminalStyle.success(f"Fetched {len(all_ohlcv)} data points")
    if not all_ohlcv:
        logger.error("Could not fetch any OHLCV data. Returning empty DataFrame")
        return pd.DataFrame()
        
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.index = df.index.tz_localize('UTC')
    df.sort_index(inplace=True)
    return df.tail(limit)

def fetch_and_add_sentiment(df, ticker_symbol='XRP'):
    """
    Fetches and integrates news sentiment data into the main DataFrame.
    """
    import matplotlib.pyplot as plt
    # Ensure main dataframe index is UTC-aware
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
    Path('sentiment').mkdir(parents=True, exist_ok=True)
    news_path = Path('sentiment') / f'news_sentiment_{ticker_symbol}.csv'
    
    try:
        news_df = pd.read_csv(news_path, index_col='timestamp', parse_dates=True)
        # Normalize index to UTC. Localize naive timestamps, convert aware timestamps to UTC
        if news_df.index.tz is None:
            news_df.index = news_df.index.tz_localize('UTC')
        else:
            news_df.index = news_df.index.tz_convert('UTC')
        last_cache_time = news_df.index.max()
        if (pd.Timestamp.now(tz='UTC') - last_cache_time) < pd.Timedelta(hours=1):
            TerminalStyle.success("Using cached sentiment data")
            news_df_hourly = news_df.resample('h').mean()
            df = df.join(news_df_hourly, how='left')
            if not df['news_sentiment'].dropna().empty:
                avg_sent = df['news_sentiment'].dropna().mean()
                TerminalStyle.info(f"ⓘ Average Sentiment: {avg_sent:+.3f}")
            df['sentiment'] = df['news_sentiment'].bfill().ffill().fillna(0)
            del df['news_sentiment']
            
            plt.figure(figsize=(10, 5))
            df['sentiment'].tail(24).plot(label='Hourly News Sentiment', marker='o', linestyle='-')
            plt.title(f'{ticker_symbol} Hourly News Sentiment (Last 24h)')
            plt.ylabel('Sentiment Score (-1 to 1)')
            plt.legend()
            plt.grid(True)
            plot_path = Path('sentiment') / f'sentiment_trend_{ticker_symbol}.png'
            plt.savefig(plot_path)
            plt.close()
            TerminalStyle.success(f"Saved sentiment plot to {plot_path}")
            return df
        else:
            TerminalStyle.info("Cache expired, refreshing sentiment data...")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        TerminalStyle.info("No cache found, fetching fresh sentiment data...")
    except Exception as e:
        logger.warning(f"Could not read cache: {e}. Fetching fresh data")

    try:
        try:
            articles = asyncio.run(scrape_financial_articles())
        except RuntimeError as e:
            if 'already running' in str(e):
                loop = asyncio.get_event_loop()
                articles = loop.run_until_complete(scrape_financial_articles())
            else:
                raise
        news_sent_df = analyze_news_sentiment(articles, ticker_symbol)
        if not news_sent_df.empty:
            news_sent_df.to_csv(news_path)
            news_df_hourly = news_sent_df.resample('h').mean()
            df = df.join(news_df_hourly, how='left')
            if not df['news_sentiment'].dropna().empty:
                avg_sent = df['news_sentiment'].dropna().mean()
                TerminalStyle.info(f"ⓘ Average Sentiment: {avg_sent:+.3f}")
            df['sentiment'] = df['news_sentiment'].bfill().ffill().fillna(0)
            del df['news_sentiment']
        else:
            df['sentiment'] = 0.0
            TerminalStyle.warning("No news data found, using neutral sentiment")
    except aiohttp.ClientError as e:
        logger.error(f"Network error during news fetch: {e}. Using neutral sentiment")
        df['sentiment'] = 0.0
    except Exception as e:
        logger.error(f"Unexpected error during news fetch: {e}. Using neutral sentiment")
        df['sentiment'] = 0.0

    plt.figure(figsize=(10, 5))
    df['sentiment'].tail(24).plot(label='Hourly News Sentiment', marker='o', linestyle='-')
    plt.title(f'{ticker_symbol} Hourly News Sentiment (Last 24h)')
    plt.ylabel('Sentiment Score (-1 to 1)')
    plt.legend()
    plt.grid(True)
    plot_path = Path('sentiment') / f'sentiment_trend_{ticker_symbol}.png'
    plt.savefig(plot_path)
    plt.close()
    TerminalStyle.success(f"Saved sentiment plot to {plot_path}")
    
    return df

# --- 4. FEATURE ENGINEERING ---
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers a comprehensive set of features for the model.
    """
    if df.empty:
        TerminalStyle.warning("Empty DataFrame provided to feature_engineering. Returning empty DataFrame.")
        return pd.DataFrame()

    required_columns = ['close', 'high', 'low', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        TerminalStyle.warning(f"Missing required columns for feature engineering: {missing_columns}. Returning original DataFrame.")
        return df

    # Momentum Indicators
    rsi = mom.RSIIndicator(df['close'], window=14)
    df['rsi'] = rsi.rsi()

    stoch = mom.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()

    roc = mom.ROCIndicator(df['close'], window=12)
    df['roc'] = roc.roc()

    # Trend Indicators
    macd = trd.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()

    ema12 = trd.EMAIndicator(df['close'], window=12)
    ema26 = trd.EMAIndicator(df['close'], window=26)
    df['ema12'] = ema12.ema_indicator()
    df['ema26'] = ema26.ema_indicator()
    df['ema_cross'] = df['ema12'] - df['ema26']

    adx = trd.ADXIndicator(df['high'], df['low'], df['close'], window=14)
    df['adx'] = adx.adx()

    cci = trd.CCIIndicator(df['high'], df['low'], df['close'], window=20)
    df['cci'] = cci.cci()

    # Volatility Indicators
    bb = vol.BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['close']

    atr = vol.AverageTrueRange(df['high'], df['low'], df['close'], window=14)
    df['atr'] = atr.average_true_range()

    kc = vol.KeltnerChannel(df['high'], df['low'], df['close'], window=20)
    df['kc_high'] = kc.keltner_channel_hband()
    df['kc_low'] = kc.keltner_channel_lband()

    # Volume Indicators
    obv = vlm.OnBalanceVolumeIndicator(df['close'], df['volume'])
    df['obv'] = obv.on_balance_volume()

    mfi = vlm.MFIIndicator(df['high'], df['low'], df['close'], df['volume'], window=14)
    df['mfi'] = mfi.money_flow_index()

    cmf = vlm.ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume'], window=20)
    df['cmf'] = cmf.chaikin_money_flow()

    # Price Features
    df['close_lag1'] = df['close'].shift(1)
    df['close_lag2'] = df['close'].shift(2)
    df['volume_ma5'] = df['volume'].rolling(window=5).mean()

    initial_rows = len(df)
    df.dropna(inplace=True)
    final_rows = len(df)
    if initial_rows > final_rows:
        logger.info(f"Dropped {initial_rows - final_rows} rows with NaN values during feature engineering")

    TerminalStyle.success("Feature engineering completed")
    return df

def detect_market_regime(df: pd.DataFrame, log_dir: str = None) -> pd.DataFrame:
    """
    Detects market regimes using K-Means clustering on key features.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    if df.empty:
        TerminalStyle.warning("Empty DataFrame provided to detect_market_regime. Returning empty DataFrame.")
        return pd.DataFrame()

    regime_features = ['rsi', 'macd', 'bb_width', 'atr', 'adx']
    missing_features = [f for f in regime_features if f not in df.columns]
    if missing_features:
        TerminalStyle.warning(f"Missing features for market regime detection: {missing_features}. Skipping regime detection.")
        df['market_regime'] = 0
        return df

    X_regime = df[regime_features].dropna()
    if len(X_regime) < 3:
        TerminalStyle.warning("Insufficient data for market regime detection. Defaulting to neutral regime.")
        df['market_regime'] = 0
        return df

    # Scale features before clustering to avoid scale dominance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_regime)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['market_regime'] = 0
    df.loc[X_regime.index, 'market_regime'] = kmeans.fit_predict(X_scaled)

    if log_dir:
        plt.figure(figsize=(10, 8))
        corr_matrix = df[regime_features + ['market_regime']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Feature Correlations with Market Regimes')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=300)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)

        file_writer = tf.summary.create_file_writer(log_dir)
        with file_writer.as_default():
            tf.summary.image("Market Regime Correlation Heatmap", image, step=0)
        plt.close()

    TerminalStyle.success("Market regime detection completed")
    return df


def enforce_feature_consistency(df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """
    Ensures the DataFrame has a consistent set of columns in a fixed order.
    Adds debug logs for column presence if requested.
    """
    # Define all expected columns in a canonical order
    base_cols = ['open', 'high', 'low', 'close', 'volume']
    
    ta_cols = [
        'rsi', 'stoch_k', 'stoch_d', 'roc', 'macd', 'macd_signal', 'ema12', 
        'ema26', 'ema_cross', 'adx', 'cci', 'bb_high', 'bb_low', 'bb_width', 
        'atr', 'kc_high', 'kc_low', 'obv', 'mfi', 'cmf', 'close_lag1', 
        'close_lag2', 'volume_ma5'
    ]
    
    sentiment_cols = ['sentiment']
    
    onchain_cols = [
        'whale_market_dominance', 'whale_sell_intensity', 'whale_buy_intensity', 'whale_net_flow_momentum',
        'exchange_inventory_momentum', 'whale_pressure_score', 'whale_net_flow_raw', 'whale_vol_6h'
    ]
    
    regime_cols = ['market_regime']
    
    leverage_cols = [
        'funding_rate_avg', 'open_interest_total', 'open_interest_okx',
        'funding_rate_8h_ma', 'oi_total_change_24h', 'oi_okx_ratio',
        'oi_z_score_week'
    ]
    
    korean_cols = [
        'bithumb_price_usd', 'bithumb_volume_usd', 'kimchi_premium_pct', 'korean_volume_share'
    ]

    ob_cols = [
        'ob_nearest_support_dist', 'ob_nearest_resistance_dist', 'ob_support_touches',
        'ob_resistance_touches', 'ob_support_strength', 'ob_resistance_strength',
        'ob_power_differential', 'ob_zone_density'
    ]

    # The full, ordered list of feature columns
    all_feature_cols = base_cols + ta_cols + sentiment_cols + onchain_cols + regime_cols + leverage_cols + korean_cols + ob_cols
    
    # Check for missing columns and log them
    missing_cols = [col for col in all_feature_cols if col not in df.columns]
    
    if debug:
        if not missing_cols:
            TerminalStyle.info("All columns are present.")
        else:
            for col in missing_cols:
                logger.warning(f"Column '{col}' is missing. It will be created and filled with 0.")

    # Reindex the DataFrame. This will add missing columns (filled with 0.0)
    # and ensure the exact column order.
    df = df.reindex(columns=all_feature_cols, fill_value=0.0)

    if debug:
        TerminalStyle.success("Feature set is now consistent.")
        
    return df


# --- 5. MODEL ARCHITECTURES & WRAPPERS ---
def get_default_hyperparameters(model_type: str) -> dict:
    """
    Returns default hyperparameters for a given model type.
    """
    if model_type in ['gru', 'lstm']:
        return {
            'n_units': 128,
            'dropout': 0.2,
            'learning_rate': 0.001
        }
    elif model_type == 'cnn_lstm':
        return {
            'conv_filters': 64,
            'kernel_size': 3,
            'n_units': 128,
            'dropout': 0.2,
            'learning_rate': 0.001
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def create_dl_model(input_shape: tuple, n_units: int = 128, dropout: float = 0.2, model_type: str = 'gru') -> Sequential:
    """
    Creates a deep learning model (GRU or LSTM) with Bidirectional layers.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    RecurrentLayer = GRU if model_type == 'gru' else LSTM
    model.add(Bidirectional(RecurrentLayer(n_units, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    
    model.add(Bidirectional(RecurrentLayer(n_units)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    
    return model

def create_cnn_lstm_model(input_shape: tuple, conv_filters: int = 64, kernel_size: int = 3, n_units: int = 128, dropout: float = 0.2) -> Sequential:
    """
    Creates a CNN-LSTM hybrid model.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    model.add(Conv1D(filters=conv_filters, kernel_size=kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    
    model.add(Bidirectional(LSTM(n_units, return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    
    model.add(Bidirectional(LSTM(n_units)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))
    
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    
    return model

class StopIfNaN(Callback):
    """Custom callback to stop training if NaN loss is detected."""
    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        loss = logs.get('loss')
        if loss is not None and np.isnan(loss):
            self.model.stop_training = True
            logger.warning("NaN loss detected. Stopping training.")




class TradingTensorBoard:
    """
    Simplified TensorBoard logger focused on trader-relevant metrics.
    Organized into clear dashboards: Market, Training, Predictions, Performance
    """
    def __init__(self, log_dir: Path, horizon: int):
        self.log_dir = Path(log_dir)
        self.horizon = horizon
        
        # Create separate writers for different dashboards
        self.writers = {
            'market': tf.summary.create_file_writer(str(self.log_dir / 'market_data')),
            'training': tf.summary.create_file_writer(str(self.log_dir / 'training_metrics')),
            'predictions': tf.summary.create_file_writer(str(self.log_dir / 'prediction_quality')),
            'performance': tf.summary.create_file_writer(str(self.log_dir / 'model_performance'))
        }
        
        self.step = 0
    
    def log_market_context(self, df: pd.DataFrame):
        """Log current market conditions - what traders need to see"""
        with self.writers['market'].as_default():
            # Use last 100 points for context
            recent_df = df.tail(100).reset_index()
            
            for _, row in recent_df.iterrows():
                step = int(row['timestamp'].timestamp())
                
                # Price action
                tf.summary.scalar('Price/Current_USDT', row['close'], step=step)
                
                # Volatility indicators
                if 'atr' in row:
                    tf.summary.scalar('Volatility/ATR', row['atr'], step=step)
                if 'bb_width' in row:
                    tf.summary.scalar('Volatility/Bollinger_Width', row['bb_width'], step=step)
                
                # Momentum
                if 'rsi' in row:
                    tf.summary.scalar('Momentum/RSI', row['rsi'], step=step)
                if 'macd' in row:
                    tf.summary.scalar('Momentum/MACD', row['macd'], step=step)
                
                # Sentiment
                if 'sentiment' in row:
                    tf.summary.scalar('Sentiment/News_Score', row['sentiment'], step=step)
                
                # On-chain (if available)
                if 'whale_net_flow' in row:
                    tf.summary.scalar('OnChain/Whale_Net_Flow_XRP', row['whale_net_flow'], step=step)
                if 'whale_accumulation_pressure' in row and row['whale_accumulation_pressure'] > 0:
                    tf.summary.scalar('OnChain/Whale_Accumulation_Pressure', row['whale_accumulation_pressure'], step=step)
                if 'whale_distribution_pressure' in row and row['whale_distribution_pressure'] > 0:
                    tf.summary.scalar('OnChain/Whale_Distribution_Pressure', row['whale_distribution_pressure'], step=step)
            
            # Market regime distribution
            if 'market_regime' in df.columns:
                regime_counts = df['market_regime'].value_counts()
                for regime, count in regime_counts.items():
                    tf.summary.scalar(f'Market_Regime/Regime_{regime}_Frequency', 
                                    count / len(df) * 100, step=0)
        
        self.writers['market'].flush()
    
    def log_whale_activity(self, whale_df: pd.DataFrame):
        """Log on-chain whale activity in real-time"""
        if whale_df.empty:
            return
            
        with self.writers['market'].as_default():
            # Aggregate metrics
            total_buy_volume = whale_df['whale_buy_volume'].sum()
            total_sell_volume = whale_df['whale_sell_volume'].sum()
            total_buy_txs = whale_df['whale_buy_count'].sum()
            total_sell_txs = whale_df['whale_sell_count'].sum()
            net_flow = total_buy_volume - total_sell_volume
            
            tf.summary.scalar('OnChain_Summary/Total_Whale_Buy_Volume_6h', total_buy_volume, step=0)
            tf.summary.scalar('OnChain_Summary/Total_Whale_Sell_Volume_6h', total_sell_volume, step=0)
            tf.summary.scalar('OnChain_Summary/Total_Whale_Net_Flow_6h', net_flow, step=0)
            tf.summary.scalar('OnChain_Summary/Total_Whale_Buy_Txs_6h', total_buy_txs, step=0)
            tf.summary.scalar('OnChain_Summary/Total_Whale_Sell_Txs_6h', total_sell_txs, step=0)
            
            # Time series of whale activity
            for _, row in whale_df.iterrows():
                step = int(row['timestamp'].timestamp())
                if row['whale_buy_volume'] > 0:
                    tf.summary.scalar('OnChain_TimeSeries/Whale_Buys', 
                                    row['whale_buy_volume'], step=step)
                if row['whale_sell_volume'] > 0:
                    tf.summary.scalar('OnChain_TimeSeries/Whale_Sells', 
                                    row['whale_sell_volume'], step=step)
        
        self.writers['market'].flush()
    
    def log_training_epoch(self, epoch: int, metrics: Dict[str, float], 
                          model_name: str, is_validation: bool = False):
        """Log training metrics per epoch"""
        prefix = 'Validation' if is_validation else 'Training'
        
        with self.writers['training'].as_default():
            # Core training metrics
            if 'loss' in metrics:
                tf.summary.scalar(f'{model_name}/{prefix}_Loss', 
                                metrics['loss'], step=epoch)
            if 'mae' in metrics:
                tf.summary.scalar(f'{model_name}/{prefix}_MAE', 
                                metrics['mae'], step=epoch)
            
            # Learning dynamics
            if 'lr' in metrics:
                tf.summary.scalar(f'{model_name}/Learning_Rate', 
                                metrics['lr'], step=epoch)
        
        self.writers['training'].flush()
    
    def log_cv_fold_results(self, fold: int, metrics: Dict[str, float]):
        """Log cross-validation results"""
        with self.writers['performance'].as_default():
            tf.summary.scalar(f'CrossValidation_{self.horizon}h/RMSE_Fold_{fold}', 
                            metrics['rmse'], step=fold)
            tf.summary.scalar(f'CrossValidation_{self.horizon}h/MAE_Fold_{fold}', 
                            metrics['mae'], step=fold)
            
            if 'directional_accuracy' in metrics:
                tf.summary.scalar(f'CrossValidation_{self.horizon}h/Direction_Accuracy_Fold_{fold}', 
                                metrics['directional_accuracy'], step=fold)
        
        self.writers['performance'].flush()
    
    def log_final_cv_summary(self, avg_metrics: Dict[str, float]):
        """Log aggregated CV results"""
        with self.writers['performance'].as_default():
            tf.summary.scalar(f'CrossValidation_{self.horizon}h/Average_RMSE', 
                            avg_metrics['rmse'], step=0)
            tf.summary.scalar(f'CrossValidation_{self.horizon}h/Average_MAE', 
                            avg_metrics['mae'], step=0)
            
            if 'directional_accuracy' in avg_metrics:
                tf.summary.scalar(f'CrossValidation_{self.horizon}h/Average_Direction_Accuracy', 
                                avg_metrics['directional_accuracy'], step=0)
            if 'dynamic_accuracy' in avg_metrics:
                tf.summary.scalar(f'CrossValidation_{self.horizon}h/Average_Dynamic_Accuracy',
                                avg_metrics['dynamic_accuracy'], step=0)
            if 'filtered_directional_accuracy' in avg_metrics:
                tf.summary.scalar(f'CrossValidation_{self.horizon}h/Filtered_Directional_Accuracy',
                                avg_metrics['filtered_directional_accuracy'], step=0)
            if 'filtered_dynamic_accuracy' in avg_metrics:
                tf.summary.scalar(f'CrossValidation_{self.horizon}h/Filtered_Dynamic_Accuracy',
                                avg_metrics['filtered_dynamic_accuracy'], step=0)

        
        self.writers['performance'].flush()
    
    def log_prediction_quality(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               current_prices: np.ndarray, confidence_scores: np.ndarray, fold: Optional[int] = None):
        """Log prediction quality metrics aligned with Trader's Mind Painting"""
        # Ensure arrays are flattened for consistent calculations
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        current_prices = current_prices.flatten()
        confidence_scores = confidence_scores.flatten()

        # Handle potential length mismatch
        min_len = min(len(y_true), len(y_pred), len(current_prices), len(confidence_scores))
        if len(y_true) != min_len:
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
            current_prices = current_prices[:min_len]
            confidence_scores = confidence_scores[:min_len]

        # --- STEP 1: Directional Accuracy (Settlement Bias) ---
        # Binary Win/Loss based on 0.0 threshold
        actual_change = y_true - current_prices
        pred_change = y_pred - current_prices
        
        # Win if signs match AND actual change is not zero (or if both are zero, technically a match but rare)
        # Using simple sign multiplication: positive product = same sign
        same_direction = (np.sign(actual_change) * np.sign(pred_change)) > 0
        
        # Handle cases where actual change is 0 (neutral) - usually counts as correct if pred is also near 0, 
        # but for binary classification often excluded or counted as loss. 
        # Let's count exact 0 match as correct, or 0 actual as 'no trend' (exclude).
        # Instruction says: "Any move in the right direction is a WIN."
        # We'll use the sign check.
        
        directional_accuracy = np.mean(same_direction) * 100

        # --- STEP 2: Dynamic Directional Accuracy (Significant Moves) ---
        # Filter out "choppy" noise. Win ONLY IF Step 1 is WIN AND Actual Move >= Threshold
        threshold_pct = TRAINING_CONFIG["DIRECTIONAL_MOVE_THRESHOLD_PCT"]
        actual_change_pct = np.abs(actual_change / current_prices) * 100
        
        is_significant = actual_change_pct >= threshold_pct
        
        # Metric: Of the significant moves, how many did we predict correctly?
        if np.sum(is_significant) > 0:
            # Filter matches by significance
            matches_on_significant = same_direction[is_significant]
            dynamic_accuracy = np.mean(matches_on_significant) * 100
        else:
            dynamic_accuracy = 0.0

        # --- Standard Metrics ---
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        safe_y_true = np.where(y_true == 0, 1e-9, y_true)
        mape = np.mean(np.abs((y_true - y_pred) / safe_y_true)) * 100
        
        step = fold if fold is not None else 0
        suffix = f'_Fold_{fold}' if fold is not None else '_Final'
        
        with self.writers['predictions'].as_default():
            tf.summary.scalar(f'Accuracy_{self.horizon}h/RMSE{suffix}', rmse, step=step)
            tf.summary.scalar(f'Accuracy_{self.horizon}h/MAE{suffix}', mae, step=step)
            tf.summary.scalar(f'Accuracy_{self.horizon}h/MAPE{suffix}', mape, step=step)
            tf.summary.scalar(f'Accuracy_{self.horizon}h/Directional_Accuracy{suffix}', 
                            directional_accuracy, step=step)
            tf.summary.scalar(f'Accuracy_{self.horizon}h/Dynamic_Accuracy{suffix}', 
                            dynamic_accuracy, step=step)
        
        self.writers['predictions'].flush()
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'dynamic_accuracy': dynamic_accuracy,
            'directional_correct_mask': same_direction,
            'significant_move_mask': is_significant,
            'dynamic_correct_mask': same_direction & is_significant, # For filtered aggregation later if needed
            'confidence_scores': confidence_scores 
        }
    
    def log_prediction_distribution(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Visualize prediction vs actual distribution"""
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.5, s=10)
        axes[0].plot([y_true.min(), y_true.max()], 
                     [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual Price')
        axes[0].set_ylabel('Predicted Price')
        axes[0].set_title(f'{self.horizon}h Prediction Accuracy')
        axes[0].grid(True, alpha=0.3)
        
        # Error distribution
        errors = ((y_pred - y_true) / y_true) * 100
        axes[1].hist(errors, bins=50, alpha=0.7, edgecolor='black')
        axes[1].axvline(0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Prediction Error (%)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Error Distribution')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to tensor
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        plt.close()
        
        with self.writers['predictions'].as_default():
            tf.summary.image(f'Prediction_Analysis_{self.horizon}h', image, step=0)
        
        self.writers['predictions'].flush()
    
    def log_backtest_chart(self, timestamps: pd.DatetimeIndex, 
                          y_true: np.ndarray, y_pred: np.ndarray):
        """Log backtest visualization"""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(15, 6))
        
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        ax.plot(timestamps, y_true_flat, label='Actual Price', 
                color='#00FF00', linewidth=2, alpha=0.8)
        ax.plot(timestamps, y_pred_flat, label='Predicted Price', 
                color='#FF00FF', linewidth=2, linestyle='--', alpha=0.8)
        
        # Highlight prediction errors
        error_pct = np.abs((y_true_flat - y_pred_flat) / y_true_flat) * 100
        colors = ['green' if e < 2 else 'ORANGE' if e < 5 else 'red' 
                  for e in error_pct]
        ax.scatter(timestamps, y_pred_flat, c=colors, s=20, alpha=0.6, 
                  label='Error: Green<2%, ORANGE<5%, Red>5%')
        
        ax.set_title(f'{self.horizon}h Backtest Performance', fontsize=14)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USDT)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Convert to tensor
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        plt.close()
        
        with self.writers['predictions'].as_default():
            tf.summary.image(f'Backtest_{self.horizon}h', image, step=0)
        
        self.writers['predictions'].flush()
    
    def log_feature_importance(self, feature_names: list, importances: np.ndarray):
        """Log feature importance for interpretability"""
        import matplotlib.pyplot as plt
        # Sort features by importance
        indices = np.argsort(importances)[-20:]  # Top 20
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(top_features)), top_importances)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel('Importance')
        ax.set_title(f'Top 20 Features for {self.horizon}h Prediction')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Convert to tensor
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        plt.close()
        
        with self.writers['training'].as_default():
            tf.summary.image(f'Feature_Importance_{self.horizon}h', image, step=0)
        
        self.writers['training'].flush()
    
    def close(self):
        """Close all writers"""
        for writer in self.writers.values():
            writer.close()

class KerasRegressorWrapper:
    """
    Wrapper for Keras models to make them scikit-learn compatible.
    """
    def __init__(self, build_fn, model_type: str, epochs: int = 100, **hyperparams):
        self.build_fn = build_fn
        self.model_type = model_type
        self.epochs = epochs
        self.hyperparams = hyperparams
        self.model = None
        self.log_dir = None

    def fit(self, X, y, validation_data=None, target_scaler=None, run_log_dir: Path = None, batch_size: int = 32, extra_callbacks: list = None):
        input_shape = (X.shape[1], X.shape[2])
        
        if self.model_type == 'cnn_lstm':
            self.model = self.build_fn(
                input_shape,
                conv_filters=self.hyperparams.get('conv_filters', 64),
                kernel_size=self.hyperparams.get('kernel_size', 3),
                n_units=self.hyperparams.get('n_units', 128),
                dropout=self.hyperparams.get('dropout', 0.2)
            )
        else:
            self.model = self.build_fn(
                input_shape,
                n_units=self.hyperparams.get('n_units', 128),
                dropout=self.hyperparams.get('dropout', 0.2),
                model_type=self.model_type
            )

        optimizer = Adam(learning_rate=self.hyperparams.get('learning_rate', 0.001))
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        if run_log_dir:
            self.log_dir = run_log_dir / self.model_type
            file_writer = tf.summary.create_file_writer(str(self.log_dir))
        else:
            file_writer = None

        callbacks = [StopIfNaN()]
        
        if validation_data is not None:
            early_stopping = EarlyStopping(
                monitor='val_loss', 
                patience=TRAINING_CONFIG["CV_PATIENCE"], 
                restore_best_weights=True, 
                min_delta=0.0001
            )
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5, 
                patience=5, 
                min_lr=1e-6, 
                verbose=0
            )
            callbacks.extend([early_stopping, reduce_lr])

        # Allow passing additional user callbacks (e.g., pruning callback)
        if extra_callbacks:
            callbacks.extend(extra_callbacks)

        else:
            reduce_lr = ReduceLROnPlateau(
                monitor='loss',
                factor=0.5, 
                patience=5, 
                min_lr=1e-6, 
                verbose=0
            )
            callbacks.append(reduce_lr)

        history = self.model.fit(
            X, y,
            epochs=self.epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=0
        )

        if file_writer:
            with file_writer.as_default():
                tf.summary.scalar('final_loss', history.history['loss'][-1], step=self.epochs)
                if 'val_loss' in history.history:
                    tf.summary.scalar('final_val_loss', history.history['val_loss'][-1], step=self.epochs)

        return self

    def predict(self, X):
        return self.model.predict(X, verbose=0)

# OPTIMIZED: Vectorized confidence interval calculation
def calculate_confidence_metrics(predictions: np.ndarray) -> dict:
    """
    Calculates improved confidence metrics from ensemble predictions using vectorized operations.
    """
    # Remove outliers using IQR (vectorized for all samples at once)
    q1 = np.percentile(predictions, 25, axis=1, keepdims=True)
    q3 = np.percentile(predictions, 75, axis=1, keepdims=True)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    
    # Mask outliers
    mask = (predictions >= lower_bound) & (predictions <= upper_bound)
    
    # Calculate metrics using masked arrays
    metrics = {}
    mean_preds = []
    std_preds = []
    ci_vals = []
    conf_scores = []
    ci_lower_q = []
    ci_upper_q = []
    
    for i in range(predictions.shape[0]):
        clean_preds = predictions[i][mask[i]]
        
        if len(clean_preds) < 2:
            clean_preds = predictions[i]
        
        mean_pred = np.mean(clean_preds)
        std_pred = np.std(clean_preds)
        
        # Quantile-based
        ci_lower_q.append(np.percentile(clean_preds, 10))
        ci_upper_q.append(np.percentile(clean_preds, 90))
        
        # Adaptive CI
        base_factor = 1.645
        agreement_factor = np.clip(1 / (1 + (std_pred / (mean_pred + 1e-9)) * 10), 0.3, 1.0)
        adaptive_ci = base_factor * std_pred * agreement_factor
        
        # Confidence score
        normalized_std = min(std_pred / (mean_pred + 1e-9), 0.1)
        confidence = (1 - (normalized_std / 0.1)) * 100
        
        mean_preds.append(mean_pred)
        std_preds.append(std_pred)
        ci_vals.append(adaptive_ci)
        conf_scores.append(confidence)
    
    return {
        'mean': np.array(mean_preds),
        'std': np.array(std_preds),
        'ci': np.array(ci_vals),
        'confidence_score': np.array(conf_scores),
        'ci_lower_quantile': np.array(ci_lower_q),
        'ci_upper_quantile': np.array(ci_upper_q)
    }

# OPTIMIZED: Memory-efficient sequence creation using stride_tricks
def create_sequences_optimized(data, seq_len):
    """
    Creates sequences using numpy stride tricks for memory efficiency.
    """
    if len(data) <= seq_len:
        logger.warning(f"Data length ({len(data)}) is less than or equal to sequence length ({seq_len}). Returning empty sequences.")
        # Return an empty array with the expected 3D shape for compatibility
        n_features = 1 if len(data.shape) == 1 else data.shape[1]
        return np.empty((0, seq_len, n_features))
    
    # Ensure data is 2D
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    n_samples = len(data) - seq_len
    n_features = data.shape[1]
    
    # Calculate shape and strides for windowing
    shape = (n_samples, seq_len, n_features)
    strides = (data.strides[0], data.strides[0], data.strides[1])
    
    # Create view using stride tricks (zero-copy)
    sequences = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
    
    # Return a copy to avoid stride issues with subsequent operations
    return sequences.copy()



# --- 6. ENHANCED TRAINING & EVALUATION ---

# NOTE: Older single-split objectives with pruning were removed to avoid duplicate
# definitions and runtime inconsistencies. The active and robust Optuna objectives
# are defined later in the file (the cross-validated ML objective and the DL
# objective which uses the KerasRegressorWrapper properly). Remove duplicates
# prevents accidental shadowing and ensures pruning/callbacks are applied via
# the wrapper's `extra_callbacks` argument when needed.



def create_ml_objective(X, y, model_name: str, n_splits: int = 3):
    """Creates an Optuna objective function for ML models."""
    def objective(trial: optuna.Trial) -> float:
        if model_name == 'lgbm':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 400),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 5, 25),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'random_state': 42, 'n_jobs': -1, 'verbose': -1
            }
            model = LGBMRegressor(**params)
        elif model_name == 'xgb':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 400),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'random_state': 42, 'n_jobs': -1
            }
            model = xgb.XGBRegressor(**params)
        else:
            return float('inf')

        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        for train_index, val_index in tscv.split(X):
            X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
            y_train_fold, y_val_fold = y[train_index], y[val_index]
            model.fit(X_train_fold, y_train_fold)
            preds = model.predict(X_val_fold)
            scores.append(mean_squared_error(y_val_fold, preds))
        
        return np.mean(scores)
    return objective

def create_dl_objective(X, y, model_type: str, n_splits: int = 3):
    """Creates an Optuna objective function for DL models."""
    def objective(trial: optuna.Trial) -> float:
        n_units = trial.suggest_int('n_units', 64, 256)
        dropout = trial.suggest_float('dropout', 0.1, 0.4)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        
        if model_type == 'cnn_lstm':
            params = {
                'conv_filters': trial.suggest_int('conv_filters', 32, 128),
                'kernel_size': trial.suggest_categorical('kernel_size', [3, 5]),
                'n_units': n_units, 'dropout': dropout, 'learning_rate': learning_rate
            }
            build_fn = create_cnn_lstm_model
        else:
            params = {'n_units': n_units, 'dropout': dropout, 'learning_rate': learning_rate}
            build_fn = create_dl_model

        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        # Use only the last split for speed during optimization
        train_index, val_index = list(tscv.split(X))[-1]
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        y_train_fold, y_val_fold = y[train_index], y[val_index]
        
        # Use fewer epochs for faster hyperparameter search
        model = KerasRegressorWrapper(build_fn=build_fn, model_type=model_type, epochs=15, **params)
        model.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold))
        
        preds = model.predict(X_val_fold)
        return mean_squared_error(y_val_fold, preds)
    return objective

def train_and_evaluate_for_horizon(horizon: int, run_log_dir: Path, sequence_length: int = TRAINING_CONFIG["SEQUENCE_LENGTH"], ticker: str = TRAINING_CONFIG["TICKER"], timeframe: str = TRAINING_CONFIG["TIMEFRAME"], training_mode: bool = False) -> None:
    """
    Trains, evaluates, and saves all models for a specific prediction horizon using rolling walk-forward validation.
    """
    
    tb_logger = TradingTensorBoard(run_log_dir, horizon)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    Path('sentiment').mkdir(parents=True, exist_ok=True)
    
    TerminalStyle.subheader("PHASE 2: DATA PREPROCESSING & FEATURE ENGINEERING")
    TerminalStyle.section_open("2A : FETCHING MARKET & SENTIMENT DATA")
    data = load_data(ticker, timeframe, limit=TRAINING_CONFIG["DATA_LIMIT"])
    data = fetch_and_add_sentiment(data, ticker.split('/')[0])
    TerminalStyle.section_close()
    
    TerminalStyle.section_open("2B : LOADING KOREAN MARKET DATA (BITHUMB)")
    data = fetch_bithumb_krw_data(data)
    TerminalStyle.section_close()
    
    TerminalStyle.section_open("2C : LOADING LEVERAGE DATA (OKX + BINANCE)")
    # Funding
    funding_avg, funding_sources = fetch_funding_rates_authenticated()
    data['funding_rate_avg'] = funding_avg
    data['funding_rate_8h_ma'] = data['funding_rate_avg'].rolling(8).mean()

    # Current OI
    oi_total, oi_okx, oi_sources = fetch_current_oi_authenticated()
    data['open_interest_total'] = oi_total
    data['open_interest_okx'] = oi_okx

    # Derived features
    data['oi_total_change_24h'] = data['open_interest_total'].pct_change(periods=24)
    data['oi_okx_ratio'] = data['open_interest_okx'] / (data['open_interest_total'] + 1e-8)
    data['oi_z_score_week'] = (
        data['open_interest_total'] - data['open_interest_total'].rolling(168).mean()
    ) / (data['open_interest_total'].rolling(168).std() + 1e-8)

    TerminalStyle.success(f"Leverage features added (funding from {funding_sources}, OI from {oi_sources})")
    TerminalStyle.section_close()

    TerminalStyle.section_open("2D : DETECTING INSTITUTIONAL ORDER BLOCKS (ICT STYLE)")
    ob_detector = UltimateOrderBlockDetector(
        lookback_candles=300,      # ~12 days on 1h — enough history, not ancient
        max_distance_pct=20.0,     # Ignore zones >20% away
        swing_length=7,
        min_impulse_pct=3.0
    )
    data = ob_detector.detect_and_add_features(data)

    data['funding_rate_avg'] = data['funding_rate_avg'].fillna(0)
    data['open_interest_total'] = data['open_interest_total'].fillna(0)
    TerminalStyle.section_close()

    # Initialize whale-related columns to 0.0 before feature engineering
    # This ensures they are always present, preventing 'missing column' warnings
    data['whale_market_dominance'] = 0.0
    data['whale_sell_intensity'] = 0.0
    data['whale_buy_intensity'] = 0.0
    data['whale_net_flow_momentum'] = 0.0
    data['exchange_inventory_momentum'] = 0.0
    data['whale_pressure_score'] = 0.0
    data['whale_net_flow_raw'] = 0.0
    data['whale_vol_6h'] = 0.0

    TerminalStyle.section_open("2E : FEATURE ENGINEERING")
    featured_data = feature_engineering(data)
    
    if not training_mode:
        tb_logger.log_market_context(featured_data)

    log_dir = run_log_dir / f'evaluation_{horizon}h'
    featured_data = detect_market_regime(featured_data, str(log_dir))
    TerminalStyle.section_close()

    TerminalStyle.training_pipeline_subheader("VERIFYING FEATURE CONSISTENCY")
    featured_data = enforce_feature_consistency(featured_data, debug=True)
    
    if ticker.split('/')[0] == 'XRP':
        TerminalStyle.subheader("PHASE 3: OMNISCIENT WHALE AUDIT (LAST 6 HOURS)")
        TerminalStyle.section_open("3A : SCANNING THE LEDGER")
        
        scanner = IntelligentWhaleScanner()
        whale_data = asyncio.run(scanner.scan_and_quantify(print_output=False))
        
        TerminalStyle.success("WHALE AUDIT COMPLETE")

        net_flow_xrp = whale_data['whale_net_flow']
        flow_direction = "INFLOW" if net_flow_xrp > 0 else "OUTFLOW" if net_flow_xrp < 0 else "Neutral"
        TerminalStyle.success(f"Detected: {whale_data['whale_buy_count']} BUYS | {whale_data['whale_sell_count']} SELLS | {whale_data['whale_transfer_count']} TRANSFERS")

        TerminalStyle.info(f"ⓘ Net Exchange Flow: {net_flow_xrp/1e6:+.2f}M XRP ({flow_direction})")
        TerminalStyle.info(f"ⓘ Time-Decay Pressure Score: {whale_data['whale_pressure_score']:,.2f}")

        quantified = scanner.get_quantified_features(whale_data)

        current_vol = featured_data['volume'].iloc[-1] + 1e-8
        featured_data['whale_market_dominance'] = whale_data['whale_total_volume'] / current_vol
        featured_data['whale_sell_intensity'] = whale_data['whale_sell_volume'] / current_vol
        featured_data['whale_buy_intensity'] = whale_data['whale_buy_volume'] / current_vol
        featured_data['whale_net_flow_momentum'] = whale_data['whale_net_flow'] / current_vol
        featured_data['exchange_inventory_momentum'] = -whale_data['whale_net_flow']
        
        featured_data['whale_pressure_score'] = quantified['whale_pressure_score']
        featured_data['whale_net_flow_raw'] = quantified['whale_net_flow_raw']
        featured_data['whale_vol_6h'] = quantified['whale_vol_6h']
        
        TerminalStyle.success("Whale intent features integrated.")
        TerminalStyle.section_close()
    else:
        TerminalStyle.warning(f"Whale tracking only available for XRP. Skipping for {ticker}.")

    featured_data['target'] = featured_data['close'].shift(-horizon)
    featured_data.dropna(inplace=True)
    
    y, X = featured_data[['target']], featured_data.drop(columns='target')
    
    # --- Rolling Walk-Forward Validation ---
    TerminalStyle.subheader("Phase 4: Rolling Walk-Forward Validation")
    n_splits = 5
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    all_fold_metrics = []
    all_y_test_actual = []
    all_final_predictions = []
    all_test_indices = []
    all_confidence_scores = []
    all_directional_correct = []
    all_dynamic_correct = []
    all_significant_moves = []
    
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        feature_scaler = RobustScaler()
        target_scaler = MinMaxScaler()
        
        X_train_scaled = feature_scaler.fit_transform(X_train)
        y_train_scaled = target_scaler.fit_transform(y_train)
        X_test_scaled = feature_scaler.transform(X_test)

        feature_columns = X_train.columns.tolist()

        # Create sequences
        X_train_seq = create_sequences_optimized(X_train_scaled, sequence_length)
        y_train_seq_cv = y_train_scaled[sequence_length:]
        X_test_seq = create_sequences_optimized(X_test_scaled, sequence_length)
        y_test_seq = y_test.values[sequence_length:]
        y_test_actual = y_test.values[sequence_length:]
        
        X_train_tab = X_train_scaled[sequence_length:]
        X_test_tab = X_test_scaled[sequence_length:]
        X_train_tab_df = pd.DataFrame(X_train_tab, columns=feature_columns)
        X_test_tab_df = pd.DataFrame(X_test_tab, columns=feature_columns)
        
        model_names = ['gru', 'lstm', 'cnn_lstm', 'lgbm', 'xgb']
        base_models = {}
        
        letter_code = ord('B')
        for name in model_names:
            best_params = {}
            if TRAINING_CONFIG["OPTIMIZE_HYPERPARAMETERS"] and OPTUNA_AVAILABLE:
                TerminalStyle.section_open(f"4{chr(letter_code)} : OPTIMIZING {name.upper()} WITH OPTUNA (FOLD {fold+1}/{n_splits})")
                n_trials = TRAINING_CONFIG["OPTUNA_TRIALS"]
                study = get_global_optuna_study(name, horizon, ticker)

                if name in ['lgbm', 'xgb']:
                    objective = create_ml_objective(X_train_tab_df, y_train_seq_cv.ravel(), name)
                else: # dl models
                    objective = create_dl_objective(X_train_seq, y_train_seq_cv, name)

                study.optimize(objective, n_trials=n_trials)
                best_params = study.best_params
                TerminalStyle.success(f"Optimization complete. Best params for {name.upper()}: {best_params}")
                TerminalStyle.section_close()
                letter_code += 1
            else:
                TerminalStyle.info(f"Skipping optimization for {name.upper()}, using default parameters.")

            TerminalStyle.training_subheader(f"TRAINING {name.upper()} (FOLD {fold+1}/{n_splits})")
            
            if name in ['gru', 'lstm', 'cnn_lstm']:
                epochs = TRAINING_CONFIG["CV_EPOCHS_DL"]
                hyperparams = get_default_hyperparameters(name)
                hyperparams.update(best_params)
                
                if name == 'gru':
                    model_instance = KerasRegressorWrapper(create_dl_model, model_type='gru', epochs=epochs, **hyperparams)
                elif name == 'lstm':
                    model_instance = KerasRegressorWrapper(create_dl_model, model_type='lstm', epochs=epochs, **hyperparams)
                elif name == 'cnn_lstm':
                    model_instance = KerasRegressorWrapper(create_cnn_lstm_model, model_type='cnn_lstm', epochs=epochs, **hyperparams)
                
                base_models[name] = model_instance.fit(X_train_seq, y_train_seq_cv, validation_data=(X_test_seq, y_test_seq), target_scaler=target_scaler, run_log_dir=run_log_dir)

            else: # lgbm, xgb
                params = TRAINING_CONFIG["MODEL_PARAMS"].get(name, {})
                params.update(best_params)
                
                if name == 'lgbm':
                    base_models[name] = LGBMRegressor(**params).fit(X_train_tab_df, y_train_seq_cv.ravel())
                elif name == 'xgb':
                    base_models[name] = xgb.XGBRegressor(**params).fit(X_train_tab_df, y_train_seq_cv.ravel())
            
            TerminalStyle.success(f"{name.upper()} trained for fold {fold+1}")
            letter_code += 1

        meta_features_test = np.column_stack([
            base_models[name].predict(X_test_seq if name in ['gru', 'lstm', 'cnn_lstm'] else X_test_tab_df)
            for name in base_models
        ])
        
        # We need to train the meta-model on the training data of this fold
        meta_features_train_fold = np.column_stack([
            base_models[name].predict(X_train_seq if name in ['gru', 'lstm', 'cnn_lstm'] else X_train_tab_df)
            for name in base_models
        ])
        meta_model_fold = Ridge().fit(meta_features_train_fold, y_train_seq_cv.ravel())

        final_predictions_scaled = meta_model_fold.predict(meta_features_test).reshape(-1, 1)
        final_predictions = target_scaler.inverse_transform(final_predictions_scaled).flatten()
        
        if len(y_test_actual) > 0 and len(final_predictions) > 0:
            current_prices_for_test = X_test['close'].values[sequence_length:]
            
            # ADDED: Calculate confidence scores for the current fold
            base_predictions_unscaled = target_scaler.inverse_transform(meta_features_test)
            uncertainty_metrics_fold = calculate_confidence_metrics(base_predictions_unscaled)
            confidence_scores_fold = uncertainty_metrics_fold['confidence_score']

            metrics_returned = tb_logger.log_prediction_quality(y_test_actual, final_predictions, current_prices_for_test, confidence_scores_fold, fold=fold)
            all_fold_metrics.append(metrics_returned)
            all_y_test_actual.append(y_test_actual)
            all_final_predictions.append(final_predictions)
            all_test_indices.append(y_test.index[sequence_length:])
            
            # ADDED: Collect confidence and correctness masks
            all_confidence_scores.extend(metrics_returned['confidence_scores'])
            all_directional_correct.extend(metrics_returned['directional_correct_mask'])
            all_dynamic_correct.extend(metrics_returned['dynamic_correct_mask'])
            all_significant_moves.extend(metrics_returned['significant_move_mask'])

    # --- Backtest Visualization ---
    if all_y_test_actual:
        TerminalStyle.subheader("Phase 5: Backtest Visualization")
        y_true_all = np.concatenate(all_y_test_actual).flatten()
        y_pred_all = np.concatenate(all_final_predictions).flatten()
        timestamps_all = pd.to_datetime(np.concatenate(all_test_indices))
        tb_logger.log_backtest_chart(timestamps_all, y_true_all, y_pred_all)
        TerminalStyle.success(f"Backtest plot for {horizon}h horizon logged to TensorBoard.")

    # --- Aggregate and Log CV Results ---
    TerminalStyle.subheader("Phase 6: Aggregated Walk-Forward Validation Results")
    if all_fold_metrics:
        avg_rmse = np.mean([m['rmse'] for m in all_fold_metrics])
        avg_mae = np.mean([m['mae'] for m in all_fold_metrics])
        avg_dir_acc = np.mean([m['directional_accuracy'] for m in all_fold_metrics])
        avg_dyn_acc = np.mean([m['dynamic_accuracy'] for m in all_fold_metrics])
        
        tb_logger.log_final_cv_summary({'rmse': avg_rmse, 'mae': avg_mae, 'directional_accuracy': avg_dir_acc, 'dynamic_accuracy': avg_dyn_acc})
        
        # TerminalStyle.metric("Average RMSE", f"{avg_rmse:.4f}")
        # TerminalStyle.metric("Average MAE", f"{avg_mae:.4f}")
        
        TerminalStyle.success(f"✓ Average Directional Accuracy (Settlement Bias): {avg_dir_acc:.1f}%")
        TerminalStyle.success(f"✓ Average Dynamic Accuracy (Significant Moves > {TRAINING_CONFIG['DIRECTIONAL_MOVE_THRESHOLD_PCT']}%): {avg_dyn_acc:.1f}%")

        TerminalStyle.success(f"Cross-validation metrics logged to TensorBoard dir: {log_dir}")

    # --- Final Model Training on All Data ---
    TerminalStyle.subheader("Phase 7: Final Model Training on All Data")
    
    feature_scaler = RobustScaler()
    target_scaler = MinMaxScaler()
    
    X_scaled = feature_scaler.fit_transform(X)
    y_scaled = target_scaler.fit_transform(y)

    feature_columns = X.columns.tolist()

    X_seq = create_sequences_optimized(X_scaled, sequence_length)
    y_seq = y_scaled[sequence_length:]
    X_tab_df = pd.DataFrame(X_scaled[sequence_length:], columns=feature_columns)
    
    model_names = ['gru', 'lstm', 'cnn_lstm', 'lgbm', 'xgb']
    base_models = {}
    
    for i, name in enumerate(model_names, 1):
        TerminalStyle.info(f"Training final {name.upper()} model...")
        # Using hyperparameters found during CV or defaults
        best_params = {}
        if TRAINING_CONFIG["OPTIMIZE_HYPERPARAMETERS"] and OPTUNA_AVAILABLE:
             study = get_global_optuna_study(name, horizon, ticker)
             try:
                best_params = study.best_params
             except:
                best_params = {}

        if name in ['gru', 'lstm', 'cnn_lstm']:
            epochs = TRAINING_CONFIG["DL_EPOCHS"]
            hyperparams = get_default_hyperparameters(name)
            hyperparams.update(best_params)
            
            if name == 'gru':
                model_instance = KerasRegressorWrapper(create_dl_model, model_type='gru', epochs=epochs, **hyperparams)
            elif name == 'lstm':
                model_instance = KerasRegressorWrapper(create_dl_model, model_type='lstm', epochs=epochs, **hyperparams)
            elif name == 'cnn_lstm':
                model_instance = KerasRegressorWrapper(create_cnn_lstm_model, model_type='cnn_lstm', epochs=epochs, **hyperparams)
            
            base_models[name] = model_instance.fit(X_seq, y_seq, run_log_dir=run_log_dir)
        else:
            params = TRAINING_CONFIG["MODEL_PARAMS"].get(name, {})
            params.update(best_params)
            if name == 'lgbm':
                base_models[name] = LGBMRegressor(**params).fit(X_tab_df, y_seq.ravel())
            elif name == 'xgb':
                base_models[name] = xgb.XGBRegressor(**params).fit(X_tab_df, y_seq.ravel())
        
        joblib.dump(base_models[name], MODEL_DIR / f'base_{name}_{horizon}h.pkl')
        TerminalStyle.success(f"Final {name.upper()} model trained and saved ({i}/{len(model_names)})")
        
    TerminalStyle.subheader("Phase 8: Final Meta-Model Training")
    
    meta_features_full = np.column_stack([
        base_models[name].predict(X_seq if name in ['gru', 'lstm', 'cnn_lstm'] else X_tab_df)
        for name in base_models
    ])
    
    meta_model = Ridge().fit(meta_features_full, y_seq.ravel())
    
    # Save all models and scalers
    TerminalStyle.subheader("PHASE 9: SAVING MODELS")
    joblib.dump(meta_model, MODEL_DIR / f'meta_model_{horizon}h.pkl')
    joblib.dump(feature_columns, MODEL_DIR / f'feature_columns_{horizon}h.pkl')
    joblib.dump(feature_scaler, MODEL_DIR / f'features_scaler_{horizon}h.pkl')
    joblib.dump(target_scaler, MODEL_DIR / f'target_scaler_{horizon}h.pkl')
    
    TerminalStyle.success("All models and scalers saved")
    
    if not training_mode and hasattr(base_models.get('lgbm'), 'feature_importances_'):
        tb_logger.log_feature_importance(feature_columns, base_models['lgbm'].feature_importances_)
    
    tb_logger.close()



def update_forecast_plot(plot_path: Path, history_df: pd.DataFrame, run_index: int, ticker: str, timeframe: str):
    """Updates a forecast plot with the latest actual price data."""
    import matplotlib.pyplot as plt
    if not plot_path.exists():
        return

    unique_timestamps = sorted(history_df['timestamp'].unique(), reverse=True)
    if len(unique_timestamps) <= run_index:
        TerminalStyle.info(f"Not enough historical runs to verify plot {plot_path.name} (run index {run_index}).")
        return

    run_timestamp = unique_timestamps[run_index]
    run_preds = history_df[history_df['timestamp'] == run_timestamp].copy()

    if run_preds.empty:
        TerminalStyle.warning(f"Could not find predictions for run at {run_timestamp} for {plot_path.name}. Skipping verification.")
        return

    TerminalStyle.info(f"Verifying forecast from {run_timestamp.strftime('%Y-%m-%d %H:%M')} and updating {plot_path.name}...")

    run_preds['horizon_hours'] = run_preds['horizon'].str.replace('h', '').astype(int)
    max_h = run_preds['horizon_hours'].max()
    
    start_time = pd.to_datetime(run_preds['timestamp'].iloc[0])
    end_time = pd.Timestamp.now(tz='UTC')
    hours_to_fetch = int((end_time - start_time).total_seconds() / 3600) + max_h + 2

    actuals_df = load_data(ticker, timeframe, limit=hours_to_fetch)
    verification_actuals = actuals_df[(actuals_df.index >= start_time) & (actuals_df.index <= end_time)]

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(15, 8))
    colors = ['#8A2BE2', '#FF69B4', '#40E0D0', '#FFA500']

    start_price = run_preds['current_price'].iloc[0]

    for i, row in enumerate(run_preds.itertuples()):
        future_time = start_time + pd.Timedelta(hours=row.horizon_hours)
        color = colors[i % len(colors)]
        ax.plot([start_time, future_time], [start_price, row.predicted_price], linestyle='--', color=color, alpha=0.7, linewidth=2)
        ax.plot(future_time, row.predicted_price, 'x', color=color, markersize=12, markeredgewidth=3, label=f'{row.horizon} Prediction', zorder=5)

    ax.plot(start_time, start_price, 'o', color='white', markersize=12, zorder=11, label='Original Forecast Price')

    if not verification_actuals.empty:
        ax.plot(verification_actuals.index, verification_actuals['close'], color='white', linewidth=2.5, label='Actual Price Path', zorder=10, alpha=0.9)

    ax.legend(loc='best', fontsize=10)
    ax.set_title(f'{ticker} Forecast vs Actual (Prediction from {start_time.strftime("%Y-%m-%d %H:%M")})', fontsize=16, color='white', fontweight='bold')
    ax.set_xlabel('Date', fontsize=12, color='white')
    ax.set_ylabel('Price (USDT)', fontsize=12, color='white')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#555555', alpha=0.3)
    
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    TerminalStyle.success(f"Verification chart updated and saved to {plot_path.name}")


def verify_and_rotate_plots():
    """Handles the rotation and verification of historical forecast plots at the start of a run."""
    prediction_dir = Path('predictions')
    
    f1 = prediction_dir / 'future_forecast_1.png'
    f2 = prediction_dir / 'future_forecast_2.png'
    f3 = prediction_dir / 'future_forecast_3.png'

    f3_exists = False
    if f2.exists():
        TerminalStyle.info("Rotating future_forecast_2.png to future_forecast_3.png")
        if f3.exists():
            f3.unlink()
        f2.rename(f3)
        f3_exists = True
    
    f2_exists = False
    if f1.exists():
        TerminalStyle.info("Rotating future_forecast_1.png to future_forecast_2.png")
        f1.rename(f2)
        f2_exists = True
    else:
        TerminalStyle.info("No previous forecast (future_forecast_1.png) to verify.")

    if not (f2_exists or f3_exists):
        return

    try:
        history_df = pd.read_csv(prediction_dir / 'predictions.csv', parse_dates=['timestamp'])
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'], utc=True)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        TerminalStyle.warning("Could not read prediction history. Skipping verification.")
        return

    if history_df.empty:
        TerminalStyle.warning("Prediction history is empty. Skipping verification.")
        return

    ticker = os.getenv("TICKER", "XRP/USDT")
    timeframe = os.getenv("TIMEFRAME", "1h")

    if f2_exists:
        update_forecast_plot(f2, history_df, run_index=1, ticker=ticker, timeframe=timeframe)
    if f3_exists:
        update_forecast_plot(f3, history_df, run_index=2, ticker=ticker, timeframe=timeframe)

def plot_future_forecast(predictions_df, ticker):
    """
    Plots ALL future prediction horizons on a single chart with a dark theme.
    """
    import matplotlib.pyplot as plt
    if predictions_df.empty:
        return

    # Use the latest run by finding the most recent timestamp
    latest_timestamp = predictions_df['timestamp'].max()
    latest_preds = predictions_df[predictions_df['timestamp'] == latest_timestamp].copy()

    if latest_preds.empty:
        return

    # --- Chart Styling ---
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Define the color palette - one color per horizon
    colors = ['#8A2BE2', '#FF69B4', '#40E0D0', '#FFA500']  # Purple, Pink, Turquoise, Orange

    # --- Data Preparation ---
    latest_preds['horizon_hours'] = latest_preds['horizon'].str.replace('h', '').astype(int)
    latest_preds = latest_preds.sort_values('horizon_hours')
    
    # Ensure we have all horizons before plotting
    if len(latest_preds) == 0:
        logger.warning("No predictions found for the latest timestamp")
        return
    
    start_time = pd.to_datetime(latest_preds['timestamp'].iloc[0])
    start_price = latest_preds['current_price'].iloc[0]

    # Plot current price as a starting point
    ax.plot(start_time, start_price, 'o', color='white', markersize=12, label='Current Price', zorder=10)

    # --- Plot ALL Horizons ---
    for i, row in enumerate(latest_preds.itertuples()):
        future_time = start_time + pd.Timedelta(hours=row.horizon_hours)
        color = colors[i % len(colors)]
        
        # Dashed line from current to predicted
        ax.plot([start_time, future_time], [start_price, row.predicted_price], 
                linestyle='--', color=color, alpha=0.8, linewidth=2)
        
        # Predicted price point
        ax.plot(future_time, row.predicted_price, 'o', color=color, markersize=10, 
                label=f'{row.horizon} Prediction', zorder=5)
        
        # Confidence interval shaded area
        # if hasattr(row, 'confidence_interval_lower') and hasattr(row, 'confidence_interval_upper'):
        #     # Create polygon for shaded confidence area
        #     times = [start_time, future_time, future_time, start_time]
        #     prices = [start_price, row.confidence_interval_upper, row.confidence_interval_lower, start_price]
        #     ax.fill(times, prices, color=color, alpha=0.15)

    # --- Formatting ---
    ax.legend(loc='best', fontsize=10)
    ax.set_title(f'{ticker} Future Price Forecast', fontsize=18, color='white', fontweight='bold')
    ax.set_xlabel('Date', fontsize=12, color='white')
    ax.set_ylabel('Price (USDT)', fontsize=12, color='white')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='#555555', alpha=0.3)
    
    # Format x-axis to show dates nicely
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    plt.xticks(rotation=45, ha='right')  
    plt.tight_layout()

        # --- Saving ---
    prediction_dir = Path('predictions')
    base_name = "future_forecast"
    plot_path = prediction_dir / f'{base_name}_1.png'  # Always save new plot as _1
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    TerminalStyle.success(f"Future forecast plot saved to {plot_path}")
    
# --- 7. FUTURE PREDICTION WITH IMPROVED UNCERTAINTY ---
def make_future_predictions(horizons: List[int], sequence_length: int = TRAINING_CONFIG["SEQUENCE_LENGTH"], ticker: str = TRAINING_CONFIG["TICKER"], timeframe: str = TRAINING_CONFIG["TIMEFRAME"]) -> None:
    """
    Generates and saves future price predictions for multiple horizons with IMPROVED confidence intervals.
    """
    TerminalStyle.subheader("PHASE 7: GENERATING FUTURE PREDICTIONS")
    
    Path('sentiment').mkdir(parents=True, exist_ok=True)
    Path('predictions').mkdir(parents=True, exist_ok=True)

    # --- Model Caching ---
    TerminalStyle.section_open("7A : CACHING MODELS")
    model_cache = {}
    for horizon in horizons:
        try:
            model_cache[horizon] = {
                'base_models': {name: joblib.load(MODEL_DIR / f'base_{name}_{horizon}h.pkl') for name in ['gru', 'lstm', 'cnn_lstm', 'lgbm', 'xgb']},
                'meta_model': joblib.load(MODEL_DIR / f'meta_model_{horizon}h.pkl'),
                'feature_columns': joblib.load(MODEL_DIR / f'feature_columns_{horizon}h.pkl'),
                'features_scaler': joblib.load(MODEL_DIR / f'features_scaler_{horizon}h.pkl'),
                'target_scaler': joblib.load(MODEL_DIR / f'target_scaler_{horizon}h.pkl')
            }
            TerminalStyle.success(f"Models for {horizon}h cached")
        except FileNotFoundError as e:
            logger.error(f"Models for {horizon}h not found. Please train first")
            return
    
    TerminalStyle.success("All models cached successfully")
    TerminalStyle.section_close()

    TerminalStyle.section_open("7B : LOADING LATEST MARKET DATA")
    data = load_data(ticker, timeframe, limit=sequence_length + 200)
    data = fetch_and_add_sentiment(data, ticker.split('/')[0])
    TerminalStyle.section_close()
    
    TerminalStyle.section_open("7C : LOADING KOREAN MARKET DATA (BITHUMB)")
    data = fetch_bithumb_krw_data(data)
    TerminalStyle.section_close()
    
    TerminalStyle.section_open("7D : LOADING LEVERAGE DATA (OKX + BINANCE)")
    # Funding
    funding_avg, funding_sources = fetch_funding_rates_authenticated()
    data['funding_rate_avg'] = funding_avg
    data['funding_rate_8h_ma'] = data['funding_rate_avg'].rolling(8).mean()

    # Current OI
    oi_total, oi_okx, oi_sources = fetch_current_oi_authenticated()
    data['open_interest_total'] = oi_total
    data['open_interest_okx'] = oi_okx

    # Derived features
    data['oi_total_change_24h'] = data['open_interest_total'].pct_change(periods=24)
    data['oi_okx_ratio'] = data['open_interest_okx'] / (data['open_interest_total'] + 1e-8)
    data['oi_z_score_week'] = (
        data['open_interest_total'] - data['open_interest_total'].rolling(168).mean()
    ) / (data['open_interest_total'].rolling(168).std() + 1e-8)

    TerminalStyle.success(f"Leverage features added (funding from {funding_sources}, OI from {oi_sources})")
    TerminalStyle.section_close()

    TerminalStyle.section_open("7E : DETECTING INSTITUTIONAL ORDER BLOCKS (ICT STYLE)")
    ob_detector = UltimateOrderBlockDetector(
        lookback_candles=300,      # ~12 days on 1h — enough history, not ancient
        max_distance_pct=20.0,     # Ignore zones >20% away
        swing_length=7,
        min_impulse_pct=3.0
    )
    data = ob_detector.detect_and_add_features(data)

    data['funding_rate_avg'] = data['funding_rate_avg'].fillna(0)
    data['open_interest_total'] = data['open_interest_total'].fillna(0)
    TerminalStyle.section_close()
    
    # Initialize whale-related columns to 0.0 before feature engineering
    # This ensures they are always present, preventing 'missing column' warnings
    data['whale_market_dominance'] = 0.0
    data['whale_sell_intensity'] = 0.0
    data['whale_buy_intensity'] = 0.0
    data['whale_net_flow_momentum'] = 0.0
    data['exchange_inventory_momentum'] = 0.0
    data['whale_pressure_score'] = 0.0
    data['whale_net_flow_raw'] = 0.0
    data['whale_vol_6h'] = 0.0

    if ticker.split('/')[0] == 'XRP':
        TerminalStyle.subheader("PHASE 8: OMNISCIENT WHALE AUDIT (LAST 6 HOURS)")
        TerminalStyle.section_open("8A : SCANNING THE LEDGER")
        
        scanner = IntelligentWhaleScanner()
        whale_data = asyncio.run(scanner.scan_and_quantify(print_output=False))
        
        TerminalStyle.success("WHALE AUDIT COMPLETE")
        TerminalStyle.section_close()

        net_flow_xrp = whale_data['whale_net_flow']
        flow_direction = "INFLOW" if net_flow_xrp > 0 else "OUTFLOW" if net_flow_xrp < 0 else "Neutral"
        TerminalStyle.success(f"Detected: {whale_data['whale_buy_count']} BUYS | {whale_data['whale_sell_count']} SELLS | {whale_data['whale_transfer_count']} TRANSFERS")

        TerminalStyle.info(f"ⓘ Net Exchange Flow: {net_flow_xrp/1e6:+.2f}M XRP ({flow_direction})")
        TerminalStyle.info(f"ⓘ Time-Decay Pressure Score: {whale_data['whale_pressure_score']:,.2f}")

        quantified = scanner.get_quantified_features(whale_data)

        current_vol = data['volume'].iloc[-1] + 1e-8
        data['whale_market_dominance'] = whale_data['whale_total_volume'] / current_vol
        data['whale_sell_intensity'] = whale_data['whale_sell_volume'] / current_vol
        data['whale_buy_intensity'] = whale_data['whale_buy_volume'] / current_vol
        data['whale_net_flow_momentum'] = whale_data['whale_net_flow'] / current_vol
        data['exchange_inventory_momentum'] = -whale_data['whale_net_flow']
        
        data['whale_pressure_score'] = quantified['whale_pressure_score']
        data['whale_net_flow_raw'] = quantified['whale_net_flow_raw']
        data['whale_vol_6h'] = quantified['whale_vol_6h']
        
        TerminalStyle.success("Whale intent features integrated.")
        TerminalStyle.section_close()
    else:
        TerminalStyle.warning(f"Whale tracking only available for XRP. Skipping for {ticker}.")

    TerminalStyle.section_open("7F : FEATURE ENGINEERING")
    featured_data = feature_engineering(data)
    featured_data = detect_market_regime(featured_data)
    TerminalStyle.section_close()

    # Enforce a consistent feature set to prevent errors
    TerminalStyle.subheader("PHASE 9: VERIFYING FEATURE CONSISTENCY")
    featured_data = enforce_feature_consistency(featured_data, debug=True)

    all_predictions = []
    current_price = featured_data['close'].iloc[-1]
    
    TerminalStyle.info(f"Current {ticker} Price: {current_price:.4f} USDT")

    timestamp = pd.Timestamp.now(tz='UTC')
    for horizon in horizons:
        try:
            if horizon not in model_cache:
                logger.warning(f"Models for {horizon}h not in cache. Skipping")
                continue

            cached_models = model_cache[horizon]
            base_models = cached_models['base_models']
            meta_model = cached_models['meta_model']
            feature_columns = cached_models['feature_columns']
            feature_scaler = cached_models['features_scaler']
            target_scaler = cached_models['target_scaler']

            last_sequence_unscaled = featured_data.tail(sequence_length).drop(columns=['target'], errors='ignore')
            last_sequence_scaled = feature_scaler.transform(last_sequence_unscaled)
            X_future_seq = np.array([last_sequence_scaled])
            X_future_tab_df = pd.DataFrame([last_sequence_scaled[-1]], columns=feature_columns)
            
            # Get predictions from all base models
            base_predictions_scaled = np.column_stack([
                base_models[name].predict(X_future_seq if name in ['gru', 'lstm', 'cnn_lstm'] else X_future_tab_df)
                for name in base_models
            ])

            # Inverse transform base predictions before calculating confidence
            base_predictions_unscaled = target_scaler.inverse_transform(base_predictions_scaled)
            
            # Calculate uncertainty using IMPROVED adaptive intervals on the UN-SCALED predictions
            uncertainty_metrics = calculate_confidence_metrics(base_predictions_unscaled)
            
            # Meta-model prediction
            future_price_scaled = meta_model.predict(base_predictions_scaled.reshape(1, -1))
            future_price = target_scaler.inverse_transform(future_price_scaled.reshape(-1, 1)).flatten()[0]

            # Transform ADAPTIVE confidence intervals
            ci_lower_scaled = future_price_scaled[0] - uncertainty_metrics['ci'][0]
            ci_upper_scaled = future_price_scaled[0] + uncertainty_metrics['ci'][0]
            ci_lower = target_scaler.inverse_transform([[ci_lower_scaled]])[0][0]
            ci_upper = target_scaler.inverse_transform([[ci_upper_scaled]])[0][0]
            
            # ADDED: Simple momentum/whale bias for final predictions
            # Assumes df is the full feature dataframe (with latest row) and whale_data from scanner
            
            # Get raw net flow
            latest_whale_net_flow = 0
            if 'whale_data' in locals() and whale_data:
                latest_whale_net_flow = whale_data.get('whale_net_flow', 0)

            latest_rsi = featured_data['rsi'].iloc[-1] if 'rsi' in featured_data.columns else 50
            
            whale_threshold = TRAINING_CONFIG["WHALE_THRESHOLD"]
            
            bias_factor = 0.0
            # Bullish bias: RSI > 50 OR Whale Net Flow > Threshold
            if latest_rsi > 50 or latest_whale_net_flow > whale_threshold:
                bias_factor = 0.01  # +1% upward bias
            # Bearish bias: RSI < 50 OR Whale Net Flow < -Threshold
            elif latest_rsi < 50 or latest_whale_net_flow < -whale_threshold:
                bias_factor = -0.01  # -1% downward bias

            predicted_price = future_price * (1 + bias_factor) # Apply bias to future_price
            
            # Optionally apply same bias to CI bounds
            confidence_interval_lower = ci_lower * (1 + bias_factor)
            confidence_interval_upper = ci_upper * (1 + bias_factor)
            
            # Also get quantile-based intervals for comparison
            quantile_lower = target_scaler.inverse_transform([[uncertainty_metrics['ci_lower_quantile'][0]]])[0][0]
            quantile_upper = target_scaler.inverse_transform([[uncertainty_metrics['ci_upper_quantile'][0]]])[0][0]
            
            confidence_score = uncertainty_metrics['confidence_score'][0]
            ci_width_pct = ((confidence_interval_upper - confidence_interval_lower) / predicted_price) * 100
            
            price_change = ((predicted_price - current_price) / current_price) * 100
            
            # Display with confidence
            TerminalStyle.prediction_box(
                current_price, 
                predicted_price, 
                price_change, 
                horizon,
                confidence_interval_lower,
                confidence_interval_upper,
                confidence_score
            )
            
            # Confidence interpretation with CI width info
            if confidence_score >= 75:
                TerminalStyle.success(f"High confidence prediction - Low market uncertainty detected")
                TerminalStyle.info(f"✓ Adaptive CI narrowed to ±{ci_width_pct:.1f}% (vs Quantile: ±{((quantile_upper-quantile_lower)/future_price)*100:.1f}%)")
            elif confidence_score >= 50:
                TerminalStyle.warning(f"Medium confidence - Moderate market uncertainty")
                TerminalStyle.info(f"CI width: ±{ci_width_pct:.1f}% of predicted price")
            else:
                TerminalStyle.warning(f"Low confidence - High market uncertainty detected. Use with caution!")
                TerminalStyle.info(f"Wide CI: ±{ci_width_pct:.1f}% due to model disagreement")
            
            TerminalStyle.info(f"Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            
            all_predictions.append({
                'horizon': f'{horizon}h',
                'current_price': current_price,
                'predicted_price': future_price,
                'expected_change_pct': price_change,
                'confidence_interval_lower': ci_lower,
                'confidence_interval_upper': ci_upper,
                'prediction_std': uncertainty_metrics['std'][0],
                'confidence_score': confidence_score,
                'ci_width_pct': ci_width_pct,
                'timestamp': timestamp
            })
            
        except Exception as e:
            logger.error(f"Prediction error for {horizon}h: {e}")
    
    if all_predictions:
        pred_df = pd.DataFrame(all_predictions)
        log_path = Path('predictions') / 'predictions.csv'
        pred_df.to_csv(log_path, mode='a', header=not log_path.exists(), index=False)
        
        TerminalStyle.subheader("PREDICTION SUMMARY")
        
        for pred in all_predictions:
            color = TerminalColors.GREEN if pred['expected_change_pct'] > 0 else TerminalColors.RED
            symbol = "▲" if pred['expected_change_pct'] > 0 else "▼"
            conf_color = TerminalColors.GREEN if pred['confidence_score'] >= 75 else TerminalColors.ORANGE if pred['confidence_score'] >= 50 else TerminalColors.RED
            
            print(f"  {TerminalColors.BOLD}{pred['horizon']:>4}{TerminalColors.ENDC} │ "
                  f"Predicted: {TerminalColors.BOLD}{pred['predicted_price']:.4f}{TerminalColors.ENDC} │ "
                  f"Range: {TerminalColors.DIM}{pred['confidence_interval_lower']:.4f}-{pred['confidence_interval_upper']:.4f}{TerminalColors.ENDC} │ "
                  f"Change: {color}{symbol} {abs(pred['expected_change_pct']):.2f}%{TerminalColors.ENDC} │ "
                  f"Confidence: {conf_color}{pred['confidence_score']:.0f}%{TerminalColors.ENDC} │ "
                  f"CI: {TerminalColors.DIM}±{pred['ci_width_pct']:.1f}%{TerminalColors.ENDC}")
        
        TerminalStyle.info(f"Predictions saved to: {log_path}")
        TerminalStyle.info(f"Using {CI_METHOD.upper()} confidence interval method")

        # Generate and save the plot - THIS SHOULD ONLY BE CALLED ONCE
        plot_future_forecast(pred_df, ticker)
    else:
        TerminalStyle.warning("No predictions were generated")



# --- 8. MAIN EXECUTION ---
if __name__ == '__main__':
    import shutil
    from datetime import datetime

    TerminalStyle.header(
        title="CLAIRVOYANT v4 || XRP FORECASTER",
        subtitle1="On-Chain Whale Tracker • Institutional Order Blocks",
        subtitle2="Global Market Premiums • Leverage & Funding Signals",
        subtitle3="Multi-Modal Sentiment & Market Analysis",
        author="-by K.Bourn-"
    )
   
    TerminalStyle.subheader("PHASE 1: INITIALIZING THE ENVIRONMENT")

    TerminalStyle.section_open("1A : FETCHING MARKETS DATA")
    TerminalStyle.info("Exchanges initialized at global scope.")
    TerminalStyle.section_close()

    TerminalStyle.section_open("1B : CONFIRMING CONFIGURATION SETTINGS")
    TerminalStyle.success(f"Confidence Interval Method: {CI_METHOD.upper()}")
    TerminalStyle.success(f"Prediction Horizons: {', '.join([f'{h}h' for h in TRAINING_CONFIG['PREDICTION_HORIZONS']])}")
    TerminalStyle.success(f"Hyperparameter Optimization: {'ENABLED' if TRAINING_CONFIG['OPTIMIZE_HYPERPARAMETERS'] else 'DISABLED'}")
    TerminalStyle.success("On-Chain Metrics: ENABLED (XRP Ledger)")
    TerminalStyle.success(f"Optuna Hyperparameters : {'ENABLED' if TRAINING_CONFIG['OPTIMIZE_HYPERPARAMETERS'] else 'DISABLED'}")
    TerminalStyle.section_close()

    TerminalStyle.section_open("1C : PRESERVING TRAINING HISTORY")
    log_root = Path('logs')
    log_root.mkdir(exist_ok=True)
    run_dirs = sorted([d for d in log_root.iterdir() if d.is_dir() and d.name.startswith('run_')])
    TerminalStyle.success(f"Keeping all {len(run_dirs)} historical runs forever")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_run_log_dir = log_root / f"run_{run_timestamp}"
    current_run_log_dir.mkdir()
    TerminalStyle.success(f"New run → {current_run_log_dir.name}")
    TerminalStyle.section_close()

    TerminalStyle.section_open("1D : VERIFYING PREVIOUS FORECAST")
    verify_and_rotate_plots()
    TerminalStyle.section_close()

    parser = argparse.ArgumentParser(description='Clairvoyant v4 - XRP Price Forecaster')
    parser.add_argument('--train', action='store_true', help='Run the training pipeline.')
    parser.add_argument('--predict', action='store_true', help='Run the prediction pipeline.')
    parser.add_argument('--training-mode', action='store_true', help='Run the training pipeline in a continuous loop.')
    args = parser.parse_args()

    # If no flags are specified, run both training and prediction
    if not args.train and not args.predict and not args.training_mode:
        args.train = True
        args.predict = True
    elif args.training_mode:
        args.predict = False


    if args.training_mode:
        try:
            while True:
                TerminalStyle.subheader("CONTINUOUS TRAINING MODE ACTIVATED")
                for h in TRAINING_CONFIG["PREDICTION_HORIZONS"]:
                    train_and_evaluate_for_horizon(horizon=h, run_log_dir=current_run_log_dir, training_mode=True)
                TerminalStyle.success("All horizons trained. Restarting training loop...")
        except KeyboardInterrupt:
            TerminalStyle.warning("CONTINUOUS TRAINING STOPPED BY USER.")

    elif args.train:
        # Train models for each horizon
        for h in TRAINING_CONFIG["PREDICTION_HORIZONS"]:
            TerminalStyle.training_pipeline_subheader(f"TRAINING PIPELINE: {h}H PREDICTION HORIZON")
            train_and_evaluate_for_horizon(horizon=h, run_log_dir=current_run_log_dir)
    
    if args.predict:
        # Generate future predictions and the forecast plot
        make_future_predictions(horizons=TRAINING_CONFIG["PREDICTION_HORIZONS"])

    TerminalStyle.subheader("CLAIRVOYANT RUN COMPLETE")
    
    TerminalStyle.info("All requested operations completed successfully.")
    TerminalStyle.info(f"Models are saved in: {MODEL_DIR}")
    TerminalStyle.info(f"Predictions and charts are in: {Path('predictions')}")
    TerminalStyle.info(f"Check TensorBoard logs for detailed training metrics: tensorboard --logdir={log_root}")
    
    print()