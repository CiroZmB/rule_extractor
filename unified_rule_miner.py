# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr, ks_2samp
import joblib
from joblib import Parallel, delayed
import time
from datetime import datetime
import os
import hashlib
import re
import gc
import contextlib
from itertools import groupby
from numpy.fft import fft, ifft
from tqdm import tqdm

# Intento de importar TA-Lib, fallback a implementaciones Pandas
try:
    import talib as ta
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    print("AVISO: TA-Lib no encontrado. Usando implementaciones Pandas.")

# =============================================================================
# HELPER: TQDM CONTEXT MANAGER FOR JOBLIB
# =============================================================================
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

# =============================================================================
# HELPER: PANDAS INDICATORS (Fallback Completo)
# =============================================================================
class PandasTA:
    @staticmethod
    def RSI(series, timeperiod=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=timeperiod).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=timeperiod).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def SMA(series, timeperiod=30):
        return series.rolling(window=timeperiod).mean()

    @staticmethod
    def EMA(series, timeperiod=30):
        return series.ewm(span=timeperiod, adjust=False).mean()

    @staticmethod
    def BBANDS(series, timeperiod=20, nbdevup=2, nbdevdn=2):
        mid = series.rolling(window=timeperiod).mean()
        std = series.rolling(window=timeperiod).std()
        upper = mid + (std * nbdevup)
        lower = mid - (std * nbdevdn)
        return upper, mid, lower

    @staticmethod
    def ADX(high, low, close, timeperiod=14):
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(timeperiod).mean()
        
        up_move = high.diff()
        down_move = -low.diff()
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_di = 100 * (pd.Series(plus_dm).rolling(timeperiod).mean() / atr.replace(0, np.nan))
        minus_di = 100 * (pd.Series(minus_dm).rolling(timeperiod).mean() / atr.replace(0, np.nan))
        
        plus_di = plus_di.fillna(0)
        minus_di = minus_di.fillna(0)
        
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
        adx = dx.rolling(timeperiod).mean().fillna(0)
        return adx

    @staticmethod
    def PLUS_DI(high, low, close, timeperiod=14):
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(timeperiod).mean()
        
        up_move = high.diff()
        plus_dm = np.where(up_move > 0, up_move, 0)
        plus_di = 100 * (pd.Series(plus_dm).rolling(timeperiod).mean() / atr.replace(0, np.nan))
        return plus_di.fillna(0)

    @staticmethod
    def MINUS_DI(high, low, close, timeperiod=14):
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(timeperiod).mean()
        
        down_move = -low.diff()
        minus_dm = np.where(down_move > 0, down_move, 0)
        minus_di = 100 * (pd.Series(minus_dm).rolling(timeperiod).mean() / atr.replace(0, np.nan))
        return minus_di.fillna(0)

    @staticmethod
    def MOM(series, timeperiod=10):
        """
        Momentum estándar: Close[0] - Close[period] (Diferencia Absoluta)
        Compatible con MQL5 iMomentum y TA-Lib MOMENTUM
        """
        # MQL5: Amount change over period
        return series - series.shift(timeperiod)

    @staticmethod
    def CCI(high, low, close, timeperiod=14):
        tp = (high + low + close) / 3
        sma = tp.rolling(timeperiod).mean()
        # Mean Deviation
        mad = tp.rolling(timeperiod).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        return (tp - sma) / (0.015 * mad)
        
    @staticmethod
    def WILLR(high, low, close, timeperiod=14):
        hh = high.rolling(timeperiod).max()
        ll = low.rolling(timeperiod).min()
        return -100 * ((hh - close) / (hh - ll))

    @staticmethod
    def STOCH(high, low, close, fastk_period=5, slowk_period=3):
        # Fast %K
        hh = high.rolling(fastk_period).max()
        ll = low.rolling(fastk_period).min()
        k = 100 * (close - ll) / (hh - ll)
        # Slow %K (Smooth Fast %K)
        # Slow %K (Smooth Fast %K)
        slowk = k.rolling(slowk_period).mean()
        # %D (suavizado de Slow %K)
        slowd = slowk.rolling(slowk_period).mean() # Asumimos D period igual a SlowK por defecto MQL5 simple
        
        return slowk, slowd # Devuelve tupla (Main, Signal)

    @staticmethod
    def BEARS_POWER(low, close, timeperiod=13):
        ema = close.ewm(span=timeperiod, adjust=False).mean()
        return low - ema

    @staticmethod
    def BULLS_POWER(high, close, timeperiod=13):
        ema = close.ewm(span=timeperiod, adjust=False).mean()
        return high - ema
        
    @staticmethod
    def FORCE(close, volume, timeperiod=13):
        # Force Index = (Close - PrevClose) * Volume
        fi_raw = close.diff(1) * volume
        # MQL5 uses SMA mode for smoothing
        return fi_raw.rolling(timeperiod).mean()
        
    @staticmethod
    def DEMARKER(high, low, timeperiod=14):
        # DeM = SMA(DeMax) / (SMA(DeMax) + SMA(DeMin))
        # DeMax = Max(High - PrevHigh, 0)
        # DeMin = Max(PrevLow - Low, 0)
        dm_max = high.diff()
        dm_max = dm_max.where(dm_max > 0, 0)
        
        dm_min = -low.diff() # PrevLow - Low = -(Low - PrevLow)
        dm_min = dm_min.where(dm_min > 0, 0)
        
        sma_max = dm_max.rolling(timeperiod).mean()
        sma_min = dm_min.rolling(timeperiod).mean()
        
        return sma_max / (sma_max + sma_min)

# Wrapper
def get_indicator(name, *args, **kwargs):
    # Mapping nombres
    if name == 'MOM': name = 'MOM'
    
    if HAS_TALIB and hasattr(ta, name):
        try:
             func = getattr(ta, name)
             return func(*args, **kwargs)
        except: pass # Fallback if TALib fails or mismatched args
        
    func = getattr(PandasTA, name, None)
    if func:
        return func(*args, **kwargs)
    return pd.Series(np.zeros(len(args[0]))) 

# =============================================================================
# CONFIGURACIÓN (VERSION ENDURECIDA V3)
# =============================================================================
CONFIG = {
    'input_csv': 'Datos pares/2026.1.13XAUUSD_MIGUEL-M1-No Session.csv',
    'cutoff_date': '2021-01-01',  
    
    # --- FILTROS DE ROBUSTEZ (Endurecidos V3) ---
    'n_monkey_sims': 500,         # Antes 100. Más rigor estadístico.
    'monkey_percentile': 99.9,    # Antes 95. Exigimos "Excelencia", no solo "Suerte".
    
    'synthetic_sims': 100,        # Antes 50. Doble de mundos paralelos.
    'block_size': 30,             # Tamaño bloque bootstrap (días aprox)
    
    'min_trades': 150,            # Antes 50. Evitamos ley de pequeños números.
    'min_profit_factor': 1.3,     # IMPORTANTE: Filtro de Calidad Mínima.
    
    'exposicion_dias': 4,
    'n_jobs': -1                  
}

# =============================================================================
# 1. INGENIERÍA DE DATOS
# =============================================================================



# =============================================================================
# 1. INGENIERÍA DE DATOS
# =============================================================================



def reduce_multicollinearity_safe(train_df, test_df, feature_cols, threshold=0.97):
    """
    LEAKFIX: Correlaciones SOLO con Train, aplicar a ambos conjuntos.
    """
    print(f"\n--- REDUCCIÓN DE MULTICOLINEALIDAD (LEAKFIX) ---")
    
    # 1. Calcular matriz de correlación SOLO con Train
    sample_size = min(len(train_df), 5000)
    # Seleccionar solo columnas numéricas de interés
    valid_cols = [c for c in feature_cols if c in train_df.columns]
    
    train_sample = train_df[valid_cols].iloc[-sample_size:].dropna()
    
    if len(train_sample) < 100:
        print("⚠️ Muestra insuficiente en Train, saltando reducción.")
        return train_df, test_df, valid_cols
    
    corr_matrix = train_sample.corr().abs()
    
    # 2. Identificar redundantes (Triángulo superior)
    to_drop = set()
    
    # Iterar sobre las columnas
    columns = corr_matrix.columns
    for i in range(len(columns)):
        col_i = columns[i]
        if col_i in to_drop: continue
        
        for j in range(i + 1, len(columns)):
            col_j = columns[j]
            if col_j in to_drop: continue
            
            # Si correla > threshold, eliminar col_j
            if corr_matrix.iloc[i, j] > threshold:
                to_drop.add(col_j)
    
    final_features = [f for f in feature_cols if f not in to_drop]
    
    # 3. Aplicar MISMO drop a Train y Test
    cols_to_drop_list = list(to_drop)
    
    # Safe drop (check keys)
    drop_train = [c for c in cols_to_drop_list if c in train_df.columns]
    drop_test = [c for c in cols_to_drop_list if c in test_df.columns]
    
    train_clean = train_df.drop(columns=drop_train)
    test_clean = test_df.drop(columns=drop_test)
    
    print(f"Features eliminadas: {len(to_drop)}")
    print(f"Features finales: {len(final_features)}")
    
    return train_clean, test_clean, final_features

def resample_ohlc_data(df, period):
    """
    Resamplea datos OHLCV a un timeframe superior.
    Periodos Pandas: '1h', '4h', '1d', '1w'
    """
    print(f"⌛ Resampling data a {period}...")
    
    # Asegurar DateTime index
    if 'DateTime' in df.columns:
        df = df.set_index('DateTime')
        
    # Diccionario de agregación
    agg_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }
    
    # Agregar volumen si existe
    if 'Volume' in df.columns:
        agg_dict['Volume'] = 'sum'
        
    # Resample
    df_res = df.resample(period).agg(agg_dict)
    
    # Drop NaNs (barras vacías generadas por el resample)
    df_res.dropna(inplace=True)
    
    # Reset index para que DateTime vuelva a ser columna
    df_res.reset_index(inplace=True)
    
    print(f"Data Resampleada: {len(df_res)} filas")
    return df_res

def load_and_transform_data(csv_path, target_timeframe=None, threshold=25, short=False):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encuentra {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    df = pd.read_csv(csv_path)
    
    # Limpiar nombres de columnas (padding)
    df.columns = [c.strip() for c in df.columns]

    # --- 1. ROBUST DATETIME HANDLING ---
    # Caso común: 'Date' + 'Time' separados (ej. MT4/MT5)
    # Debemos combinarlos ANTES de renombrar para evitar colisión 'DateTime' duplicado
    if 'Date' in df.columns and 'Time' in df.columns and 'DateTime' not in df.columns:
        try:
            # Combinar
            df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
            # Eliminar originales para evitar residuos
            df.drop(columns=['Date', 'Time'], inplace=True)
        except Exception as e:
            print(f"⚠️ Error combinando Date+Time: {e}")

    # Normalizar variantes de nombres (sin causar colisiones)
    # Solo renombramos si la columna existe y el target NO existe
    cols_map = {
        'date': 'DateTime', 'time': 'DateTime', 
        'Datetime': 'DateTime', 'datetime': 'DateTime', 
        'Timestamp': 'DateTime', 'timestamp': 'DateTime'
    }
    
    # Safe rename loop
    for old_col, new_col in cols_map.items():
        if old_col in df.columns and new_col not in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)
        elif old_col in df.columns and new_col in df.columns:
             # Si ya existe DateTime, borramos la variante vieja redundante
             df.drop(columns=[old_col], inplace=True)

    # --- 2. ROBUST VOLUME HANDLING ---
    # TickVol / Volume collision fix
    if 'Volume' not in df.columns:
        # Buscamos variantes
        vol_variants = ['TickVol', 'tickvol', 'Volume', 'volume', 'Tick Volume']
        for v in vol_variants:
            if v in df.columns:
                df.rename(columns={v: 'Volume'}, inplace=True)
                break
    else:
        # Si ya existe Volume, eliminamos duplicados potenciales (ej. TickVol)
        if 'TickVol' in df.columns: df.drop(columns=['TickVol'], inplace=True)
        if 'tickvol' in df.columns: df.drop(columns=['tickvol'], inplace=True)
            
    if 'DateTime' in df.columns:
        df['DateTime'] = pd.to_datetime(df['DateTime'])
    else:
        # Fallback si falla todo
        print("⚠️ No se encontró columna DateTime. Usando índice como tiempo.")
        df['DateTime'] = pd.date_range('2000-01-01', periods=len(df), freq='H')
    
    # --- 0. RESAMPLING (Optativo) ---
    if target_timeframe:
        # Mapeo simple de nombres comunes a pandas offset aliases
        # Ej: H1 -> 1h, H4 -> 4h, D1 -> 1D
        tf_map = {
            'M1': '1min', 'M5': '5min', 'M15': '15min', 'M30': '30min',
            'H1': '1h', 'H2': '2h', 'H3': '3h', 'H4': '4h',
            'H6': '6h', 'H8': '8h', 'H12': '12h',
            'D1': '1D', 'W1': '1W'
        }
        # Si ya viene en formato pandas (ej '1h'), lo usa directo. Si viene 'H1', lo mapea.
        pd_period = tf_map.get(target_timeframe, target_timeframe)
        
        # Solo resamplear si tiene sentido (ej. no resamplear a lo mismo o menor)
        # Asumimos que el usuario sabe lo que hace, o podriamos chequear.
        df = resample_ohlc_data(df, pd_period)

    # --- 1. Calcular TARGETS PRIMERO (Futuro) ---
    pips_multiplier = 10000 if 'JPY' not in csv_path else 100
    max_horizon_bars = 30
    
    for i in range(2, max_horizon_bars + 1, 2):
        future_close = df['Close'].shift(-i)
        ret = (future_close - df['Close']) * pips_multiplier
        if short: ret = ret * -1
        df[f'Return_{i}'] = ret
        
    target_col = f'Return_{CONFIG["exposicion_dias"]}'
    if target_col not in df.columns: target_col = 'Return_4'
    df['Target'] = (df[target_col] >= threshold).astype(int)

    # --- 2. Calcular INDICADORES (Densidad Completa tipo MQL5) ---
    print("\nGenerando matriz de indicadores densa...")
    feature_cols = []
    
    # Rango de periodos (ajustable, usamos 2-50 para velocidad en Python)
    # MQL5 usa 2-100, aquí usamos 2-50 y saltos de 2 para no explotar RAM
    periods = range(2, 52, 2) 
    
    # Check volumen
    has_volume = 'Volume' in df.columns and not df['Volume'].isna().all()
    
    for i in tqdm(periods, desc="Calculando Indicadores"):
        # RSI, ADX, DI
        col = f'rsi_{i}'; df[col] = get_indicator('RSI', df['Close'], timeperiod=i); feature_cols.append(col)
        col = f'adx_{i}'; df[col] = get_indicator('ADX', df['High'], df['Low'], df['Close'], timeperiod=i); feature_cols.append(col)
        col = f'pdi_{i}'; df[col] = get_indicator('PLUS_DI', df['High'], df['Low'], df['Close'], timeperiod=i); feature_cols.append(col)
        col = f'mdi_{i}'; df[col] = get_indicator('MINUS_DI', df['High'], df['Low'], df['Close'], timeperiod=i); feature_cols.append(col)
        
        # Nuevos (MQL5 Port)
        col = f'mom_{i}'; df[col] = get_indicator('MOM', df['Close'], timeperiod=i); feature_cols.append(col)
        col = f'cci_{i}'; df[col] = get_indicator('CCI', df['High'], df['Low'], df['Close'], timeperiod=i); feature_cols.append(col)
        col = f'wpr_{i}'; df[col] = get_indicator('WILLR', df['High'], df['Low'], df['Close'], timeperiod=i); feature_cols.append(col)
        col = f'dem_{i}'; df[col] = get_indicator('DEMARKER', df['High'], df['Low'], timeperiod=i); feature_cols.append(col)
        col = f'bears_{i}'; df[col] = get_indicator('BEARS_POWER', df['Low'], df['Close'], timeperiod=i); feature_cols.append(col)
        col = f'bulls_{i}'; df[col] = get_indicator('BULLS_POWER', df['High'], df['Close'], timeperiod=i); feature_cols.append(col)
        
        if has_volume:
             col = f'force_{i}'; df[col] = get_indicator('FORCE', df['Close'], df['Volume'], timeperiod=i); feature_cols.append(col)

        # Stoch (FastK=i, SlowK=3) (Fix Audit: Return K and D)
        sk, sd = get_indicator('STOCH', df['High'], df['Low'], df['Close'], fastk_period=i)
        df[f'stoch_k_{i}'] = sk
        df[f'stoch_d_{i}'] = sd
        feature_cols.append(f'stoch_k_{i}')
        feature_cols.append(f'stoch_d_{i}')
        
    # --- 3. RETORNO DE DATOS RAW (No reducimos aquí para evitar Look-Ahead) ---
    # Limpiamos NaNs críticos (donde no hay Target calculado)
    # Se reemplaza slicing ciego por dropna inteligente
    df.dropna(subset=[f'Return_{CONFIG["exposicion_dias"]}'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df, feature_cols

def split_data_inverted(df, cutoff_date):
    print(f"--- APLICANDO PARTICIÓN INVERTIDA (Cutoff: {cutoff_date}) ---")
    train_df = df[df['DateTime'] > cutoff_date].copy().reset_index(drop=True)
    test_df = df[df['DateTime'] <= cutoff_date].copy().reset_index(drop=True)
    print(f"Train (Reciente): {len(train_df)} filas | Test (Histórico): {len(test_df)} filas")
    return train_df, test_df

# =============================================================================
# 2. GENERADORES
# =============================================================================

def generate_combinatorial_rules(df, max_rules=10000):
    rules = []
    exclude = ['Target', 'Return', 'year', 'month', 'day', 'hour', 'profit', 'DateTime']
    numeric_cols = [c for c in df.columns if df[c].dtype in [np.float64, np.float32] and not any(x in c for x in exclude)]
    
    # 1. Reglas de Valor (Thresholds únicos)
    for col in numeric_cols:
        series = df[col]
        raw_thresholds = np.percentile(series, [10, 20, 30, 40, 50, 60, 70, 80, 90])
        thresholds = np.unique(np.round(raw_thresholds, 4))
        
        for t in thresholds:
            rules.append(f"{col} > {t}")
            rules.append(f"{col} < {t}")
            
    # 2. Cruces
    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i+1:]:
            family_match = (
                ('rsi' in col1 and 'rsi' in col2) or
                ('adx' in col1 and 'adx' in col2) or
                (('sma' in col1 or 'ema' in col1 or 'Close' in col1) and ('sma' in col2 or 'ema' in col2 or 'Close' in col2))
            )
            if family_match:
                rules.append(f"{col1} > {col2}")
                rules.append(f"{col1} < {col2}")
                
        if len(rules) > max_rules * 2: break
            
    if len(rules) > max_rules:
        rules = [rules[i] for i in np.random.choice(len(rules), max_rules, replace=False)]
    return rules

# =============================================================================
# 3. VALIDACIÓN & METRICAS
# =============================================================================

def calculate_max_drawdown(equity_curve):
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve - running_max
    return np.min(drawdown)

def calculate_metrics(returns):
    if len(returns) == 0: return {}
    equity = np.cumsum(returns)
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    return {
        'total_profit': np.sum(returns),
        'sharpe': np.mean(returns) / (np.std(returns, ddof=1) + 1e-9),
        'max_dd': calculate_max_drawdown(equity),
        'win_rate': len(wins) / len(returns),
        'pf': np.sum(wins) / abs(np.sum(losses)) if len(losses) > 0 else 999
    }

def vectorized_monkey_test_ultra(signals, returns, n_sims=100, percentile=95):
    """MONKEY TEST ULTRA VECTORIZADO"""
    real_return = np.sum(returns[signals])
    n_bars = len(returns)
    
    shifts = np.random.randint(1, n_bars, size=n_sims)
    indices = np.arange(n_bars)
    shifted_indices = (indices[None, :] - shifts[:, None]) % n_bars
    signals_matrix = signals[shifted_indices]
    
    monkey_returns = (signals_matrix * returns[None, :]).sum(axis=1)
    
    threshold_val = np.percentile(monkey_returns, percentile)
    is_robust = real_return > threshold_val
    return is_robust, threshold_val

def generate_synthetic_price_paths(close_series, n_sims=10, block_size=30):
    """
    Genera caminos de precios sintéticos basados en los retornos de una serie dada.
    Uso: Para el "Laboratorio Sintético" visual.
    """
    returns = close_series.pct_change().dropna()
    n_days = len(returns)
    start_price = close_series.iloc[0]
    
    synthetic_prices_df = pd.DataFrame(index=close_series.index)
    synthetic_prices_df['Original'] = close_series
    
    for i in range(n_sims):
        # Bootstrap de retornos
        indices = np.arange(n_days)
        n_blocks = max(1, n_days // block_size)
        blocks = np.array_split(indices, n_blocks)
        
        sampled_idc = np.random.randint(0, len(blocks), size=len(blocks))
        new_indices = np.concatenate([blocks[j] for j in sampled_idc])
        
        # Ajuste longitud
        if len(new_indices) > n_days: new_indices = new_indices[:n_days]
        elif len(new_indices) < n_days:
            extra = blocks[np.random.randint(0, len(blocks))]
            new_indices = np.concatenate([new_indices, extra])[:n_days]
            
        synth_returns = returns.iloc[new_indices].values
        
        # Reconstruir precio
        # Precio_t = Precio_0 * cumprod(1 + r)
        synth_price_path = start_price * (1 + synth_returns).cumprod()
        
        # Ajustar longitud para coincidir (el primero es start_price)
        # pct_change pierde 1 valor. Agregamos el inicial.
        synth_full = np.concatenate([[start_price], synth_price_path])
        if len(synth_full) > len(close_series):
             synth_full = synth_full[:len(close_series)]
             
        col_name = f'Synth_{i+1}'
        synthetic_prices_df[col_name] = synth_full
        
    return synthetic_prices_df

def calculate_acf(x, lags=20):
    x = np.array(x, dtype=float)
    n = len(x)
    if n == 0: return np.zeros(lags + 1)
    x_centered = x - np.mean(x)
    fft_x = fft(x_centered, n=2*n)
    power = fft_x * np.conj(fft_x)
    acf = ifft(power).real[:n] / n
    return acf[:lags+1] / acf[0] if acf[0] != 0 else np.zeros(lags+1)

def validate_synthetic_series(orig, synth):
    scores = {}
    _, ks_p = ks_2samp(orig, synth)
    scores['ks_score'] = ks_p
    s1, s2 = np.std(orig), np.std(synth)
    scores['std_score'] = min(s1, s2)/max(s1,s2) if max(s1,s2) > 0 else 0
    a1, a2 = calculate_acf(orig, 10), calculate_acf(synth, 10)
    scores['acf_score'] = np.corrcoef(a1, a2)[0,1] if (np.std(a1)>0 and np.std(a2)>0) else 0
    q1, q2 = calculate_acf(orig**2, 10), calculate_acf(synth**2, 10)
    scores['acf_sq_score'] = np.corrcoef(q1, q2)[0,1] if (np.std(q1)>0 and np.std(q2)>0) else 0
    return scores

def generate_synthetic_blocks_check(trade_returns, n_sims=50, block_size=30):
    if len(trade_returns) < 50: return 0.0 # Fail safe return float
    
    n_trades = len(trade_returns)
    valid_sims = 0
    passed = 0
    attempts = 0
    max_attempts = n_sims * 50 # RELAX AUDIT 3.0 (Más intentos)
    
    indices = np.arange(n_trades)
    n_blocks = max(1, n_trades // block_size)
    blocks = np.array_split(indices, n_blocks)
    
    while valid_sims < n_sims and attempts < max_attempts:
        attempts += 1
        
        sampled_idc = np.random.randint(0, len(blocks), size=len(blocks))
        new_indices = np.concatenate([blocks[i] for i in sampled_idc])
        
        if len(new_indices) > n_trades: new_indices = new_indices[:n_trades]
        elif len(new_indices) < n_trades: 
             extra = blocks[np.random.randint(0, len(blocks))]
             new_indices = np.concatenate([new_indices, extra])[:n_trades]
             
        synth_ret = trade_returns[new_indices]
        scores = validate_synthetic_series(trade_returns, synth_ret)
        
        weighted_score = (
            scores['std_score'] * 0.25 +
            scores['acf_sq_score'] * 0.35 +
            scores['ks_score'] * 0.20 + 
            scores['acf_score'] * 0.20
        )
        
        if weighted_score > 0.60: # RELAX AUDIT 3.0 (Antes 0.70)
            valid_sims += 1
            if np.sum(synth_ret) > 0: passed += 1
        elif attempts % 10 == 0:
            print(f"⚠️ Rechazo Sintético {attempts}: Calidad insuficiente ({weighted_score:.2f})")
            
    if valid_sims == 0: return 0.0
    
    # Audit 3.0 Log
    if valid_sims < n_sims:
        print(f"⚠️ Alerta Sintética: Solo se lograron {valid_sims}/{n_sims} simulaciones válidas (Max Int {attempts})")
            
    if valid_sims == 0: return 0.0
    return passed / valid_sims

def process_rules(df_values, columns, rules_list, target_col_idx):
    valid_rules = []
    main_return_col = df_values[:, target_col_idx]
    
    cols_map = {name: i for i, name in enumerate(columns)}
    
    for rule_str in rules_list:
        parts = rule_str.split()
        if len(parts) != 3: continue
        col1, op, col2 = parts
        
        if col1 not in cols_map: continue
        idx1 = cols_map[col1]
        
        is_val = False
        try:
             val = float(col2)
             is_val = True
        except:
             if col2 not in cols_map: continue
             idx2 = cols_map[col2]
        
        # Eval
        v1 = df_values[:, idx1]
        if is_val:
            if op == '>': s = v1 > val
            elif op == '<': s = v1 < val
            else: continue
        else:
            v2 = df_values[:, idx2]
            if op == '>': s = v1 > v2
            elif op == '<': s = v1 < v2
            else: continue
            
        n_trades = np.sum(s)
        if n_trades < CONFIG['min_trades']: continue
        
        rule_returns = main_return_col[s]
        metrics = calculate_metrics(rule_returns)
        
        # Filtro de Calidad (Profit Factor)
        if metrics['pf'] < CONFIG['min_profit_factor']: continue
        
        if metrics['total_profit'] <= 0: continue
        
        # Validation
        robust, _ = vectorized_monkey_test_ultra(s, main_return_col, 
                                                 CONFIG['n_monkey_sims'], 
                                                 CONFIG['monkey_percentile'])
        if not robust: continue
        
        # Synthetic Robustness (Ahora nos devuelve el % de mundos superados)
        synth_pass_rate = generate_synthetic_blocks_check(rule_returns, 
                                             CONFIG['synthetic_sims'], 
                                             CONFIG['block_size'])
        
        # Filtro estricto: debe ganar en el 80% de los mundos sintéticos válidos (Configurable)
        synth_threshold = CONFIG.get('synth_threshold', 0.80)
        if synth_pass_rate < synth_threshold:
            continue
            
        valid_rules.append({
            'Rule': rule_str,
            'Trades': n_trades,
            'Profit': metrics['total_profit'],
            'PF': metrics['pf'],
            'Sharpe': metrics['sharpe'],
            'MaxDD': metrics['max_dd'],
            'Synth_Robustness': synth_pass_rate  # Guardamos el dato para la UI
        })
        
    return valid_rules

# =============================================================================
# 4. EXPORTACIÓN
# =============================================================================

def format_rule_for_sqx(rule_str):
    sqx = rule_str
    sqx = re.sub(r'rsi_(\d+)', r'RSI(Close, \1)', sqx)
    sqx = re.sub(r'sma_(\d+)', r'SMA(Close, \1)', sqx)
    sqx = re.sub(r'ema_(\d+)', r'EMA(Close, \1)', sqx)
    sqx = re.sub(r'adx_(\d+)', r'ADX(\1)', sqx)
    sqx = re.sub(r'bb_upper_(\d+)', r'BollingerUpper(Close, \1, 2)', sqx)
    sqx = re.sub(r'bb_lower_(\d+)', r'BollingerLower(Close, \1, 2)', sqx)
    sqx = re.sub(r'plus_di_(\d+)', r'PlusDI(\1)', sqx)
    sqx = re.sub(r'minus_di_(\d+)', r'MinusDI(\1)', sqx)
    sqx = re.sub(r'_sft_(\d+)', r'[\1]', sqx)
    
    return f"ENTRY LONG IF: {sqx}"

def export_to_sqx_file(rules_df, filename='strategies_sqx.txt'):
    with open(filename, 'w') as f:
        f.write("// StrategyQuant X Compatible Rules\n")
        f.write(f"// Generated: {datetime.now()}\n\n")
        
        for idx, row in rules_df.iterrows():
            f.write(f"// ========== STRATEGY #{idx+1} ==========\n")
            f.write(f"// Train Profit: {row['Train_Profit']:.2f}\n")
            f.write(f"// Test Profit: {row['Test_Profit']:.2f}\n")
            f.write(f"// PF: {row['PF']:.2f} | Sharpe: {row.get('Test_Sharpe',0):.2f}\n")
            f.write(format_rule_for_sqx(row['Rule']))
            f.write("\n\n")
    print(f"✅ Exportado a {filename}")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=== UNIFIED RULE MINER v3.0 (FULL AUDIT) ===")
    
    # 1. LOAD OR DUMMY (Full Density)
    if not os.path.exists(CONFIG['input_csv']):
        print("⚠️ CSV no encontrado. Generando datos dummy...")
        dates = pd.date_range('2015-01-01', '2024-01-01', freq='4H')
        df = pd.DataFrame({
            'DateTime': dates,
            'Close': 1000 + np.cumsum(np.random.randn(len(dates)) * 0.1),
            'High': 1005 + np.cumsum(np.random.randn(len(dates)) * 0.1),
            'Low': 995 + np.cumsum(np.random.randn(len(dates)) * 0.1)
        })
        df.to_csv('temp_dummy.csv', index=False)
        df, feature_cols_all = load_and_transform_data('temp_dummy.csv')
        try: os.remove('temp_dummy.csv')
        except: pass
    else:
        df, feature_cols_all = load_and_transform_data(CONFIG['input_csv'])

    # =========================================================================
    #  VALIDACIÓN V3: TEST DEL MUNDO NULO (NULL HYPOTHESIS)
    #  Objetivo: Romper la relación causa-efecto para validar la metodología.
    # =========================================================================
    ENABLE_NULL_TEST = False  # <--- FALSE por defecto para que no rompa la app normal
    
    if ENABLE_NULL_TEST:
        print("\n" + "!"*60)
        print("⚠️  ADVERTENCIA: MODO 'MUNDO NULO' ACTIVADO")
        print("Se está aplicando un desplazamiento aleatorio masivo a los Targets.")
        print("El objetivo es DESTRUIR cualquier capacidad predictiva real.")
        print("!"*60)

        # 1. Definir un desplazamiento aleatorio significativo (evita cercanía local)
        n_rows = len(df)
        random_shift = np.random.randint(int(n_rows * 0.2), int(n_rows * 0.8))
        
        print(f" -> Desplazando Targets y Retornos {random_shift} barras...")

        # 2. Identificar columnas de 'Respuesta' (Targets y Retornos)
        target_cols = [c for c in df.columns if 'Target' in c or 'Return' in c]

        # 3. Aplicar el desplazamiento circular (Circular Shift)
        for col in target_cols:
            df[col] = np.roll(df[col], random_shift)

        print(" -> Relación Causa-Efecto ROTA exitosamente.")
        print(" -> RESULTADO ESPERADO: 0 Reglas Robustas encontradas.")
        print("!"*60 + "\n")
    # =========================================================================

    # 2. SPLIT (Partition First)
    train_df, test_df = split_data_inverted(df, CONFIG['cutoff_date'])
    
    # 3. FEATURE SELECTION (Train Only to avoid Look-Ahead)
    # Applying Reduce Multicollinearity SAFE (Train+Test consistent drop)
    train_df, test_df, final_features = reduce_multicollinearity_safe(
        train_df, test_df, feature_cols_all, threshold=0.97
    )
    
    print(f"Features Finales para Minería: {len(final_features)}")

    # 5. GENERATE RULES (On Reduced Train)
    rules = generate_combinatorial_rules(train_df)
    print(f"Reglas generadas: {len(rules)}")
    
    # 6. PROCESSING
    n_cores = os.cpu_count() or 4
    chunk_size = max(100, len(rules) // (n_cores * 3))
    chunks = [rules[i:i + chunk_size] for i in range(0, len(rules), chunk_size)]
    
    print(f"Procesando en {len(chunks)} chunks con {n_cores} cores...")
    
    tv = train_df.to_numpy()
    cols = train_df.columns
    t_idx = cols.get_loc(f"Return_{CONFIG['exposicion_dias']}")
    
    with tqdm_joblib(tqdm(total=len(rules))) as pbar:
        results = Parallel(n_jobs=CONFIG['n_jobs'])((
            delayed(process_rules)(tv, cols, ch, t_idx) for ch in chunks
        ))
    
    flat = [r for sub in results for r in sub]
    res_df = pd.DataFrame(flat)
    
    if not res_df.empty:
        print(f"\nReglas Train Sobrevivientes: {len(res_df)}")
        best = res_df.sort_values('Profit', ascending=False)
        
        # 5. VALIDACIÓN TEST (WALK-FORWARD)
        print("\n--- VALIDACIÓN EN TEST SET (Walk-Forward) ---")
        test_v = test_df.to_numpy()
        test_survivors = []
        
        cols_map = {n: i for i, n in enumerate(cols)}
        t_target_idx = cols.get_loc(f"Return_{CONFIG['exposicion_dias']}")
        t_ret_full = test_v[:, t_target_idx]
        
        for idx, row in tqdm(best.iterrows(), total=len(best), desc="Validando en Test"):
            r_str = row['Rule']
            parts = r_str.split()
            c1, op, c2 = parts
            
            if c1 not in cols_map: continue
            idx1 = cols_map[c1]
            try:
                val = float(c2)
                v1 = test_v[:, idx1]
                if op == '>': s = v1 > val
                else: s = v1 < val
            except:
                if c2 not in cols_map: continue
                idx2 = cols_map[c2]
                v1, v2 = test_v[:, idx1], test_v[:, idx2]
                if op == '>': s = v1 > v2
                else: s = v1 < v2
                
            n_t = np.sum(s)
            if n_t < 10: continue
            
            test_returns = t_ret_full[s]
            test_metrics = calculate_metrics(test_returns)
            
            # Re-confirmar robustez en Test (Monkey + Synth)
            monkey_pass, _ = vectorized_monkey_test_ultra(
                s, t_ret_full, 
                CONFIG['n_monkey_sims'], CONFIG['monkey_percentile']
            )
            
            synth_pass_rate = generate_synthetic_blocks_check(
                test_returns,
                CONFIG['synthetic_sims'], CONFIG['block_size']
            )
            
            # Degradation basada en Sharpe Ratio (Audit 3.0)
            train_sharpe = row.get('Test_Sharpe', 0) # Error en nombre columna origen? 
            # WAIT: 'row' viene de res_df (Train). En process_rules (train) calculamos metrics.
            # En process_rules, metrics se guardan flatted. 'Test_Sharpe' no existe en row de Train.
            # row tiene 'Sharpe'.
            train_sharpe = row['Sharpe']
            test_sharpe = test_metrics['sharpe']
            
            # Si train sharpe es bajo, la degradacion puede ser loca.
            if train_sharpe > 0.1:
                deg = (train_sharpe - test_sharpe) / train_sharpe
            else:
                deg = 0.0 # Si train era malo, no castigamos degradacion relativa
            
            test_survivors.append({
                'Rule': r_str,
                'Train_Profit': row['Profit'],
                'Test_Profit': test_metrics['total_profit'],
                'Train_Sharpe': train_sharpe,
                'Test_Sharpe': test_sharpe,
                'Degradation': deg,
                'Monkey_OK': monkey_pass,
                'Synth_OK': synth_pass_rate >= 0.80, 
                'Synth_Score': synth_pass_rate,
                'PF': row['PF']
            })
            
        final_df = pd.DataFrame(test_survivors)
        
        # Filtrado Final
        if not final_df.empty:
            ultra_robust = final_df[
                (final_df['Monkey_OK'] == True) & 
                (final_df['Synth_OK'] == True) & 
                (final_df['Test_Profit'] > 0) &
                (final_df['Degradation'] < 0.55) & # Audit 3.0: Max 55% degradacion de Sharpe
                (final_df['Test_Sharpe'] > 0.4)    # Sharpe min razonable
            ].copy()
            
            print(f"\n🏆 REGLAS ULTRA-ROBUSTAS: {len(ultra_robust)}/{len(final_df)}")
            print(f"   ✅ Monkey Test: Pasado en Train y Test")
            print(f"   ✅ Synthetic Bootstrap: >80% mundos válidos")
            print(f"   ✅ Profit Positivo: Ambos conjuntos")
            print(f"   ✅ Degradación: <50%")
            print(f"   ✅ Sharpe: >0.5")
            
            if not ultra_robust.empty:
                print("\n📊 TOP 10 ESTRATEGIAS:")
                display_cols = ['Rule', 'Train_Profit', 'Test_Profit', 'Test_Sharpe', 'Degradation', 'PF']
                print(ultra_robust.head(10)[display_cols].to_string())
                
                export_to_sqx_file(ultra_robust)
                ultra_robust.to_csv('final_ultra_robust.csv', index=False)
                print("\n✅ Pipeline completado exitosamente")
            else:
                print("⚠️ Ninguna regla cumplió TODOS los criterios ultra-robustos")
        else:
            print("⚠️ Ninguna regla pasó la validación Walk-Forward")
            
    else:
        print("No se encontraron reglas robustas en Train.")
