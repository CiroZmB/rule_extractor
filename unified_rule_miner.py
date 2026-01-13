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

# Wrapper
def get_indicator(name, *args, **kwargs):
    if HAS_TALIB:
        func = getattr(ta, name)
        return func(*args, **kwargs)
    else:
        func = getattr(PandasTA, name, None)
        if func:
            return func(*args, **kwargs)
        return pd.Series(np.zeros(len(args[0]))) 

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
CONFIG = {
    'input_csv': 'GBPNZ_H12.csv',
    'cutoff_date': '2021-01-01',  
    'n_monkey_sims': 100,         
    'monkey_percentile': 95,      
    'synthetic_sims': 50,         
    'block_size': 30,             
    'min_trades': 50,
    'exposicion_dias': 4,
    'n_jobs': -1                  
}

# =============================================================================
# 1. INGENIERÍA DE DATOS
# =============================================================================

def load_and_transform_data(csv_path, threshold=25, short=False):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encuentra {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    if 'DateTime' in df.columns:
        df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # --- 1. Calcular TARGETS PRIMERO (Futuro) ---
    pips_multiplier = 10000 if 'JPY' not in csv_path else 100
    max_horizon_bars = 30
    
    for i in range(2, max_horizon_bars + 1, 2):
        future_close = df['Close'].shift(-i)
        ret = (future_close - df['Close']) * pips_multiplier
        if short: ret = ret * -1
        df[f'Return_{i}'] = ret
        
    target_col = f'Return_{CONFIG["exposicion_dias"]}'
    # Si no existe target col (por configuración errada), fallback
    if target_col not in df.columns:
        target_col = 'Return_4'
        
    df['Target'] = (df[target_col] >= threshold).astype(int)

    # --- 2. Calcular INDICADORES BASE ---
    for i in range(2, 51, 2):
        df[f'rsi_{i}'] = get_indicator('RSI', df['Close'], timeperiod=i)
        df[f'adx_{i}'] = get_indicator('ADX', df['High'], df['Low'], df['Close'], timeperiod=i)
        # Solo calculamos DI si tenemos TA-Lib o usará el fallback de PandasTA
        df[f'plus_di_{i}'] = get_indicator('PLUS_DI', df['High'], df['Low'], df['Close'], timeperiod=i)
        df[f'minus_di_{i}'] = get_indicator('MINUS_DI', df['High'], df['Low'], df['Close'], timeperiod=i)
        
    for i in range(2, 201, 10): 
        df[f'sma_{i}'] = get_indicator('SMA', df['Close'], timeperiod=i)
        df[f'ema_{i}'] = get_indicator('EMA', df['Close'], timeperiod=i)
        
    for i in range(10, 51, 10):
        res = get_indicator('BBANDS', df['Close'], timeperiod=i, nbdevup=2, nbdevdn=2)
        if hasattr(res, '__len__') and len(res) == 3:
            df[f'bb_upper_{i}'], _, df[f'bb_lower_{i}'] = res
        else:
            df[f'bb_upper_{i}'], df[f'bb_lower_{i}'] = 0, 0

    # --- 3. Crear SHIFTS (Lags) ---
    cols_to_shift = [c for c in df.columns if any(x in c for x in ['rsi', 'adx', 'sma', 'ema', 'bb', 'plus', 'minus', 'Close'])]
    for col in cols_to_shift:
        for shift in [1, 2, 3]:
            df[f'{col}_sft_{shift}'] = df[col].shift(shift)

    # --- 4. DATA CLEANING ---
    df = df.iloc[:-max_horizon_bars].copy()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

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
    if len(trade_returns) < 50: return False
    
    n_trades = len(trade_returns)
    valid_sims = 0
    passed = 0
    attempts = 0
    max_attempts = n_sims * 10
    
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
        
        if weighted_score > 0.80:
            valid_sims += 1
            if np.sum(synth_ret) > 0: passed += 1
            
    if valid_sims == 0: return False
    return (passed / valid_sims) >= 0.80

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
        
        if metrics['total_profit'] <= 0: continue
        
        # Validation
        robust, _ = vectorized_monkey_test_ultra(s, main_return_col, 
                                                 CONFIG['n_monkey_sims'], 
                                                 CONFIG['monkey_percentile'])
        if not robust: continue
        
        if not generate_synthetic_blocks_check(rule_returns, 
                                             CONFIG['synthetic_sims'], 
                                             CONFIG['block_size']):
            continue
            
        valid_rules.append({
            'Rule': rule_str,
            'Trades': n_trades,
            'Profit': metrics['total_profit'],
            'PF': metrics['pf'],
            'Sharpe': metrics['sharpe'],
            'MaxDD': metrics['max_dd']
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
    
    # 1. LOAD OR DUMMY
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
        df = load_and_transform_data('temp_dummy.csv')
        try: os.remove('temp_dummy.csv')
        except: pass
    else:
        df = load_and_transform_data(CONFIG['input_csv'])

    # 2. SPLIT
    train_df, test_df = split_data_inverted(df, CONFIG['cutoff_date'])
    
    # 3. GENERAR
    rules = generate_combinatorial_rules(train_df)
    print(f"Reglas generadas: {len(rules)}")
    
    # 4. PROCESAMIENTO PARALELO
    n_cores = os.cpu_count() or 4
    chunk_size = max(100, len(rules) // (n_cores * 3))
    chunks = [rules[i:i + chunk_size] for i in range(0, len(rules), chunk_size)]
    
    print(f"Procesando en {len(chunks)} chunks con {n_cores} cores...")
    
    tv = train_df.to_numpy()
    cols = train_df.columns
    t_idx = cols.get_loc(f"Return_{CONFIG['exposicion_dias']}")
    
    with tqdm_joblib(tqdm(total=len(rules))) as pbar:
        results = Parallel(n_jobs=CONFIG['n_jobs'])(
            delayed(process_rules)(tv, cols, ch, t_idx) for ch in chunks
        )
    
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
            
            synth_pass = generate_synthetic_blocks_check(
                test_returns,
                CONFIG['synthetic_sims'], CONFIG['block_size']
            )
            
            deg = (row['Profit'] - test_metrics['total_profit']) / row['Profit'] if row['Profit'] > 0 else 999
            
            test_survivors.append({
                'Rule': r_str,
                'Train_Profit': row['Profit'],
                'Test_Profit': test_metrics['total_profit'],
                'Test_Sharpe': test_metrics['sharpe'],
                'Degradation': deg,
                'Monkey_OK': monkey_pass,
                'Synth_OK': synth_pass,
                'PF': row['PF']
            })
            
        final_df = pd.DataFrame(test_survivors)
        
        # Filtrado Final
        if not final_df.empty:
            ultra_robust = final_df[
                (final_df['Monkey_OK']) & 
                (final_df['Synth_OK']) &
                (final_df['Test_Profit'] > 0)
            ].copy()
            
            print(f"\nReglas Ultra-Robustas (Train + Test OK): {len(ultra_robust)}")
            if not ultra_robust.empty:
                export_to_sqx_file(ultra_robust)
                ultra_robust.to_csv('final_ultra_robust.csv')
        else:
            print("Ninguna regla pasó la validación Walk-Forward.")
            
    else:
        print("No se encontraron reglas robustas en Train.")
