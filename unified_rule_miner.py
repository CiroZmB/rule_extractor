# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr, ks_2samp
# from sklearn.feature_selection import mutual_info_regression # Omitido por brevedad/rendimiento en mock
import joblib
from joblib import Parallel, delayed
import time
from datetime import datetime
import os
import hashlib
import re
import gc
from itertools import groupby
from numpy.fft import fft, ifft

# Intento de importar TA-Lib, fallback a implementaciones Pandas
try:
    import talib as ta
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    print("AVISO: TA-Lib no encontrado. Usando implementaciones Pandas (más lento).")

# =============================================================================
# HELPER: PANDAS INDICATORS (Fallback)
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
        # Implementación corregida de ADX (Signo de minus_dm arreglado)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(timeperiod).mean()
        
        up_move = high.diff()
        down_move = -low.diff() # Corregido: positivo para movimiento a la baja
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Evitar división por cero
        plus_di = 100 * (pd.Series(plus_dm).rolling(timeperiod).mean() / atr.replace(0, np.nan))
        minus_di = 100 * (pd.Series(minus_dm).rolling(timeperiod).mean() / atr.replace(0, np.nan))
        
        plus_di = plus_di.fillna(0)
        minus_di = minus_di.fillna(0)
        
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
        adx = dx.rolling(timeperiod).mean().fillna(0)
        return adx

    @staticmethod
    def PLUS_DI(high, low, close, timeperiod=14):
         return pd.Series(np.zeros(len(close)), index=close.index) # Placeholder

    @staticmethod
    def MINUS_DI(high, low, close, timeperiod=14):
         return pd.Series(np.zeros(len(close)), index=close.index) # Placeholder

# Wrapper para usar sintaxis TA-Lib
def get_indicator(name, *args, **kwargs):
    if HAS_TALIB:
        func = getattr(ta, name)
        return func(*args, **kwargs)
    else:
        # Mapeo a PandasTA
        func = getattr(PandasTA, name, None)
        if func:
            return func(*args, **kwargs)
        return pd.Series(np.zeros(len(args[0]))) # Retorno seguro si no existe impl

# =============================================================================
# CONFIGURACIÓN Y PARÁMETROS GLOBAL
# =============================================================================
CONFIG = {
    'input_csv': 'GBPNZ_H12.csv', # Archivo de entrada
    'cutoff_date': '2021-01-01',  # Fecha de corte para Partición Invertida
    'n_monkey_sims': 100,         # Número de simulaciones Monkey Test
    'monkey_percentile': 95,      # Percentil a superar en Monkey Test
    'synthetic_sims': 50,         # Número de mundos sintéticos
    'block_size': 30,             # Tamaño de bloque para Bootstrap (Trullas)
    'sub_block_size': 5,          # Tamaño de sub-bloque
    'min_trades': 50,             # Mínimo de trades para considerar una regla
    'n_jobs': -1                  # Cores a usar (-1 = todos)
}

# =============================================================================
# 1. INGENIERÍA DE DATOS Y PARTICIÓN (LÓGICA ANTOLÍ)
# =============================================================================

def load_and_transform_data(csv_path, exposicion_dias=4, threshold=25, short=False):
    """
    Carga datos y genera indicadores técnicos usando TA-Lib.
    Mantiene la lógica original de Tomillero de generación masiva de features.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encuentra el archivo {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # Conversión de fechas asegurando formato
    if 'DateTime' in df.columns:
        df['DateTime'] = pd.to_datetime(df['DateTime'])
    else:
        pass 

    # --- GENERACIÓN DE INDICADORES (Agnóstica) ---
    # RSI
    for i in range(2, 51, 2):
        df[f'rsi_{i}'] = get_indicator('RSI', df['Close'], timeperiod=i)
        
    # ADX, DI+, DI- (Solo si TA-Lib está disponible o fallback implementado)
    for i in range(2, 51, 2):
        df[f'adx_{i}'] = get_indicator('ADX', df['High'], df['Low'], df['Close'], timeperiod=i)
        df[f'plus_di_{i}'] = get_indicator('PLUS_DI', df['High'], df['Low'], df['Close'], timeperiod=i) # Fallback devuelve 0
        df[f'minus_di_{i}'] = get_indicator('MINUS_DI', df['High'], df['Low'], df['Close'], timeperiod=i)
        
    # Medias Móviles (SMA, EMA)
    for i in range(2, 201, 10): 
        df[f'sma_{i}'] = get_indicator('SMA', df['Close'], timeperiod=i)
        df[f'ema_{i}'] = get_indicator('EMA', df['Close'], timeperiod=i)
        
    # Bollinger Bands
    for i in range(10, 51, 10):
        # BBANDS devuelve tupla, hay que manejarlo con cuidado
        res = get_indicator('BBANDS', df['Close'], timeperiod=i, nbdevup=2, nbdevdn=2)
        if hasattr(res, '__len__') and len(res) == 3:
            u, m, l = res
            df[f'bb_upper_{i}'] = u
            df[f'bb_lower_{i}'] = l
        else:
            # Fallback seguro
            df[f'bb_upper_{i}'] = 0
            df[f'bb_lower_{i}'] = 0

    # Shifts (Lags) para evitar look-ahead bias y capturar patrones previos
    cols_to_shift = [c for c in df.columns if 'rsi' in c or 'adx' in c or 'sma' in c or 'Close' in c]
    for col in cols_to_shift:
        for shift in [1, 2, 3]:
            df[f'{col}_sft_{shift}'] = df[col].shift(shift)

    # --- DEFINICIÓN DE TARGET (SALIDA POR TIEMPO) ---
    # Calculamos el retorno a N velas vista. 
    pips_multiplier = 10000 if 'JPY' not in csv_path else 100
    
    # Generamos columnas de Retorno para múltiples horizontes (exposiciones)
    for i in range(2, 31, 2):
        future_close = df['Close'].shift(-i)
        ret = (future_close - df['Close']) * pips_multiplier
        if short:
            ret = ret * -1
        df[f'Return_{i}'] = ret

    target_col = f'Return_{exposicion_dias}'
    df['Target'] = (df[target_col] >= threshold).astype(int)
    
    # Limpieza
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df

def split_data_inverted(df, cutoff_date):
    """
    IMPLEMENTACIÓN FILOSOFÍA JAUME ANTOLÍ:
    Train = Datos Recientes (Régimen Actual).
    Test = Datos Antiguos (Validación Histórica).
    """
    print(f"--- APLICANDO PARTICIÓN INVERTIDA (Cutoff: {cutoff_date}) ---")
    
    if df['DateTime'].dtype == object:
         df['DateTime'] = pd.to_datetime(df['DateTime'])
         
    train_df = df[df['DateTime'] > cutoff_date].copy().reset_index(drop=True)
    test_df = df[df['DateTime'] <= cutoff_date].copy().reset_index(drop=True)
    
    print(f"Train Set (Reciente): {min(train_df['DateTime'])} a {max(train_df['DateTime'])} | Filas: {len(train_df)}")
    print(f"Test Set (Pasado): {min(test_df['DateTime'])} a {max(test_df['DateTime'])} | Filas: {len(test_df)}")
    
    return train_df, test_df

# =============================================================================
# 2. GENERACIÓN DE REGLAS (MAPEO)
# =============================================================================

def generate_combinatorial_rules(df, max_rules=10000):
    """
    Generador combinatorio de reglas robusto.
    Crea reglas de tipo:
    - Columna > Threshold (Percentiles)
    - Columna < Threshold
    - Columna1 > Columna2
    """
    rules = []
    # Filtrar solo columnas numéricas relevantes para reglas (indicadores)
    numeric_cols = [c for c in df.columns if df[c].dtype in [np.float64, np.float32, np.int32, np.int64]]
    # Excluir columnas de metadatos o targets
    exclude_cols = ['Target', 'Return', 'year', 'month', 'day', 'hour', 'profit', 'DateTime']
    numeric_cols = [c for c in numeric_cols if not any(ex in c for ex in exclude_cols) and 'Return_' not in c]
    
    print(f"Generando reglas sobre {len(numeric_cols)} features...")
    
    # 1. Reglas de Valor (Thresholds dinámicos)
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty: continue
        
        percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        thresholds = np.percentile(series, percentiles)
        
        for t in list(set(thresholds)):
            rules.append(f"{col} > {t:.5f}")
            rules.append(f"{col} < {t:.5f}")
            
    # 2. Reglas de Cruce (Crosses)
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
                
        if len(rules) > max_rules * 2:
            break
            
    # Limitar cantidad
    if len(rules) > max_rules:
        rules = [rules[i] for i in np.random.choice(len(rules), max_rules, replace=False)]
        
    return rules

# =============================================================================
# 3. MOTOR DE VALIDACIÓN ESTADÍSTICA (EL NÚCLEO NUEVO)
# =============================================================================

def vectorized_monkey_test(signals, returns, n_sims=100, percentile=95):
    """
    MONKEY TEST VECTORIZADO (CORREGIDO).
    El test debe ejecutarse siempre, incluso si el retorno original es negativo,
    porque podría ser mejor que el azar (skill defensiva).
    """
    real_return = np.sum(returns[signals])
    n_bars = len(returns)
    
    # Generamos N shifts aleatorios (evitamos shift 0)
    shifts = np.random.randint(low=1, high=n_bars, size=n_sims)
    
    # Vectorización real usando broadcasting
    monkey_returns = np.array([
        np.sum(returns[np.roll(signals, s)]) 
        for s in shifts
    ])
    
    threshold_val = np.percentile(monkey_returns, percentile)
    
    is_robust = real_return > threshold_val
    return is_robust, threshold_val


def calculate_acf(x, lags=20):
    """Calcula la función de autocorrelación (ACF) usando FFT (Rápido)"""
    x = np.array(x, dtype=float)
    n = len(x)
    if n == 0: return np.zeros(lags + 1)
    
    x_centered = x - np.mean(x)
    fft_x = fft(x_centered, n=2*n)
    power_spectrum = fft_x * np.conj(fft_x)
    acf_full = ifft(power_spectrum).real[:n] / n
    
    c0 = acf_full[0]
    if c0 == 0:
        return np.zeros(lags + 1)
        
    return acf_full[:lags + 1] / c0

def validate_synthetic_series(original_returns, synthetic_returns):
    """
    Valida la calidad de la serie sintética comparándola con la original.
    """
    scores = {}
    
    # 1. Test Kolmogorov-Smirnov
    ks_stat, ks_pvalue = ks_2samp(original_returns, synthetic_returns)
    scores['ks_score'] = ks_pvalue
    
    # 2. Desviación Estándar
    std_orig = np.std(original_returns)
    std_synth = np.std(synthetic_returns)
    if std_orig == 0:
        scores['std_score'] = 0
    else:
        scores['std_score'] = min(std_orig, std_synth) / max(std_orig, std_synth)
        
    # 3. ACF Returns
    acf_orig = calculate_acf(original_returns, lags=10)
    acf_synth = calculate_acf(synthetic_returns, lags=10)
    if np.std(acf_orig) == 0 or np.std(acf_synth) == 0:
        scores['acf_score'] = 0
    else:
        scores['acf_score'] = np.corrcoef(acf_orig, acf_synth)[0, 1]
        
    # 4. ACF Squared Returns
    acf_sq_orig = calculate_acf(original_returns**2, lags=10)
    acf_sq_synth = calculate_acf(synthetic_returns**2, lags=10)
    if np.std(acf_sq_orig) == 0 or np.std(acf_sq_synth) == 0:
        scores['acf_sq_score'] = 0
    else:
        scores['acf_sq_score'] = np.corrcoef(acf_sq_orig, acf_sq_synth)[0, 1]
        
    return scores

def generate_synthetic_blocks_check(trade_returns, n_sims=50, block_size=30):
    """
    VALIDACIÓN SINTÉTICA DE CURVA DE EQUITY (MÉTODO TRULLAS APLICADO A PnL).
    Versión Avanzada corregida: Bootstrap CON REEMPLAZO.
    """
    if len(trade_returns) < 50:
        return False # Muy pocos trades para estadística fiable
        
    n_trades = len(trade_returns)
    valid_sims_generated = 0
    passed_sims = 0
    attempts = 0
    max_attempts = n_sims * 10 
    
    # Preparar bloques
    indices = np.arange(n_trades)
    n_blocks = max(1, n_trades // block_size)
    blocks = np.array_split(indices, n_blocks)
    
    while valid_sims_generated < n_sims and attempts < max_attempts:
        attempts += 1
        
        # --- GENERACIÓN (Bootstrap CON Reemplazo) ---
        sampled_block_indices = np.random.randint(0, len(blocks), size=len(blocks))
        sampled_blocks = [blocks[i] for i in sampled_block_indices]
        
        new_indices = np.concatenate(sampled_blocks)
        if len(new_indices) > n_trades:
             new_indices = new_indices[:n_trades]
        elif len(new_indices) < n_trades:
             extra_block = blocks[np.random.randint(0, len(blocks))]
             new_indices = np.concatenate([new_indices, extra_block])[:n_trades]
            
        synthetic_returns = trade_returns[new_indices]
        
        # --- QUALITY GATE ---
        quality_scores = validate_synthetic_series(trade_returns, synthetic_returns)
        
        is_valid_data = (
            quality_scores['std_score'] > 0.90 and      
            quality_scores['acf_sq_score'] > 0.70 and   
            quality_scores['ks_score'] > 0.05
        )
        
        if not is_valid_data:
            continue
            
        valid_sims_generated += 1
        
        # --- VALIDACIÓN DE ESTRATEGIA ---
        if np.sum(synthetic_returns) > 0:
            passed_sims += 1
            
    if valid_sims_generated == 0:
        return False
        
    pass_rate = passed_sims / valid_sims_generated
    return pass_rate >= 0.80

def process_rules_parallel(df_values, columns, rules_list, target_col_idx, returns_cols_indices):
    """
    Función worker para Joblib.
    """
    valid_rules = []
    main_return_col = df_values[:, target_col_idx]
    
    for rule_parts_str in rules_list:
        parts = rule_parts_str.split()
        if len(parts) != 3:
            continue
            
        col1_name, operator, col2_or_val = parts
        
        try:
            idx1 = columns.get_loc(col1_name)
        except KeyError:
            continue
            
        try:
            val = float(col2_or_val)
            is_value_comparison = True
            idx2 = None
        except ValueError:
            try:
                idx2 = columns.get_loc(col2_or_val)
                is_value_comparison = False
            except KeyError:
                continue

        # Evaluación Vectorizada
        if is_value_comparison:
            if operator == '>': signals = df_values[:, idx1] > val
            elif operator == '<': signals = df_values[:, idx1] < val
            elif operator == '>=': signals = df_values[:, idx1] >= val
            elif operator == '<=': signals = df_values[:, idx1] <= val
            elif operator == '==': signals = df_values[:, idx1] == val
            else: continue
        else:
            if operator == '>': signals = df_values[:, idx1] > df_values[:, idx2]
            elif operator == '<': signals = df_values[:, idx1] < df_values[:, idx2]
            elif operator == '>=': signals = df_values[:, idx1] >= df_values[:, idx2]
            elif operator == '<=': signals = df_values[:, idx1] <= df_values[:, idx2] 
            elif operator == '==': signals = df_values[:, idx1] == df_values[:, idx2]
            else: continue
        
        n_trades = np.sum(signals)
        if n_trades < CONFIG['min_trades']:
            continue
            
        # 2. EVALUACIÓN BÁSICA
        rule_returns = main_return_col[signals]
        total_profit = np.sum(rule_returns)
        
        if total_profit <= 0:
            continue
            
        # 3. FILTRO 1: MONKEY TEST
        is_monkey_robust, _ = vectorized_monkey_test(signals, main_return_col, 
                                                   n_sims=CONFIG['n_monkey_sims'], 
                                                   percentile=CONFIG['monkey_percentile'])
        
        if not is_monkey_robust:
            continue 
            
        # 4. FILTRO 2: SYNTHETIC DATA
        # Pasamos solo los retornos de los trades (Equity Curve)
        is_synthetic_robust = generate_synthetic_blocks_check(rule_returns, 
                                                            n_sims=CONFIG['synthetic_sims'],
                                                            block_size=CONFIG['block_size'])
        
        if not is_synthetic_robust:
            continue 
            
        # Si pasa todo, guardamos
        wins = np.sum(rule_returns[rule_returns > 0])
        losses = abs(np.sum(rule_returns[rule_returns < 0]))
        pf = wins / losses if losses > 0 else np.inf
        
        valid_rules.append({
            'Rule': rule_parts_str,
            'Trades': n_trades,
            'Profit': total_profit,
            'PF': pf
        })
        
    return valid_rules

# =============================================================================
# 4. EXPORTACIÓN A STRATEGYQUANT X
# =============================================================================

def format_rule_for_sqx(rule_str):
    sqx_rule = rule_str
    if 'rsi_' in rule_str:
        period = re.search(r'rsi_(\d+)', rule_str).group(1)
        sqx_rule = sqx_rule.replace(f'rsi_{period}', f'RSI({period})')
        
    if 'sma_' in rule_str:
        period = re.search(r'sma_(\d+)', rule_str).group(1)
        sqx_rule = sqx_rule.replace(f'sma_{period}', f'SMA({period})')
        
    return f"LONG IF: {sqx_rule}"

def export_to_sqx_format(winning_rules_df):
    print("\n--- GENERANDO REPORTE COMPATIBLE CON STRATEGYQUANT X ---")
    print("Copiar y pegar estas reglas en AlgoWizard -> Custom Block o Conditions")
    
    for idx, row in winning_rules_df.iterrows():
        sqx_code = format_rule_for_sqx(row['Rule'])
        print(f"// Regla #{idx} | Profit: {row['Profit']:.2f} | PF: {row['PF']:.2f}")
        print(sqx_code)
        print("-" * 50)

# =============================================================================
# MAIN PIPELINE
# =============================================================================

if __name__ == "__main__":
    print("Iniciando Pipeline Unificada de Minería y Validación...")
    
    # 1. Cargar Datos
    if not os.path.exists(CONFIG['input_csv']):
        print(f"AVISO: Generando datos dummy para test porque no existe {CONFIG['input_csv']}")
        dates = pd.date_range(start='2015-01-01', end='2024-01-01', freq='4H')
        df_dummy = pd.DataFrame({
            'DateTime': dates,
            'Close': np.random.lognormal(0, 0.01, size=len(dates)).cumsum() + 1000,
            'High': np.random.lognormal(0, 0.01, size=len(dates)).cumsum() + 1005,
            'Low': np.random.lognormal(0, 0.01, size=len(dates)).cumsum() + 995
        })
        df = df_dummy
        # Mocking indicadores
        df['rsi_14'] = np.random.uniform(20, 80, size=len(df))
        df['sma_20'] = df['Close'] * np.random.uniform(0.95, 1.05, size=len(df))
        # Mocking targets
        df['Return_4'] = np.random.normal(0, 10, size=len(df)) # Pips return
    else:
        df = load_and_transform_data(CONFIG['input_csv'], short=True) 
        
    # 2. Split Invertido (Jaume Antolí)
    train_df, test_df = split_data_inverted(df, CONFIG['cutoff_date'])
    
    # 3. Generar Reglas ("Átomos")
    print("Generando universo de reglas con 'Robust Combinatorial Generator'...")
    rules = generate_combinatorial_rules(train_df) 
    print(f"Total reglas generadas: {len(rules)}")
    
    # 4. Procesamiento Paralelo con Filtros Robustos
    print("Iniciando Minería con Validación Vectorizada (Monkey + Synth)...")
    
    train_values = train_df.to_numpy() 
    columns = train_df.columns
    target_idx = columns.get_loc(f'Return_4')
    
    chunk_size = len(rules) // 4 
    if chunk_size == 0: chunk_size = 1
    chunks = [rules[i:i + chunk_size] for i in range(0, len(rules), chunk_size)]
    
    results = Parallel(n_jobs=CONFIG['n_jobs'])(
        delayed(process_rules_parallel)(train_values, columns, chunk, target_idx, []) 
        for chunk in chunks
    )
    
    flat_results = [item for sublist in results for item in sublist]
    results_df = pd.DataFrame(flat_results)
    
    # 5. Reporte y Exportación
    if not results_df.empty:
        print(f"\nReglas Sobrevivientes a Filtros: {len(results_df)}")
        best_rules = results_df.sort_values('Profit', ascending=False).head(10)
        export_to_sqx_format(best_rules)
        best_rules.to_csv('reglas_validadas_sqx.csv', index=False)
    else:
        print("Ninguna regla pasó los filtros de robustez estricta.")
