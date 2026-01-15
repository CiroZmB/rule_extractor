
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import sys

# Importar lógica del Minero Unificado
sys.path.append(os.getcwd())
import unified_rule_miner as miner

# Configuración de página
st.set_page_config(page_title="QuantiLab: Scientific Rule Miner", layout="wide", page_icon="🔬")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]

    # 1. ROBUST DATETIME
    if 'Date' in df.columns and 'Time' in df.columns and 'DateTime' not in df.columns:
        try:
            df['DateTime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str))
            df.drop(columns=['Date', 'Time'], inplace=True)
        except Exception: pass

    cols_map = {'date':'DateTime','time':'DateTime','Datetime':'DateTime','datetime':'DateTime','Timestamp':'DateTime'}
    for old, new in cols_map.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old:new}, inplace=True)
        elif old in df.columns and new in df.columns:
             df.drop(columns=[old], inplace=True)

    # 2. ROBUST VOLUME
    if 'Volume' not in df.columns:
        for v in ['TickVol','tickvol','volume','Tick Volume']:
            if v in df.columns:
                df.rename(columns={v:'Volume'}, inplace=True)
                break
    else:
            if v in df.columns: df.drop(columns=[v], inplace=True)
            
    # FORCE NUMERIC TYPES (Prevent Object/String pollution)
    num_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for c in num_cols:
        if c in df.columns:
            # Coerce errors to NaN then drop or fill
            df[c] = pd.to_numeric(df[c], errors='coerce')
    
    # Validation
    df.dropna(subset=['Close', 'DateTime'], inplace=True, ignore_index=True)

    if 'DateTime' in df.columns:
        df['DateTime'] = pd.to_datetime(df['DateTime'])
    else:
        return pd.DataFrame()
        
    return df

def resample_ohlc_data(df, period):
    """Resamplea datos OHLCV a un nuevo timeframe."""
    if period == '1min': return df
    
    df_res = df.set_index('DateTime').resample(period)
    
    agg_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }
    if 'Volume' in df.columns:
        agg_dict['Volume'] = 'sum'
        
    df_res = df_res.agg(agg_dict).dropna().reset_index()
    return df_res

def plot_synthetic_lab(df_prices):
    # DOWNSAMPLING VISUAL (Solo para el gráfico)
    MAX_POINTS = 2000
    if len(df_prices) > MAX_POINTS:
        step = len(df_prices) // MAX_POINTS
        df_vis = df_prices.iloc[::step].copy()
    else:
        df_vis = df_prices
        
    fig = go.Figure()
    synth_cols = [c for c in df_vis.columns if 'Synth' in c]
    for col in synth_cols:
        fig.add_trace(go.Scatter(
            x=df_vis.index, y=df_vis[col], 
            mode='lines', line=dict(width=1, color='rgba(100, 100, 100, 0.2)'),
            showlegend=False
        ))
    fig.add_trace(go.Scatter(
        x=df_vis.index, y=df_vis['Original'], 
        mode='lines', line=dict(width=2, color='cyan'),
        name='Real Market'
    ))
    fig.update_layout(
        title={
            'text': "Laboratorio Sintético: 50 Realidades Alternativas (Bootstrap)",
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        template="plotly_dark",
        height=500,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# =============================================================================
# MAIN UI STRUCTURE (ETL-First Architecture)
# =============================================================================
st.title("🔬 QuantiLab: Scientific Rule Factory")

# Session State Init
if 'processed_data' not in st.session_state: st.session_state['processed_data'] = None
if 'config' not in st.session_state: st.session_state['config'] = {}

# Tabs
tabs = st.tabs(["⚙️ 1. Config & Data (ETL)", "📊 2. Lab Sintético", "⛏️ 3. Minería", "📈 4. Resultados"])

# --- TAB 0: ETL (Load -> Filter -> Resample -> Cache) ---
with tabs[0]:
    st.header("⚙️ Ingeniería de Datos (ETL)")
    st.info("""
    **🔧 PROCESO INTERNO DE ESTA PESTAÑA:**
    1.  **Carga y Limpieza:** Se leen los datos crudos y se fuerzan las conversiones a numérico (se descarta texto/basura).
    2.  **Filtrado Temporal:** Se recortan los datos fuera del rango de fecha seleccionado para ahorrar memoria.
    3.  **Resampling (Compresión):** Se transforman las velas M1 (1 minuto) al Timeframe Objetivo (ej. H1) sumando volúmenes y recalculando OHLC.
    4.  **Caché:** Se guardan solo los datos limpios en la memoria de sesión; los datos crudos se eliminan.
    """)
    
    col_up, col_conf = st.columns([1, 2])
    
    with col_up:
        uploaded_file = st.file_uploader("📂 Cargar XML/CSV (M1)", type=["csv"])
        
        # Carga Inicial (Raw)
        if uploaded_file:
            if 'raw_data' not in st.session_state or st.session_state.get('last_file') != uploaded_file.name:
                df_raw = load_data(uploaded_file)
                st.session_state['raw_data'] = df_raw
                st.session_state['last_file'] = uploaded_file.name
                st.success(f"Cargado: {len(df_raw):,} velas M1")
            else:
                df_raw = st.session_state['raw_data']
        elif os.path.exists('GBPNZ_H12.csv'):
             # Demo fallback
             if 'raw_data' not in st.session_state:
                 df_raw = load_data('GBPNZ_H12.csv')
                 st.session_state['raw_data'] = df_raw
             else:
                 df_raw = st.session_state.get('raw_data')

    if 'raw_data' in st.session_state:
        df_raw = st.session_state['raw_data']
        abs_min = df_raw['DateTime'].min().date()
        abs_max = df_raw['DateTime'].max().date()
        
        with col_conf:
            with st.form("etl_config"):
                st.subheader("Configuración del Experimento")
                c1, c2 = st.columns(2)
                
                with c1:
                    # 1. Filtro Fecha
                    global_dates = st.date_input("Rango Global (Train + Filtro)", 
                                                value=(abs_min, abs_max), min_value=abs_min, max_value=abs_max)
                    
                    # 2. Resampling
                    target_tf = st.selectbox("Timeframe Objetivo (Resampling)", 
                                            ["H1", "H4", "H12", "D1", "M30", "M15", "M5", "M1"], 
                                            index=0, help="Convierte M1 al TF de la estrategia PARA SIEMPRE.")
                
                with c2:
                    # 3. Validation Split
                    cutoff_date = st.date_input("Fecha de Corte (Train | Test)", value=pd.to_datetime('2023-01-01'))
                    partition_mode = st.radio("Modo Validación", ["Invertido (Antolí)", "Clásico (Walk-Forward)"])
                    
                st.markdown("---")
                c3, c4 = st.columns(2)
                with c3:
                    direction_mode = st.radio("Dirección", ["Long", "Short", "Ambos"], index=2, horizontal=True)
                with c4:
                    exposure = st.slider("Exposición (Barras)", 2, 20, 4)
                    min_trades = st.number_input("Min Trades", value=50)

                submit_etl = st.form_submit_button("⚡ PROCESAR Y OPTIMIZAR DATOS", type="primary")
        
        if submit_etl:
            if len(global_dates) != 2:
                st.error("Selecciona rango completo.")
                st.stop()
                
            with st.spinner("🔄 Procesando: Filtrando fechas y Resampleando a " + target_tf + "..."):
                # 1. Filtro Fechas
                mask = (df_raw['DateTime'].dt.date >= global_dates[0]) & (df_raw['DateTime'].dt.date <= global_dates[1])
                df_temp = df_raw.loc[mask].copy()
                
                # 2. Resampling (ETL Core)
                tf_map = {'M1':'1min','M5':'5min','M15':'15min','M30':'30min',
                          'H1':'1h','H4':'4h','H12':'12h','D1':'1D'}
                
                if target_tf != "M1":
                    df_proc = resample_ohlc_data(df_temp, tf_map[target_tf])
                    msg_resample = f"Resampleado de {len(df_temp):,} (M1) a {len(df_proc):,} ({target_tf})"
                else:
                    df_proc = df_temp
                    msg_resample = f"Datos M1 Crudos: {len(df_proc):,} velas (¡Cuidado con el rendimiento!)"
                
                # 3. Guardar en Session
                st.session_state['processed_data'] = df_proc
                st.session_state['config'] = {
                    'cutoff': cutoff_date, 'partition': partition_mode, 
                    'direction': direction_mode, 'exposure': exposure,
                    'min_trades': min_trades, 'target_tf': target_tf
                }
                
                st.success(f"✅ ¡Datos Listos! {msg_resample}. Pasa a la siguiente pestaña.")
                st.session_state['etl_done'] = True

# --- COMMON DATA CHECK ---
if not st.session_state.get('etl_done'):
    st.info("👈 Por favor, carga y procesa los datos en la Pestaña 1 primero.")
    st.stop()

df = st.session_state['processed_data']
cfg = st.session_state['config']

# --- TAB 1: LAB SINTÉTICO ---
with tabs[1]:
    st.header(f"🧬 Lab (Datos: {len(df)} velas {cfg['target_tf']})")
    
    st.info("""
    **🧬 PROCESO INTERNO: VALIDACIÓN DE HIPÓTESIS NULA**
    1.  **Bootstrapping:** Se generan 50 "Mundos Paralelos" reordenando bloques aleatorios de tus precios reales.
    2.  **Objetivo:** Romper la secuencia temporal pero mantener la volatilidad y memoria (autocorrelación).
    3.  **¿PARA QUÉ SIRVE?** 
        *   **Aquí (Visual):** Para que COMPRUEBES que los mundos son realistas.
        *   **En Minería (Invisible):** Cada regla candidata debe ganar en el 80% de estos mundos para ser aceptada. Si falla, se descarta.
    """)
    
    c_config, c_btn = st.columns([3, 1])
    with c_config:
        # Selector local para Lab
        min_d, max_d = df['DateTime'].min().date(), df['DateTime'].max().date()
        lab_period = st.date_input("Periodo ADN", value=(min_d, max_d), min_value=min_d, max_value=max_d, key='lab_d')
        
    with c_btn:
        st.write("") # Spacer
        st.write("")
        gen_clicked = st.button("🧪 Generar Mundos", use_container_width=True)

    if gen_clicked:
        mask = (df['DateTime'].dt.date >= lab_period[0]) & (df['DateTime'].dt.date <= lab_period[1])
        lab_input = df.loc[mask, 'Close']
        
        with st.spinner("Generando..."):
            synths = miner.generate_synthetic_price_paths(lab_input, n_sims=50, block_size=30)
            
            # 1. Gráfico (Full Width)
            st.plotly_chart(plot_synthetic_lab(synths), use_container_width=True)
            
            # 2. Métricas
            scores = []
            orig = lab_input.pct_change().dropna()
            for c in synths.columns:
                if 'Synth' in c:
                    s = synths[c].pct_change().dropna()
                    l = min(len(orig), len(s))
                    m = miner.validate_synthetic_series(orig.iloc[:l], s.iloc[:l])
                    scores.append(m)
            
            avg = pd.DataFrame(scores).mean(numeric_only=True)
            c1,c2,c3 = st.columns(3)
            c1.metric("Similitud Volatilidad", f"{avg['std_score']:.2%}")
            c2.metric("ACF Score (Clustering)", f"{avg['acf_sq_score']:.2f}")
            c3.metric("KS Score", f"{avg['ks_score']:.2f}")

            with st.expander("🔍 Ver Detalles por Universo (Auditoría)"):
                st.dataframe(pd.DataFrame(scores).style.format("{:.2f}"))

# --- TAB 2: MINERÍA ---
with tabs[2]:
    st.header("⛏️ Minería de Reglas")
    
    st.info("""
    **⛏️ PROCESO INTERNO: FUERZA BRUTA INTELIGENTE**
    1.  **Partición Invertida:** Entrenamos en lo RECIENTE (Train), validamos en lo ANTIGUO (Test).
    2.  **Cálculo Denso:** Se calculan ~200 indicadores técnicos por vela en paralelo.
    3.  **Feature Selection:** Si dos indicadores son iguales (correlación > 97%), se elimina uno para no perder tiempo.
    4.  **Búsqueda:** Se prueban miles de combinaciones (ej. RSI > 70) buscando las que tengan Profit Positivo.
    """)
    
    # Preparar queue
    queue = []
    if cfg['direction'] in ['Long', 'Ambos']: queue.append({'L':'Long', 'S':False})
    if cfg['direction'] in ['Short', 'Ambos']: queue.append({'L':'Short', 'S':True})
    
    st.write(f"Configuración: {cfg['target_tf']} | Exp: {cfg['exposure']} | Train/Test Corte: {cfg['cutoff']}")
    
    # --- CONFIGURACIÓN DE RIGOR AVANZADA ---
    with st.expander("⚙️ Configuración Avanzada de Rigor (Relax Protocol)"):
        st.markdown("**Ajuste fino de filtros de calidad:**")
        c_rig1, c_rig2, c_rig3 = st.columns(3)
        with c_rig1:
            ui_synth_threshold = st.slider("Umbral Sintético (Mundos)", 0.0, 1.0, 0.80, help="Min % de mundos que debe ganar.")
        with c_rig2:
            ui_monkey_pct = st.number_input("Monkey Percentile", 50, 99, 95, help="Exigencia vs Azar (95% estándar)")
        with c_rig3:
            ui_profit_factor = st.number_input("Min Profit Factor", 1.0, 3.0, 1.3, step=0.1)

    if st.button("🚀 INICIAR PROCESO DE MINERÍA"):
        progress = st.progress(0)
        status = st.empty()
        results = []
        test_data = {}
        
        # Guardar dataset procesado temporal para el miner
        # El miner espera un CSV path, así que guardamos el DF PROCESADO
        temp_csv = "temp_processed_for_mining.csv"
        df.to_csv(temp_csv, index=False)
        
        try:
            # Update MINER CONFIG with UI values
            miner.CONFIG['input_csv'] = temp_csv
            miner.CONFIG['cutoff_date'] = str(cfg['cutoff'])
            miner.CONFIG['exposicion_dias'] = cfg['exposure']
            miner.CONFIG['min_trades'] = cfg['min_trades']
            # Rigor params
            miner.CONFIG['synth_threshold'] = ui_synth_threshold
            miner.CONFIG['monkey_percentile'] = ui_monkey_pct
            miner.CONFIG['min_profit_factor'] = ui_profit_factor
            
            for i, task in enumerate(queue):
                status.text(f"Minando {task['L']}...")
                # Carga "Dummy" (ya procesado)
                # Pasamos target_tf=None porque YA procesamos el resampling en Tab 0.
                df_m, cols = miner.load_and_transform_data(temp_csv, target_timeframe=None, short=task['S'])
                
                train, test = miner.split_data_inverted(df_m, str(cfg['cutoff']))
                train, test, feats = miner.reduce_multicollinearity_safe(train, test, cols, threshold=0.97)
                
                test_data[task['L']] = (test, train.columns) # Guardar para validación
                
                rules = miner.generate_combinatorial_rules(train, max_rules=3000)
                
                # Evaluar
                chunk_res = miner.process_rules(train.to_numpy(), train.columns, rules, train.columns.get_loc(f"Return_{cfg['exposure']}"))
                for r in chunk_res: r['Direction'] = task['L']
                results.extend(chunk_res)
                progress.progress((i+1)/len(queue))
                
            if results:
                st.session_state['mining_results'] = pd.DataFrame(results)
                st.session_state['test_data'] = test_data
                st.success(f"¡Éxito! {len(results)} reglas encontradas.")
            else:
                st.warning("No se encontraron reglas robustas.")
                
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            if os.path.exists(temp_csv): os.remove(temp_csv)

# --- TAB 3: RESULTADOS ---
with tabs[3]:
    st.header("🏆 Resultados")
    
    st.info("""
    **🏆 PROCESO INTERNO: PRUEBA DE ESTRÉS**
    1.  **Walk-Forward:** Las reglas ganadoras se corren en el pasado desconocido (Test Set).
    2.  **Análisis de Degradación:** Comparamos cuánto Profit perdieron al salir de la zona de confort (Train).
    3.  **Veredicto:** Solo pasan las reglas que sobreviven al cambio de régimen de mercado (Profit > 0 y Hold > 10%).
    """)
    if 'mining_results' in st.session_state:
        df_res = st.session_state['mining_results']
        st.dataframe(df_res.sort_values('Profit', ascending=False).head(50))
        
        if st.button("Validar en Test (Walk-Forward)"):
            # Lógica de Validación Walk-Forward (Restaurada)
            st.info("🛡️ Validando reglas en datos fuera de muestra (Test Set)...")
            
            top_rules = df_res.sort_values('Profit', ascending=False).head(20)
            test_data_store = st.session_state.get('test_data', {})
            results_wf = []
            
            progress_wf = st.progress(0)
            
            for i, (idx, row) in enumerate(top_rules.iterrows()):
                direction = row['Direction']
                
                # Check si tenemos datos de test para esa dirección
                if direction not in test_data_store:
                    continue
                    
                # Recuperar (DF Test, Columnas usadas en Train)
                test_df, train_cols = test_data_store[direction]
                
                # Preparar numpy arrays
                # IMPORTANTE: Re-alinear columnas. El TestDF tiene todas las columnas, 
                # pero necesitamos mapearlas a los índices que espera la regla.
                test_v = test_df.to_numpy()
                cols_map = {n: k for k, n in enumerate(test_df.columns)} # Map nombre -> indice en Test
                
                # Target Index (Return_N)
                target_col = f"Return_{cfg['exposure']}"
                if target_col not in cols_map:
                    continue
                t_target_idx = cols_map[target_col]
                t_ret_full = test_v[:, t_target_idx]
                
                # Evaluar Regla
                try:
                    r_str = row['Rule']
                    parts = r_str.split()
                    c1, op, c2 = parts
                    
                    # Resolver índices en el Test Set actual
                    idx1 = cols_map.get(c1)
                    if idx1 is None: continue 
                    
                    v1 = test_v[:, idx1]
                    
                    try:
                        val = float(c2)
                        s = (v1 > val) if op == '>' else (v1 < val)
                    except:
                        idx2 = cols_map.get(c2)
                        if idx2 is None: continue
                        v2 = test_v[:, idx2]
                        s = (v1 > v2) if op == '>' else (v1 < v2)
                        
                    n_t = np.sum(s)
                    if n_t > 5: # Filtro mínimo de trades en test
                        test_ret = t_ret_full[s]
                        metrics = miner.calculate_metrics(test_ret)
                        metrics['Rule'] = r_str
                        metrics['Direction'] = direction
                        results_wf.append(metrics)
                        
                except Exception as e:
                    print(f"Error WF: {e}")
                    
                progress_wf.progress((i+1)/len(top_rules))
                
            if results_wf:
                st.subheader("📊 Resultados Walk-Forward (Test)")
                df_wf = pd.DataFrame(results_wf)
                st.dataframe(df_wf.style.format({
                    'total_profit': '{:.2f}', 'profit': '{:.2f}', # Handle variants
                    'pf': '{:.2f}', 'sharpe': '{:.2f}'
                }))
                
                # Comparativa Train vs Test (Robustez)
                st.markdown("### 🆚 Train vs Test (Robustez)")
                
                # Merge con suffixes
                df_compare = pd.merge(
                    top_rules[['Rule', 'Profit', 'Sharpe', 'PF', 'Direction']], 
                    df_wf[['Rule', 'total_profit', 'sharpe', 'pf']], 
                    on='Rule', suffixes=('_Train', '_Test')
                )
                
                # Renombrar para claridad del usuario
                df_compare.rename(columns={
                    'Profit': 'Profit (Train)', 'total_profit': 'Profit (Test)',
                    'Sharpe': 'Sharpe (Train)', 'sharpe': 'Sharpe (Test)',
                    'PF': 'PF (Train)', 'pf': 'PF (Test)'
                }, inplace=True)
                
                # Calcular Métricas de Degradación
                df_compare['Profit_Hold%'] = (df_compare['Profit (Test)'] / df_compare['Profit (Train)']) * 100
                df_compare['Sharpe_Diff'] = df_compare['Sharpe (Test)'] - df_compare['Sharpe (Train)']
                
                # Definir Criterio de Aprobación (PASSED ALL TESTS)
                def check_pass(row):
                    if row['Profit (Test)'] <= 0: return "❌ Negativo en Test"
                    if row['Profit_Hold%'] < 10: return "❌ Colapso Profit (<10%)"
                    if row['Sharpe (Test)'] < 0.05: return "❌ Sharpe Débil"
                    return "✅ PASSED"
                
                df_compare['Verdict'] = df_compare.apply(check_pass, axis=1)
                
                # Reordenar columnas
                cols = ['Verdict', 'Rule', 'Direction', 'Profit (Train)', 'Profit (Test)', 
                        'Sharpe (Train)', 'Sharpe (Test)', 'Profit_Hold%', 'Sharpe_Diff']
                # Filtrar solo col existentes (por si acaso)
                cols = [c for c in cols if c in df_compare.columns]
                df_compare = df_compare[cols]

                # Formato y Orden
                st.dataframe(df_compare.style.format({
                    'Profit (Train)': '{:.2f}', 'Profit (Test)': '{:.2f}',
                    'Sharpe (Train)': '{:.2f}', 'Sharpe (Test)': '{:.2f}',
                    'Profit_Hold%': '{:.1f}%', 'Sharpe_Diff': '{:.2f}'
                }).background_gradient(subset=['Profit_Hold%'], cmap='RdYlGn', vmin=0, vmax=100))
                
            else:
                st.warning("⚠️ No se generaron trades en el periodo de Prueba (Test).")
    else:
        st.info("Sin resultados aún.")
