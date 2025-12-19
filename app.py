import streamlit as st
import pandas as pd
import io
import os
import numpy as np
import plotly.graph_objects as go
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression

# ==========================================
# 1. HELPER FUNCTIONS (General)
# ==========================================

def get_file_headers(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, nrows=0)
        else:
            df = pd.read_excel(uploaded_file, nrows=0)
        return list(df.columns)
    except Exception as e:
        st.error(f"Error reading headers: {e}")
        return []

@st.cache_data
def load_and_reindex(file, file_ext, date_col, time_col, target_col, is_combined, fsm_range=None):
    try:
        df = pd.read_csv(file) if file_ext == 'csv' else pd.read_excel(file)
        
        # Parse Dates
        try:
            if is_combined:
                df['Timestamp'] = pd.to_datetime(df[date_col], dayfirst=True)
            else:
                df['Timestamp'] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[time_col].astype(str), dayfirst=True)
        except KeyError:
            return None, f"Column mismatch in {file.name}"

        # Clean & Sort
        df = df.drop_duplicates(subset=['Timestamp']).sort_values('Timestamp').set_index('Timestamp')
        
        # Numeric Check
        if target_col not in df.columns:
            return None, f"Target '{target_col}' not found."
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')

        # Reindexing Range
        if fsm_range:
            start_date, end_date = fsm_range
            full_range = pd.date_range(start=f"{start_date} 00:00:00", end=f"{end_date} 23:00:00", freq='h')
        else:
            full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
        
        # Create Skeleton
        df_reindexed = df[[target_col]].reindex(full_range)
        df_reindexed.columns = ['Value'] 
        valid_original_count = df_reindexed['Value'].notna().sum()
        
        return df_reindexed, valid_original_count

    except Exception as e:
        return None, str(e)

def convert_to_download(df, output_opt, header_name):
    df_exp = df.reset_index().rename(columns={'index': 'Timestamp', 'Value': header_name})
    if output_opt == "Separate Columns":
        df_exp['Date'] = df_exp['Timestamp'].dt.strftime('%d/%m/%Y')
        df_exp['Time'] = df_exp['Timestamp'].dt.strftime('%H:%M')
        return df_exp[['Date', 'Time', header_name]]
    return df_exp[['Timestamp', header_name]]

# ==========================================
# 2. STANDARD IMPUTATION (Tab 2)
# ==========================================

def apply_standard_imputation(df, method):
    df_out = df.copy()
    if method == "LOCF (Last Observation Carried Forward)":
        df_out['Value'] = df_out['Value'].ffill()
    elif method == "Linear Interpolation":
        df_out['Value'] = df_out['Value'].interpolate(method='linear')
    elif method == "Cubic Spline":
        try: df_out['Value'] = df_out['Value'].interpolate(method='spline', order=3)
        except: pass 
    elif method == "Linear Regression":
        df_out['Time_Idx'] = np.arange(len(df_out))
        known = df_out[df_out['Value'].notnull()]
        unknown = df_out[df_out['Value'].isnull()]
        if not unknown.empty and len(known) > 1:
            model = LinearRegression()
            model.fit(known[['Time_Idx']], known['Value'])
            df_out.loc[df_out['Value'].isnull(), 'Value'] = model.predict(unknown[['Time_Idx']])
        df_out = df_out.drop(columns=['Time_Idx'])
    elif method == "K-Nearest Neighbors (KNN)":
        imputer = KNNImputer(n_neighbors=5)
        df_out['Time_Idx'] = np.arange(len(df_out))
        imp = imputer.fit_transform(df_out[['Time_Idx', 'Value']])
        df_out['Value'] = imp[:, 1]
        df_out = df_out.drop(columns=['Time_Idx'])
    return df_out

# ==========================================
# 3. ADVANCED FSM IMPUTATION (Tab 4)
# ==========================================

def find_na_gaps(x: pd.Series):
    """Find consecutive NaN runs."""
    is_na = x.isna().to_numpy()
    n = len(is_na)
    gaps = []
    in_gap = False
    start = None
    for i in range(n):
        if is_na[i] and not in_gap:
            in_gap = True
            start = i
        elif (not is_na[i]) and in_gap:
            in_gap = False
            gaps.append((start, i - 1))
    if in_gap:
        gaps.append((start, n - 1))
    return gaps

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.sqrt(np.sum(diff * diff)))

def fsm_find_best_match(values, gap_start, gap_end, m, n, const_c=0.0, max_candidates=None):
    N = len(values)
    gap_len = gap_end - gap_start + 1
    left_start = max(0, gap_start - m)
    left_len = gap_start - left_start
    right_end = min(N - 1, gap_end + n)
    right_len = right_end - gap_end

    if left_len == 0 and right_len == 0:
        return None, 0, gap_len, 0

    # Build Query Sequence I
    I_parts = []
    if left_len > 0: I_parts.append(values[left_start:gap_start])
    I_parts.append(np.full(gap_len, const_c, dtype=float))
    if right_len > 0: I_parts.append(values[gap_end + 1:right_end + 1])
    I_full = np.concatenate(I_parts).astype(float)
    z = len(I_full)

    valid_mask = ~np.isnan(I_full)
    if not np.any(valid_mask): return None, left_len, gap_len, right_len
    I_valid = I_full[valid_mask]

    # Search Candidates
    best_dist = np.inf
    best_S = None
    indices = np.arange(0, N - z + 1)
    
    # Optimization: Random subsample if too large
    if max_candidates is not None and len(indices) > max_candidates:
        rng = np.random.default_rng(42)
        indices = np.sort(rng.choice(indices, size=max_candidates, replace=False))

    for start in indices:
        end = start + z - 1
        if not (end < gap_start or start > gap_end): continue # Skip overlap

        S_window = values[start:end + 1].astype(float)
        if np.any(np.isnan(S_window[valid_mask])): continue

        S_for_dist = S_window.copy()
        gap_pos_in_I = np.zeros(z, dtype=bool)
        gap_pos_in_I[left_len:left_len + gap_len] = True
        S_for_dist[gap_pos_in_I] = const_c # Mask gap area in candidate

        d = euclidean_distance(I_valid, S_for_dist[valid_mask])
        if d < best_dist:
            best_dist = d
            best_S = S_window
            if best_dist == 0: break

    return best_S, left_len, gap_len, right_len

def fsm_impute_gap_scale(values, gap_start, gap_end, S_window, left_len, gap_len, right_len):
    x = values.copy()
    parts = []
    if left_len > 0: parts.append(x[gap_start - left_len:gap_start])
    parts.append(np.full(gap_len, np.nan))
    if right_len > 0: parts.append(x[gap_end + 1:gap_end + 1 + right_len])
    
    I_full = np.concatenate(parts).astype(float)
    known_mask = ~np.isnan(I_full)
    if not np.any(known_mask): return None

    query_known = I_full[known_mask]
    S_known = S_window[known_mask]
    
    q_min, q_max = np.nanmin(query_known), np.nanmax(query_known)
    s_min, s_max = np.nanmin(S_known), np.nanmax(S_known)

    if np.isclose(s_max - s_min, 0.0): scale = 1.0
    else: scale = (q_max - q_min) / (s_max - s_min)
    
    shift = q_min - s_min * scale
    S_scaled = S_window * scale + shift
    
    x[gap_start:gap_end + 1] = S_scaled[left_len:left_len + gap_len]
    return x

def fsm_impute_gap_diff(values, gap_start, gap_end, S_window, left_len, gap_len, right_len):
    x = values.copy()
    if left_len > 0:
        prev_idx = gap_start - 1
        if np.isnan(x[prev_idx]): return None
        current = x[prev_idx]
        s_pos = left_len
        for k in range(gap_len):
            diff = S_window[s_pos + k] - S_window[s_pos + k - 1]
            current += diff
            x[gap_start + k] = current
        return x
    return None

def apply_fsm_series(series, mode="FSM_scale", m_factor=1.0, max_candidates=10000):
    x = series.astype(float).to_numpy()
    gaps = find_na_gaps(series)
    
    # Progress placeholder
    prog_text = st.empty()
    
    for i, (start, end) in enumerate(gaps):
        gap_len = end - start + 1
        m = max(1, int(m_factor * gap_len))
        
        if i % 5 == 0: prog_text.text(f"Processing gap {i+1}/{len(gaps)} (len={gap_len})...")
        
        S_window, left_len, g_len, right_len = fsm_find_best_match(
            x, start, end, m, m, const_c=0.0, max_candidates=max_candidates
        )
        
        if S_window is not None:
            if mode == "FSM_scale":
                x_new = fsm_impute_gap_scale(x, start, end, S_window, left_len, g_len, right_len)
            else:
                x_new = fsm_impute_gap_diff(x, start, end, S_window, left_len, g_len, right_len)
            
            if x_new is not None:
                x = x_new
    
    prog_text.empty()
    return pd.Series(x, index=series.index, name=series.name)

# ==========================================
# 4. MAIN STREAMLIT APP
# ==========================================

st.set_page_config(page_title="HydroGap Filler", layout="wide")
st.title("🌊 Hydrological Data Processor")

# --- SIDEBAR ---
st.sidebar.header("1. Upload Files")
uploaded_files = st.sidebar.file_uploader("Select CSV or Excel files", type=['csv', 'xlsx'], accept_multiple_files=True)

date_col, time_col, target_col = None, None, None

if uploaded_files:
    st.sidebar.header("2. Column Mapping")
    first_file = uploaded_files[0]
    first_file.seek(0)
    columns = get_file_headers(first_file)
    first_file.seek(0)

    if columns:
        is_combined = st.sidebar.checkbox("Date & Time in ONE column", value=True)
        if is_combined:
            default_ix = next((i for i, c in enumerate(columns) if 'date' in c.lower() or 'time' in c.lower()), 0)
            date_col = st.sidebar.selectbox("Select Datetime Column", columns, index=default_ix)
        else:
            d_ix = next((i for i, c in enumerate(columns) if 'date' in c.lower()), 0)
            t_ix = next((i for i, c in enumerate(columns) if 'time' in c.lower()), 1)
            date_col = st.sidebar.selectbox("Select Date Column", columns, index=d_ix)
            time_col = st.sidebar.selectbox("Select Time Column", columns, index=t_ix)
        
        st.sidebar.markdown("---")
        target_col = st.sidebar.selectbox("Select Target Value Column", columns, index=len(columns)-1)
        st.sidebar.markdown("---")
        output_opt = st.sidebar.radio("Download Format", ["Separate Columns", "Combined Column"])
    else:
        st.error("Could not read columns from the first file.")
else:
    st.info("👈 Please upload files in the sidebar to begin.")

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📅 1. Calendar Completion", 
    "🛠️ 2. Std Imputation", 
    "📊 3. Viz & Summary",
    "🤖 4. Advanced FSM"
])

if 'processed_data' not in st.session_state:
    st.session_state.processed_data = {} 

# --- TAB 1: SKELETON ---
with tab1:
    st.header("Step 1: Create Hourly Skeleton")
    if uploaded_files and target_col:
        use_fsm_range = st.checkbox("Enable FSM Range (Force Start/End)")
        fsm_dates = None
        if use_fsm_range:
            c1, c2 = st.columns(2)
            fsm_dates = (c1.date_input("Start"), c2.date_input("End"))

        if st.button("Run Calendar Completion"):
            st.session_state.processed_data = {} 
            for file in uploaded_files:
                file.seek(0)
                ext = file.name.split('.')[-1]
                df_gaps, orig_count = load_and_reindex(file, ext, date_col, time_col, target_col, is_combined, fsm_dates)
                if isinstance(orig_count, str):
                    st.error(f"{file.name}: {orig_count}")
                else:
                    st.session_state.processed_data[file.name] = {
                        'df_raw_gaps': df_gaps,
                        'original_count': orig_count,
                        'df_imputed': None,
                        'df_fsm': None  # Store FSM result separately
                    }
            st.success("Calendar completion done!")

        if st.session_state.processed_data:
            st.write("---")
            for fname, data in st.session_state.processed_data.items():
                with st.expander(f"Preview: {fname}"):
                    st.dataframe(data['df_raw_gaps'].head(5))

# --- TAB 2: STANDARD IMPUTATION ---
with tab2:
    st.header("Step 2: Mathematical Imputation")
    st.info("Use this tab for standard methods (Linear, Spline, KNN). For FSM, go to Tab 4.")
    
    if st.session_state.processed_data:
        method = st.selectbox("Method", ["Linear Interpolation", "LOCF (Last Observation Carried Forward)", "Cubic Spline", "Linear Regression", "K-Nearest Neighbors (KNN)"])
        
        if st.button("Apply Standard Imputation"):
            for fname, data in st.session_state.processed_data.items():
                st.session_state.processed_data[fname]['df_imputed'] = apply_standard_imputation(data['df_raw_gaps'], method)
            st.success(f"Applied {method}!")

        if any(d['df_imputed'] is not None for d in st.session_state.processed_data.values()):
             for fname, data in st.session_state.processed_data.items():
                if data['df_imputed'] is not None:
                    with st.expander(f"Download Standard: {fname}"):
                        df_out = convert_to_download(data['df_imputed'], output_opt, target_col)
                        c1, c2 = st.columns(2)
                        c1.download_button("📥 CSV", df_out.to_csv(index=False).encode('utf-8'), f"Std_Imputed_{fname}.csv", "text/csv")
                        buf = io.BytesIO()
                        with pd.ExcelWriter(buf, engine='xlsxwriter') as writer: df_out.to_excel(writer, index=False)
                        c2.download_button("📥 Excel", buf.getvalue(), f"Std_Imputed_{fname}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --- TAB 3: VIZ (Standard) ---
with tab3:
    st.header("Step 3: Quality Report (Standard)")
    if st.session_state.processed_data:
        for fname, data in st.session_state.processed_data.items():
            if data['df_imputed'] is not None:
                st.subheader(f"📈 {fname}")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data['df_imputed'].index, y=data['df_imputed']['Value'], mode='lines', name='Imputed', line=dict(color='red')))
                raw_valid = data['df_raw_gaps'].dropna()
                fig.add_trace(go.Scatter(x=raw_valid.index, y=raw_valid['Value'], mode='markers', name='Original', marker=dict(color='blue', size=4)))
                st.plotly_chart(fig, use_container_width=True)

# --- TAB 4: ADVANCED FSM ---
with tab4:
    st.header("🤖 Step 4: Full Subsequence Matching (FSM)")
    st.info("Finds historical patterns similar to the gap context and copies them.")
    
    if not st.session_state.processed_data:
        st.warning("Please run Step 1 (Calendar Completion) first.")
    else:
        c1, c2, c3 = st.columns(3)
        fsm_mode = c1.selectbox("FSM Mode", ["FSM_scale (Recommended)", "FSM_diff"], help="Scale: Adjusts pattern magnitude. Diff: Adjusts pattern trend.")
        m_factor = c2.number_input("Context Factor", 0.5, 5.0, 1.0, 0.1, help="1.0 means context window = gap length.")
        max_cands = c3.number_input("Max Candidates", 1000, 100000, 5000, 1000, help="Limit search space for speed.")

        if st.button("Run FSM Imputation"):
            for fname, data in st.session_state.processed_data.items():
                st.write(f"Processing **{fname}**...")
                # Run FSM
                df_fsm = apply_fsm_series(
                    data['df_raw_gaps']['Value'], 
                    mode=fsm_mode.split()[0], 
                    m_factor=m_factor, 
                    max_candidates=int(max_cands)
                )
                # Store Result in new key
                st.session_state.processed_data[fname]['df_fsm'] = df_fsm.to_frame(name='Value')
            st.success("FSM Imputation Complete!")

        # Display Results
        st.write("---")
        for fname, data in st.session_state.processed_data.items():
            if data['df_fsm'] is not None:
                st.subheader(f"📊 Results: {fname}")
                
                # 1. Downloads
                clean_name = os.path.splitext(fname)[0]
                df_out = convert_to_download(data['df_fsm'], output_opt, target_col)
                
                c1, c2 = st.columns(2)
                c1.download_button("📥 Download FSM Data (CSV)", df_out.to_csv(index=False).encode('utf-8'), f"FSM_{clean_name}.csv", "text/csv")
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine='xlsxwriter') as writer: df_out.to_excel(writer, index=False)
                c2.download_button("📥 Download FSM Data (Excel)", buf.getvalue(), f"FSM_{clean_name}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                
                # 2. Time Series Plot (Original vs FSM)
                fig_ts = go.Figure()
                fig_ts.add_trace(go.Scatter(x=data['df_fsm'].index, y=data['df_fsm']['Value'], mode='lines', name='FSM Imputed', line=dict(color='red', dash='dot')))
                raw_valid = data['df_raw_gaps'].dropna()
                fig_ts.add_trace(go.Scatter(x=raw_valid.index, y=raw_valid['Value'], mode='lines+markers', name='Original', marker=dict(color='blue', size=3)))
                fig_ts.update_layout(title=f"Time Series: Original vs FSM ({fname})", height=400, hovermode='x unified')
                st.plotly_chart(fig_ts, use_container_width=True)
                
                # HTML Download for TS Plot
                buf_ts = io.StringIO()
                fig_ts.write_html(buf_ts, include_plotlyjs='cdn')
                st.download_button("📷 Download Time Series Plot (HTML)", buf_ts.getvalue().encode(), f"Plot_TS_{clean_name}.html", "text/html")
                
                # 3. Seasonality Plot (Original vs FSM)
                try:
                    df_orig_plot = data['df_raw_gaps'].copy()
                    df_fsm_plot = data['df_fsm'].copy()
                    
                    df_orig_plot['Month'] = df_orig_plot.index.month
                    df_fsm_plot['Month'] = df_fsm_plot.index.month
                    
                    orig_seas = df_orig_plot.groupby('Month')['Value'].mean()
                    fsm_seas = df_fsm_plot.groupby('Month')['Value'].mean()
                    
                    fig_seas = go.Figure()
                    fig_seas.add_trace(go.Scatter(x=orig_seas.index, y=orig_seas.values, mode='lines+markers', name='Original Seasonality', line=dict(color='blue')))
                    fig_seas.add_trace(go.Scatter(x=fsm_seas.index, y=fsm_seas.values, mode='lines+markers', name='FSM Seasonality', line=dict(color='red', dash='dash')))
                    
                    fig_seas.update_layout(
                        title=f"Monthly Seasonality Check ({fname})", 
                        xaxis=dict(tickmode='array', tickvals=list(range(1,13)), ticktext=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']),
                        xaxis_title="Month", yaxis_title="Average Level"
                    )
                    st.plotly_chart(fig_seas, use_container_width=True)
                    
                    # HTML Download for Seasonality Plot
                    buf_seas = io.StringIO()
                    fig_seas.write_html(buf_seas, include_plotlyjs='cdn')
                    st.download_button("📷 Download Seasonality Plot (HTML)", buf_seas.getvalue().encode(), f"Plot_Seasonality_{clean_name}.html", "text/html")
                except Exception as e:
                    st.warning(f"Could not generate seasonality plot: {e}")
                
                st.markdown("---")
