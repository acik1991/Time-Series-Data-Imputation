import streamlit as st
import pandas as pd
import io
import numpy as np
import plotly.graph_objects as go
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression

# --- 1. Helper Functions ---

@st.cache_data
def load_and_reindex(file, file_ext, date_col, time_col, target_input, is_combined, auto_target, fsm_range=None):
    try:
        # Load
        df = pd.read_csv(file) if file_ext == 'csv' else pd.read_excel(file)
        
        # Auto-Detect Logic
        if auto_target:
            target_col = df.columns[-1]
        else:
            target_col = target_input

        # Parse Dates
        try:
            if is_combined:
                df['Timestamp'] = pd.to_datetime(df[date_col], dayfirst=True)
            else:
                df['Timestamp'] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[time_col].astype(str), dayfirst=True)
        except KeyError:
            return None, f"Column not found! Check your headers. Found: {list(df.columns)}"

        # Clean & Sort
        df = df.drop_duplicates(subset=['Timestamp']).sort_values('Timestamp').set_index('Timestamp')
        
        # Ensure target column is numeric (force non-numeric to NaN)
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')

        # Identify Range
        if fsm_range:
            start_date, end_date = fsm_range
            full_range = pd.date_range(start=f"{start_date} 00:00:00", end=f"{end_date} 23:00:00", freq='h')
        else:
            full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
        
        # Reindex (Create Gaps)
        df_reindexed = df[[target_col]].reindex(full_range)
        df_reindexed.columns = ['Value'] 
        
        # Count actual valid data points (excluding NaNs in original file)
        valid_original_count = df_reindexed['Value'].notna().sum()
        
        return df_reindexed, valid_original_count

    except Exception as e:
        return None, str(e)

def apply_imputation(df, method):
    df_out = df.copy()
    
    if method == "LOCF (Last Observation Carried Forward)":
        df_out['Value'] = df_out['Value'].ffill()
        
    elif method == "Linear Interpolation":
        df_out['Value'] = df_out['Value'].interpolate(method='linear')
        
    elif method == "Cubic Spline":
        try:
            df_out['Value'] = df_out['Value'].interpolate(method='spline', order=3)
        except:
            pass # Fallback or keep NaN
        
    elif method == "Linear Regression":
        df_out['Time_Idx'] = np.arange(len(df_out))
        known = df_out[df_out['Value'].notnull()]
        unknown = df_out[df_out['Value'].isnull()]
        if not unknown.empty and len(known) > 1:
            model = LinearRegression()
            model.fit(known[['Time_Idx']], known['Value'])
            pred = model.predict(unknown[['Time_Idx']])
            df_out.loc[df_out['Value'].isnull(), 'Value'] = pred
        df_out = df_out.drop(columns=['Time_Idx'])

    elif method == "K-Nearest Neighbors (KNN)":
        imputer = KNNImputer(n_neighbors=5)
        df_out['Time_Idx'] = np.arange(len(df_out))
        imputed_array = imputer.fit_transform(df_out[['Time_Idx', 'Value']])
        df_out['Value'] = imputed_array[:, 1]
        df_out = df_out.drop(columns=['Time_Idx'])
        
    return df_out

def convert_to_download(df, output_opt, header_name):
    df_exp = df.reset_index().rename(columns={'index': 'Timestamp', 'Value': header_name})
    if output_opt == "Separate Columns":
        df_exp['Date'] = df_exp['Timestamp'].dt.strftime('%d/%m/%Y')
        df_exp['Time'] = df_exp['Timestamp'].dt.strftime('%H:%M')
        return df_exp[['Date', 'Time', header_name]]
    return df_exp[['Timestamp', header_name]]

# --- 2. Streamlit UI ---

st.set_page_config(page_title="HydroGap Filler", layout="wide")
st.title("🌊 Hydrological Data Processor")

# Sidebar
st.sidebar.header("1. File Configuration")
is_combined = st.sidebar.checkbox("Date & Time in ONE column", value=True)

if is_combined:
    date_input = st.sidebar.text_input("Datetime Header", value="Start of Interval (UTC+08:00)")
    time_input = None
else:
    date_input = st.sidebar.text_input("Date Header", value="Date")
    time_input = st.sidebar.text_input("Time Header", value="Time")

st.sidebar.markdown("---")
auto_target = st.sidebar.checkbox("Auto-detect Target Column", value=True)
if not auto_target:
    target_input = st.sidebar.text_input("Target Header", value="Water Level")
else:
    target_input = None
    st.sidebar.info("Using last column as target.")

st.sidebar.markdown("---")
output_opt = st.sidebar.radio("Download Format", ["Separate Columns", "Combined Column"])
uploaded_files = st.file_uploader("Upload Files", type=['csv', 'xlsx'], accept_multiple_files=True)

tab1, tab2, tab3 = st.tabs(["📅 1. Calendar Completion", "🛠️ 2. Imputation Strategy", "📊 3. Viz & Summary"])

if 'processed_data' not in st.session_state:
    st.session_state.processed_data = {} 

# --- TAB 1 ---
with tab1:
    st.header("Step 1: Create Hourly Skeleton")
    use_fsm_range = st.checkbox("Enable FSM Range")
    fsm_dates = None
    if use_fsm_range:
        c1, c2 = st.columns(2)
        fsm_dates = (c1.date_input("Start"), c2.date_input("End"))

    if st.button("Run Calendar Completion") and uploaded_files:
        st.session_state.processed_data = {} 
        for file in uploaded_files:
            ext = file.name.split('.')[-1]
            df_gaps, orig_count = load_and_reindex(file, ext, date_input, time_input, target_input, is_combined, auto_target, fsm_dates)
            if isinstance(orig_count, str):
                st.error(f"{file.name}: {orig_count}")
            else:
                st.session_state.processed_data[file.name] = {
                    'df_raw_gaps': df_gaps,
                    'original_count': orig_count, # Valid non-nulls in original
                    'df_imputed': None
                }
        st.success("Done! Go to Tab 2.")

# --- TAB 2 ---
with tab2:
    st.header("Step 2: Fill Gaps")
    if st.session_state.processed_data:
        method = st.selectbox("Imputation Method", ["LOCF (Last Observation Carried Forward)", "Linear Interpolation", "Cubic Spline", "Linear Regression", "K-Nearest Neighbors (KNN)"])
        if st.button("Apply Imputation"):
            for fname, data in st.session_state.processed_data.items():
                st.session_state.processed_data[fname]['df_imputed'] = apply_imputation(data['df_raw_gaps'], method)
            st.success(f"Applied {method}!")

        st.write("---")
        for fname, data in st.session_state.processed_data.items():
            if data['df_imputed'] is not None:
                with st.expander(f"Download: {fname}"):
                    disp_name = target_input if target_input else "Water Level"
                    df_out = convert_to_download(data['df_imputed'], output_opt, disp_name)
                    st.download_button("Download CSV", df_out.to_csv(index=False).encode('utf-8'), f"Imputed_{fname}.csv", "text/csv")

# --- TAB 3 (UPDATED) ---
with tab3:
    st.header("Step 3: Visualization & Report")
    if st.session_state.processed_data:
        summary_list = []
        
        for fname, data in st.session_state.processed_data.items():
            if data['df_imputed'] is not None:
                # --- CORRECTED MATH ---
                total_slots = len(data['df_imputed'])
                
                # Count actual values (non-NaN)
                valid_original = data['original_count'] # Calculated in load step
                valid_final = data['df_imputed']['Value'].notna().sum()
                
                # Actually filled = Final Valid - Original Valid
                filled_count = valid_final - valid_original
                
                # Prevent negative numbers if original had dupes we cleaned
                if filled_count < 0: filled_count = 0
                
                # Percentage of the *Total Timeline* that is now valid data
                completeness = (valid_final / total_slots) * 100
                
                summary_list.append({
                    "File Name": fname,
                    "Total Hours (Range)": total_slots,
                    "Original Valid Data": valid_original,
                    "Values Successfully Filled": filled_count,
                    "Final Completeness (%)": round(completeness, 2)
                })

                # --- PLOTLY GRAPH ---
                st.subheader(f"📈 {fname}")
                
                fig = go.Figure()
                
                # 1. Imputed Line (Red) - Background
                fig.add_trace(go.Scatter(
                    x=data['df_imputed'].index, 
                    y=data['df_imputed']['Value'],
                    mode='lines',
                    name='Imputed (Filled)',
                    line=dict(color='red', width=2)
                ))
                
                # 2. Original Data (Blue) - Foreground
                # We filter out NaNs from raw gaps to only show valid original points
                raw_valid = data['df_raw_gaps'].dropna()
                fig.add_trace(go.Scatter(
                    x=raw_valid.index, 
                    y=raw_valid['Value'],
                    mode='markers', # Markers show exactly where real data exists
                    name='Original Data',
                    marker=dict(color='blue', size=4)
                ))
                
                fig.update_layout(title=f"Imputation Results: {fname}", xaxis_title="Date", yaxis_title="Water Level (m)", template="plotly_white")
                
                # Display Interactive Chart
                st.plotly_chart(fig, use_container_width=True)
                
                # Download Button for HTML
                buffer = io.StringIO()
                fig.write_html(buffer, include_plotlyjs='cdn')
                html_bytes = buffer.getvalue().encode()
                
                st.download_button(
                    label="📷 Download Graph (Interactive HTML)",
                    data=html_bytes,
                    file_name=f"Graph_{fname}.html",
                    mime="text/html"
                )

        st.write("---")
        st.subheader("📊 Summary Table")
        st.table(pd.DataFrame(summary_list))
