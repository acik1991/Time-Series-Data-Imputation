import streamlit as st
import pandas as pd
import io
import os
import numpy as np
import plotly.graph_objects as go
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression

# --- 1. Helper Functions ---

def get_file_headers(uploaded_file):
    """
    Reads just the first few rows to extract column names efficiently.
    """
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
        # Load
        df = pd.read_csv(file) if file_ext == 'csv' else pd.read_excel(file)
        
        # Parse Dates
        try:
            if is_combined:
                df['Timestamp'] = pd.to_datetime(df[date_col], dayfirst=True)
            else:
                df['Timestamp'] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[time_col].astype(str), dayfirst=True)
        except KeyError:
            return None, f"Column mismatch! The file {file.name} does not have the same columns as the first file."

        # Clean & Sort
        df = df.drop_duplicates(subset=['Timestamp']).sort_values('Timestamp').set_index('Timestamp')
        
        # Ensure target column is numeric
        if target_col not in df.columns:
            return None, f"Target column '{target_col}' not found in {file.name}."
            
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')

        # Identify Range
        if fsm_range:
            start_date, end_date = fsm_range
            full_range = pd.date_range(start=f"{start_date} 00:00:00", end=f"{end_date} 23:00:00", freq='h')
        else:
            full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
        
        # Reindex
        df_reindexed = df[[target_col]].reindex(full_range)
        df_reindexed.columns = ['Value'] 
        
        # Count actual valid data points
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

# --- SIDEBAR: Upload First, Then Configure ---
st.sidebar.header("1. Upload Files")
uploaded_files = st.sidebar.file_uploader("Select CSV or Excel files", type=['csv', 'xlsx'], accept_multiple_files=True)

# Default variables
date_col = None
time_col = None
target_col = None

if uploaded_files:
    # --- DYNAMIC COLUMN CONFIGURATION ---
    st.sidebar.header("2. Column Mapping")
    st.sidebar.info("Detected columns from the first file.")
    
    # Read headers from the first file to populate dropdowns
    first_file = uploaded_files[0]
    first_file.seek(0) # Reset buffer before reading
    columns = get_file_headers(first_file)
    first_file.seek(0) # Reset buffer again for later processing

    if columns:
        is_combined = st.sidebar.checkbox("Date & Time in ONE column", value=True)
        
        if is_combined:
            # Try to auto-select a column that looks like 'Date'
            default_ix = next((i for i, c in enumerate(columns) if 'date' in c.lower() or 'time' in c.lower()), 0)
            date_col = st.sidebar.selectbox("Select Datetime Column", columns, index=default_ix)
        else:
            d_ix = next((i for i, c in enumerate(columns) if 'date' in c.lower()), 0)
            t_ix = next((i for i, c in enumerate(columns) if 'time' in c.lower()), 1)
            date_col = st.sidebar.selectbox("Select Date Column", columns, index=d_ix)
            time_col = st.sidebar.selectbox("Select Time Column", columns, index=t_ix)
        
        # Target Column Selection
        st.sidebar.markdown("---")
        # Default to the last column as it's usually the value
        target_col = st.sidebar.selectbox("Select Target Value Column", columns, index=len(columns)-1)
        
        st.sidebar.markdown("---")
        output_opt = st.sidebar.radio("Download Format", ["Separate Columns", "Combined Column"])
    else:
        st.error("Could not read columns from the first file.")

else:
    st.info("👈 Please upload files in the sidebar to begin.")

# --- MAIN TABS ---
tab1, tab2, tab3 = st.tabs(["📅 1. Calendar Completion", "🛠️ 2. Imputation Strategy", "📊 3. Viz & Summary"])

if 'processed_data' not in st.session_state:
    st.session_state.processed_data = {} 

# --- TAB 1 ---
with tab1:
    st.header("Step 1: Create Hourly Skeleton")
    
    if uploaded_files and target_col:
        use_fsm_range = st.checkbox("Enable FSM Range")
        fsm_dates = None
        if use_fsm_range:
            c1, c2 = st.columns(2)
            fsm_dates = (c1.date_input("Start"), c2.date_input("End"))

        if st.button("Run Calendar Completion"):
            st.session_state.processed_data = {} 
            for file in uploaded_files:
                file.seek(0) # IMPORTANT: Reset buffer for pandas read
                ext = file.name.split('.')[-1]
                
                df_gaps, orig_count = load_and_reindex(file, ext, date_col, time_col, target_col, is_combined, fsm_dates)
                
                if isinstance(orig_count, str):
                    st.error(f"{file.name}: {orig_count}")
                else:
                    st.session_state.processed_data[file.name] = {
                        'df_raw_gaps': df_gaps,
                        'original_count': orig_count,
                        'df_imputed': None
                    }
            st.success("Done! Go to Tab 2.")
            
        # Preview
        if st.session_state.processed_data:
            st.write("---")
            for fname, data in st.session_state.processed_data.items():
                clean_name = os.path.splitext(fname)[0] 
                
                with st.expander(f"View Gaps: {fname}"):
                    df_show = convert_to_download(data['df_raw_gaps'], output_opt, target_col)
                    st.dataframe(df_show.head(10), use_container_width=True)
                    
                    c1, c2 = st.columns(2)
                    c1.download_button("📥 Download CSV", df_show.to_csv(index=False).encode('utf-8'), f"Gaps_{clean_name}.csv", "text/csv")
                    
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine='xlsxwriter') as writer: df_show.to_excel(writer, index=False)
                    c2.download_button("📥 Download Excel", buf.getvalue(), f"Gaps_{clean_name}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    elif not uploaded_files:
        st.warning("Upload files to see options.")

# --- TAB 2 ---
with tab2:
    st.header("Step 2: Fill Gaps")
    if st.session_state.processed_data:
        method = st.selectbox("Imputation Method", ["LOCF (Last Observation Carried Forward)", "Linear Interpolation", "Cubic Spline", "Linear Regression", "K-Nearest Neighbors (KNN)"])
        if st.button("Apply Imputation"):
            for fname, data in st.session_state.processed_data.items():
                st.session_state.processed_data[fname]['df_imputed'] = apply_imputation(data['df_raw_gaps'], method)
            st.success(f"Applied {method}!")

        if any(d['df_imputed'] is not None for d in st.session_state.processed_data.values()):
            st.write("---")
            for fname, data in st.session_state.processed_data.items():
                if data['df_imputed'] is not None:
                    clean_name = os.path.splitext(fname)[0]

                    with st.expander(f"Download: {fname}"):
                        df_out = convert_to_download(data['df_imputed'], output_opt, target_col)
                        
                        col1, col2 = st.columns(2)
                        col1.download_button("📥 Download CSV", df_out.to_csv(index=False).encode('utf-8'), f"Imputed_{clean_name}.csv", "text/csv", key=f"c_{fname}")
                        
                        buf = io.BytesIO()
                        with pd.ExcelWriter(buf, engine='xlsxwriter') as writer: df_out.to_excel(writer, index=False)
                        col2.download_button("📥 Download Excel", buf.getvalue(), f"Imputed_{clean_name}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"x_{fname}")

# --- TAB 3 ---
with tab3:
    st.header("Step 3: Visualization & Report")
    if st.session_state.processed_data:
        summary_list = []
        for fname, data in st.session_state.processed_data.items():
            if data['df_imputed'] is not None:
                clean_name = os.path.splitext(fname)[0]
                
                total = len(data['df_imputed'])
                valid_orig = data['original_count']
                valid_final = data['df_imputed']['Value'].notna().sum()
                filled = valid_final - valid_orig
                if filled < 0: filled = 0
                
                summary_list.append({
                    "File Name": fname,
                    "Total Hours": total,
                    "Original Data": valid_orig,
                    "Filled": filled,
                    "Completeness (%)": round((valid_final/total)*100, 2)
                })

                st.subheader(f"📈 {fname}")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data['df_imputed'].index, y=data['df_imputed']['Value'], mode='lines', name='Imputed', line=dict(color='red')))
                raw_valid = data['df_raw_gaps'].dropna()
                fig.add_trace(go.Scatter(x=raw_valid.index, y=raw_valid['Value'], mode='markers', name='Original', marker=dict(color='blue', size=4)))
                fig.update_layout(title=fname, height=400, template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                
                buf = io.StringIO()
                fig.write_html(buf, include_plotlyjs='cdn')
                st.download_button("📷 Download Graph (HTML)", buf.getvalue().encode(), f"Graph_{clean_name}.html", "text/html")

        if summary_list:
            st.write("---")
            st.subheader("📊 Global Summary Table")
            df_summary = pd.DataFrame(summary_list)
            st.table(df_summary)
            
            # --- NEW: DOWNLOAD BUTTONS FOR SUMMARY ---
            col1, col2 = st.columns(2)
            
            # CSV Download
            csv = df_summary.to_csv(index=False).encode('utf-8')
            col1.download_button(
                label="📥 Download Summary (CSV)",
                data=csv,
                file_name="Summary_Report.csv",
                mime="text/csv"
            )
            
            # Excel Download
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
                df_summary.to_excel(writer, index=False)
            col2.download_button(
                label="📥 Download Summary (Excel)",
                data=buf.getvalue(),
                file_name="Summary_Report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
