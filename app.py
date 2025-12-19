import streamlit as st
import pandas as pd
import io
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression

# --- 1. Helper Functions ---

@st.cache_data
def load_and_reindex(file, file_ext, date_col, time_col, target_col, is_combined, fsm_range=None):
    """
    Loads data and creates the 'Skeleton' (Calendar with gaps).
    """
    try:
        # Load
        df = pd.read_csv(file) if file_ext == 'csv' else pd.read_excel(file)
        
        # Parse Dates
        if is_combined:
            df['Timestamp'] = pd.to_datetime(df[date_col], dayfirst=True)
        else:
            df['Timestamp'] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[time_col].astype(str), dayfirst=True)

        # Clean & Sort
        df = df.drop_duplicates(subset=['Timestamp']).sort_values('Timestamp').set_index('Timestamp')
        
        # Identify Range
        if fsm_range:
            # FSM: Force specific start/end dates
            start_date, end_date = fsm_range
            full_range = pd.date_range(start=f"{start_date} 00:00:00", end=f"{end_date} 23:00:00", freq='h')
        else:
            # Standard: Use file limits
            full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
        
        # Reindex (Create Gaps)
        # We keep the original data in a column named 'Original' to compare later
        df_reindexed = df[[target_col]].reindex(full_range)
        df_reindexed.columns = ['Value'] # Standardize column name for processing
        
        return df_reindexed, len(df) # Return data and original count
    except Exception as e:
        return None, str(e)

def apply_imputation(df, method):
    """
    Applies the mathematical filling logic.
    """
    df_out = df.copy()
    
    if method == "LOCF (Last Observation Carried Forward)":
        df_out['Value'] = df_out['Value'].ffill()
        
    elif method == "Linear Interpolation":
        df_out['Value'] = df_out['Value'].interpolate(method='linear')
        
    elif method == "Cubic Spline":
        # Requires scipy
        df_out['Value'] = df_out['Value'].interpolate(method='spline', order=3)
        
    elif method == "Linear Regression":
        # Create numeric time index for regression
        df_out['Time_Idx'] = np.arange(len(df_out))
        
        known = df_out[df_out['Value'].notnull()]
        unknown = df_out[df_out['Value'].isnull()]
        
        if not unknown.empty and len(known) > 1:
            model = LinearRegression()
            model.fit(known[['Time_Idx']], known['Value'])
            predicted = model.predict(unknown[['Time_Idx']])
            df_out.loc[df_out['Value'].isnull(), 'Value'] = predicted
            
        df_out = df_out.drop(columns=['Time_Idx'])

    elif method == "K-Nearest Neighbors (KNN)":
        imputer = KNNImputer(n_neighbors=5)
        # KNN needs a 2D array. We use Time Index + Value
        df_out['Time_Idx'] = np.arange(len(df_out))
        imputed_array = imputer.fit_transform(df_out[['Time_Idx', 'Value']])
        df_out['Value'] = imputed_array[:, 1]
        df_out = df_out.drop(columns=['Time_Idx'])
        
    return df_out

def convert_to_download(df, output_opt, target_col_name):
    """
    Formats the dataframe for CSV/Excel download.
    """
    df_export = df.reset_index().rename(columns={'index': 'Timestamp'})
    df_export = df_export.rename(columns={'Value': target_col_name})
    
    if output_opt == "Separate Columns":
        df_export['Date'] = df_export['Timestamp'].dt.strftime('%d/%m/%Y')
        df_export['Time'] = df_export['Timestamp'].dt.strftime('%H:%M')
        return df_export[['Date', 'Time', target_col_name]]
    else:
        df_export['Datetime'] = df_export['Timestamp'].dt.strftime('%d/%m/%Y %H:%M')
        return df_export[['Datetime', target_col_name]]

# --- 2. Streamlit UI Layout ---

st.set_page_config(page_title="HydroGap Filler", layout="wide")
st.title("🌊 Hydrological Data Processor")

# --- SIDEBAR: Global Settings ---
st.sidebar.header("1. File Configuration")
is_combined = st.sidebar.checkbox("Date & Time in ONE column", value=True)

if is_combined:
    date_input = st.sidebar.text_input("Datetime Column", value="Start of Interval (UTC+08:00)")
    time_input = None
else:
    date_input = st.sidebar.text_input("Date Column", value="Date")
    time_input = st.sidebar.text_input("Time Column", value="Time")

target_input = st.sidebar.text_input("Target Value Column", value="Water Level")
output_opt = st.sidebar.radio("Download Format", ["Separate Columns", "Combined Column"])

uploaded_files = st.file_uploader("Upload Files (CSV/Excel)", type=['csv', 'xlsx'], accept_multiple_files=True)

# --- MAIN TABS ---
tab1, tab2, tab3 = st.tabs(["📅 1. Calendar Completion", "🛠️ 2. Imputation Strategy", "📊 3. Viz & Summary"])

# Initialize Session State to hold processed data
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = {} 
    # Structure: {filename: {'df_raw_gaps': df, 'original_count': int, 'df_imputed': df}}

# --- TAB 1: CALENDAR COMPLETION ---
with tab1:
    st.header("Step 1: Standardize Dates")
    st.info("This step identifies missing hours and inserts gaps (NaN). No values are invented yet.")
    
    use_fsm_range = st.checkbox("Enable FSM Range (Force specific Start/End dates for all files)")
    fsm_dates = None
    if use_fsm_range:
        col_d1, col_d2 = st.columns(2)
        start_d = col_d1.date_input("Start Date")
        end_d = col_d2.date_input("End Date")
        fsm_dates = (start_d, end_d)

    if st.button("Run Calendar Completion") and uploaded_files:
        st.session_state.processed_data = {} # Reset
        
        for file in uploaded_files:
            ext = file.name.split('.')[-1]
            df_gaps, orig_count = load_and_reindex(file, ext, date_input, time_input, target_input, is_combined, fsm_dates)
            
            if isinstance(orig_count, str): # Error caught
                st.error(f"Error in {file.name}: {orig_count}")
            else:
                # Save to session state
                st.session_state.processed_data[file.name] = {
                    'df_raw_gaps': df_gaps,
                    'original_count': orig_count,
                    'df_imputed': None # Not imputed yet
                }
        st.success("Calendar completion done! You can now download raw data with gaps or move to Tab 2.")

    # Display Results for Tab 1
    if st.session_state.processed_data:
        st.write("---")
        for fname, data in st.session_state.processed_data.items():
            with st.expander(f"View Gaps: {fname}"):
                df_show = convert_to_download(data['df_raw_gaps'], output_opt, target_input)
                st.dataframe(df_show.head(10), use_container_width=True)
                
                # Download Button for Gaps Data
                csv = df_show.to_csv(index=False).encode('utf-8')
                st.download_button(f"Download (With Gaps) - {fname}", csv, f"Gaps_{fname}.csv", "text/csv")

# --- TAB 2: IMPUTATION ---
with tab2:
    st.header("Step 2: Fill Missing Values")
    
    if not st.session_state.processed_data:
        st.warning("Please run 'Calendar Completion' in Tab 1 first.")
    else:
        method = st.selectbox("Select Imputation Method", 
                              ["LOCF (Last Observation Carried Forward)", 
                               "Linear Interpolation", 
                               "Cubic Spline", 
                               "Linear Regression", 
                               "K-Nearest Neighbors (KNN)"])
        
        if st.button("Apply Imputation"):
            for fname, data in st.session_state.processed_data.items():
                # Apply math to the 'df_raw_gaps'
                df_filled = apply_imputation(data['df_raw_gaps'], method)
                # Store back in session state
                st.session_state.processed_data[fname]['df_imputed'] = df_filled
            st.success(f"Applied {method} to all files!")

        # Display Results for Tab 2
        st.write("---")
        for fname, data in st.session_state.processed_data.items():
            if data['df_imputed'] is not None:
                with st.expander(f"View Imputed: {fname}"):
                    df_show = convert_to_download(data['df_imputed'], output_opt, target_input)
                    st.dataframe(df_show.head(10), use_container_width=True)
                    
                    # Excel Download Logic
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                        df_show.to_excel(writer, index=False)
                    
                    col1, col2 = st.columns(2)
                    st.download_button(f"📥 Download CSV", df_show.to_csv(index=False).encode('utf-8'), f"Imputed_{fname}.csv", "text/csv", key=f"dl_csv_{fname}")
                    st.download_button(f"📥 Download Excel", excel_buffer.getvalue(), f"Imputed_{fname}.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"dl_xls_{fname}")

# --- TAB 3: VISUALIZATION & SUMMARY ---
with tab3:
    st.header("Step 3: Quality Report")
    
    if not st.session_state.processed_data:
        st.warning("Please process data in previous tabs first.")
    else:
        summary_list = []
        
        for fname, data in st.session_state.processed_data.items():
            if data['df_imputed'] is not None:
                # 1. Calculate Stats
                total_rows = len(data['df_imputed'])
                orig_rows = data['original_count']
                filled_count = total_rows - orig_rows
                
                summary_list.append({
                    "File Name": fname,
                    "Total Expected": total_rows,
                    "Original Data Points": orig_rows,
                    "Gaps Filled": filled_count,
                    "Fill %": round((filled_count/total_rows)*100, 2)
                })
                
                # 2. Visualization
                st.subheader(f"📈 {fname}")
                
                # Combine raw and imputed for plotting
                # We plot the Imputed line first (as background) and Raw points on top
                chart_data = pd.DataFrame({
                    "Imputed/Filled": data['df_imputed']['Value'],
                    "Original Data": data['df_raw_gaps']['Value'] # This has NaNs where gaps were
                })
                
                st.line_chart(chart_data, color=["#FF4B4B", "#0000FF"]) 
                # Note: Blue (Original) will break at gaps, Red (Imputed) will connect them.
        
        # 3. Summary Table
        st.write("---")
        st.subheader("📊 Global Summary Table")
        if summary_list:
            df_summary = pd.DataFrame(summary_list)
            st.table(df_summary)
            
            # Download Summary
            st.download_button("Download Summary Report", df_summary.to_csv(index=False).encode('utf-8'), "Summary_Report.csv", "text/csv")
