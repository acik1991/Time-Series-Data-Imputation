import streamlit as st
import pandas as pd
import io
from sklearn.impute import KNNImputer

def process_data(file, file_extension, date_col, time_col, target_col, is_combined, output_format, fill_method, fsm_dates=None):
    try:
        # --- 1. Load Data ---
        if file_extension == 'csv':
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        # --- 2. Create Datetime Index ---
        if is_combined:
            df['Timestamp'] = pd.to_datetime(df[date_col], dayfirst=True)
        else:
            df['Timestamp'] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[time_col].astype(str), dayfirst=True)

        df = df.drop_duplicates(subset=['Timestamp'])
        df = df.sort_values('Timestamp')
        df = df.set_index('Timestamp')
        
        df_subset = df[[target_col]].copy()

        # --- 3. Define the Sequence Range ---
        if fill_method == "FSM (Full Sequence Method)" and fsm_dates:
            start_date, end_date = fsm_dates
            # Create range based on user-defined full sequence
            full_range = pd.date_range(start=start_date, end=end_date, freq='h')
        else:
            # Create range based only on file limits
            full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
        
        df_complete = df_subset.reindex(full_range)

        # --- 4. Imputation Logic ---
        if fill_method == "LOCF (Last Observation Carried Forward)":
            df_complete[target_col] = df_complete[target_col].ffill()
        
        elif fill_method == "Linear Interpolation":
            df_complete[target_col] = df_complete[target_col].interpolate(method='linear')
            
        elif fill_method == "K-Nearest Neighbors (KNN)":
            imputer = KNNImputer(n_neighbors=5)
            df_complete['temp_idx'] = range(len(df_complete))
            imputed_data = imputer.fit_transform(df_complete[['temp_idx', target_col]])
            df_complete[target_col] = imputed_data[:, 1]
            df_complete = df_complete.drop(columns=['temp_idx'])
            
        elif fill_method == "FSM (Full Sequence Method)":
            # For FSM, we often leave data as-is or apply a basic fill. 
            # Here we provide the full sequence; you can add .ffill() if desired.
            pass

        # --- 5. Formatting Output ---
        df_complete = df_complete.reset_index().rename(columns={'index': 'Timestamp'})
        
        if output_format == "Separate Columns (Date, Time)":
            df_complete['Date'] = df_complete['Timestamp'].dt.strftime('%d/%m/%Y')
            df_complete['Time'] = df_complete['Timestamp'].dt.strftime('%H:%M')
            final_df = df_complete[['Date', 'Time', target_col]]
        else:
            df_complete['Datetime'] = df_complete['Timestamp'].dt.strftime('%d/%m/%Y %H:%M')
            final_df = df_complete[['Datetime', target_col]]
        
        return final_df

    except Exception as e:
        st.error(f"Error: {e}")
        return None

# --- UI Setup ---
st.set_page_config(page_title="WL Data Imputation Tool", layout="wide")
st.title("🌊 Water Level Data Imputation Tool")

# Sidebar
st.sidebar.header("⚙️ 1. Column Mapping")
is_combined = st.sidebar.checkbox("Date & Time in ONE column", value=True)

if is_combined:
    date_input = st.sidebar.text_input("Datetime Column", value="Start of Interval (UTC+08:00)")
    time_input = None
else:
    date_input = st.sidebar.text_input("Date Column", value="Date")
    time_input = st.sidebar.text_input("Time Column", value="Time")

target_input = st.sidebar.text_input("Target Column", value="0740231WL_SG. LANAS DI AIR LANAS KELANTAN (5718401)")

st.sidebar.header("🛠️ 2. Imputation Method")
fill_method = st.sidebar.selectbox(
    "Choose method:",
    ["None (Keep Gaps)", "LOCF (Last Observation Carried Forward)", "Linear Interpolation", "FSM (Full Sequence Method)", "K-Nearest Neighbors (KNN)"]
)

# FSM Specific Inputs
fsm_dates = None
if fill_method == "FSM (Full Sequence Method)":
    st.sidebar.info("Define the full sequence bounds:")
    fsm_start = st.sidebar.date_input("Sequence Start Date")
    fsm_end = st.sidebar.date_input("Sequence End Date")
    fsm_dates = (fsm_start, fsm_end)

st.sidebar.header("📤 3. Export Settings")
output_opt = st.sidebar.radio("Output structure:", ["Separate Columns (Date, Time)", "Single Combined Column (Datetime)"])

uploaded_files = st.file_uploader("Upload Files", type=['csv', 'xlsx'], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        ext = uploaded_file.name.split('.')[-1]
        with st.expander(f"📄 {uploaded_file.name}", expanded=True):
            result = process_data(uploaded_file, ext, date_input, time_input, target_input, is_combined, output_opt, fill_method, fsm_dates)
            
            if result is not None:
                st.dataframe(result.head(10), use_container_width=True)
                
                col1, col2 = st.columns(2)
                csv_data = result.to_csv(index=False).encode('utf-8')
                col1.download_button("📥 Download CSV", data=csv_data, file_name=f"Processed_{uploaded_file.name}.csv", key=f"csv_{uploaded_file.name}")
                
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    result.to_excel(writer, index=False)
                col2.download_button("📥 Download Excel", data=excel_buffer.getvalue(), file_name=f"Processed_{uploaded_file.name}.xlsx", key=f"xlsx_{uploaded_file.name}")
