import streamlit as st
import pandas as pd
import io

def process_data(file, file_extension, date_col, time_col, target_col):
    try:
        # --- 1. Load Data ---
        if file_extension == 'csv':
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        # --- 2. Create Datetime Index ---
        # Combine user-selected date and time columns
        df['Timestamp'] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[time_col].astype(str))
        df = df.drop_duplicates(subset=['Timestamp'])
        df = df.sort_values('Timestamp')
        df = df.set_index('Timestamp')
        
        # Keep only the target data column
        df_subset = df[[target_col]]

        # --- 3. Fill Missing Timestamps (Hourly) ---
        full_range = pd.date_range(
            start=df.index.min(),
            end=df.index.max(),
            freq='h'
        )
        df_complete = df_subset.reindex(full_range)

        # --- 4. Re-format for Export ---
        df_complete = df_complete.reset_index()
        df_complete = df_complete.rename(columns={'index': 'Timestamp'})
        df_complete['Date'] = df_complete['Timestamp'].dt.strftime('%d/%m/%Y')
        df_complete['Time'] = df_complete['Timestamp'].dt.strftime('%H:%M')
        
        # Final column order
        df_complete = df_complete[['Date', 'Time', target_col]]
        return df_complete
    except Exception as e:
        st.error(f"Processing error: {e}")
        return None

# --- Streamlit UI ---
st.set_page_config(page_title="Data Completer", layout="wide")
st.title("📊 Custom Water Level Data Completer")

# Sidebar for column configuration
st.sidebar.header("Column Settings")
date_input = st.sidebar.text_input("Date Column Name", value="Date")
time_input = st.sidebar.text_input("Time Column Name", value="Time")
target_input = st.sidebar.text_input("Target Data Column", value="Water Level (m) - Raw")

st.info("💡 Tip: Ensure the column names in the sidebar match your file exactly (case-sensitive).")

uploaded_files = st.file_uploader(
    "Upload CSV or Excel files", 
    type=['csv', 'xlsx'], 
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        extension = uploaded_file.name.split('.')[-1]
        
        with st.expander(f"📁 {uploaded_file.name}", expanded=True):
            # Process the data with user inputs
            df_result = process_data(uploaded_file, extension, date_input, time_input, target_input)
            
            if df_result is not None:
                st.success(f"Successfully processed! Total rows: {len(df_result)}")
                
                # Show preview of processed data
                st.dataframe(df_result.head(10), use_container_width=True)
                
                # --- Download Section ---
                col1, col2 = st.columns(2)
                
                # CSV Download
                csv_buffer = io.StringIO()
                df_result.to_csv(csv_buffer, index=False)
                col1.download_button(
                    label="📥 Download CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"Cleaned_{uploaded_file.name.split('.')[0]}.csv",
                    mime="text/csv",
                    key=f"csv_{uploaded_file.name}"
                )
                
                # Excel Download
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    df_result.to_excel(writer, index=False)
                col2.download_button(
                    label="📥 Download Excel",
                    data=excel_buffer.getvalue(),
                    file_name=f"Cleaned_{uploaded_file.name.split('.')[0]}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"xlsx_{uploaded_file.name}"
                )
