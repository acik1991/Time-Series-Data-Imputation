import streamlit as st
import pandas as pd
import io

def process_data(file, file_extension, date_col, time_col, target_col, is_combined):
    try:
        if file_extension == 'csv':
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        # --- Handle Datetime Logic ---
        if is_combined:
            # If Date and Time are in one column (like your image)
            df['Timestamp'] = pd.to_datetime(df[date_col], dayfirst=True)
        else:
            # If they are in separate columns
            df['Timestamp'] = pd.to_datetime(df[date_col].astype(str) + ' ' + df[time_col].astype(str), dayfirst=True)

        df = df.drop_duplicates(subset=['Timestamp'])
        df = df.sort_values('Timestamp')
        df = df.set_index('Timestamp')
        
        # Select only the target column
        df_subset = df[[target_col]]

        # --- Fill Missing Hourly Gaps ---
        full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='h')
        df_complete = df_subset.reindex(full_range)

        # --- Reformat for Output ---
        df_complete = df_complete.reset_index().rename(columns={'index': 'Timestamp'})
        df_complete['Date'] = df_complete['Timestamp'].dt.strftime('%d/%m/%Y')
        df_complete['Time'] = df_complete['Timestamp'].dt.strftime('%H:%M')
        
        return df_complete[['Date', 'Time', target_col]]

    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None

# --- UI Setup ---
st.set_page_config(page_title="WL Data Completer", layout="wide")
st.title("📊 Water Level Data Completer")

# Sidebar Configuration
st.sidebar.header("⚙️ Column Mapping")
is_combined = st.sidebar.checkbox("Date & Time are in ONE column", value=True)

if is_combined:
    date_label = "Combined Datetime Column"
    default_date = "Start of Interval (UTC+08:00)" # Based on your image
    time_input = None
else:
    date_label = "Date Column"
    default_date = "Date"
    time_input = st.sidebar.text_input("Time Column Name", value="Time")

date_input = st.sidebar.text_input(date_label, value=default_date)
target_input = st.sidebar.text_input("Target Data Column", value="0740231WL_SG. LANAS DI AIR LANAS KELANTAN (5718401)")

uploaded_files = st.file_uploader("Upload Files", type=['csv', 'xlsx'], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        ext = uploaded_file.name.split('.')[-1]
        with st.expander(f"📄 {uploaded_file.name}"):
            # Process
            result = process_data(uploaded_file, ext, date_input, time_input, target_input, is_combined)
            
            if result is not None:
                st.dataframe(result.head(10))
                
                # Downloads
                csv = result.to_csv(index=False).encode('utf-8')
                st.download_button("Download CSV", data=csv, file_name=f"Clean_{uploaded_file.name}.csv", mime='text/csv')
