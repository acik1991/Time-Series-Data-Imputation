🌊 Water Level Data Completer (Streamlit)
This repository contains a Streamlit web application designed to automate the cleaning and gap-filling of environmental time-series data (specifically water level data). It identifies missing hourly timestamps in raw datasets and fills them to ensure a continuous record for hydrologic analysis.

🚀 Features
Flexible Data Upload: Supports both .csv and .xlsx (Excel) file formats.

Batch Processing: Upload multiple files at once and process them individually.

Customizable Mapping: User-friendly sidebar to specify which columns represent:

Date (e.g., 'Date', 'Tanggal')

Time (e.g., 'Time', 'Waktu')

Target Data (e.g., 'Water Level (m) - Raw', 'WL')

Gap Filling: Automatically detects missing hours between the start and end date and reindexes the data to a standard 1-hour frequency.

One-Click Export: Download processed results back to your local machine as either CSV or Excel files.

🛠️ How It Works
Load: The app reads your uploaded file into a Pandas DataFrame.

Indexing: It combines the user-defined Date and Time columns into a single Datetime index.

Resampling/Reindexing: It creates a complete date range with an hourly frequency (H) and inserts missing rows (NaNs) where data gaps exist.

Formatting: It formats the output back into a clean DD/MM/YYYY and HH:MM structure.

Output: Users preview the data in-browser before downloading.# Fill-in-Calander-for-time-series-data
This repository contains a Streamlit web application designed to automate the cleaning and gap-filling of environmental time-series data


📋 Input Requirements
To ensure the script runs correctly, your input files should contain:

A column for Date.

A column for Time.

A numeric column for the Value (e.g., Water Level).

Note: The app will attempt to parse various date formats automatically using Pandas' to_datetime logic.

📄 License
This project is open-source and available under the MIT License.
