import pandas as pd
import os
import io

def load_data(source):
    """
    Loads market data from a file path or a file-like object (Streamlit uploader).
    Handles standard MT5 CSV export formats (UTF-16 encoding, tab/comma separators).
    
    Args:
        source (str or file-like): Path to CSV or Streamlit UploadedFile object.
        
    Returns:
        pd.DataFrame: Normalized DataFrame with 'time' index and lower-case columns.
    """
    df = None
    last_error = None
    
    # Candidates to try: (separator, encoding)
    candidates = [
        ('\t', 'utf-16'),
        ('\t', 'utf-16-le'),
        (',', 'utf-16'),
        (',', 'utf-8'),
        ('\t', 'utf-8'),
        (';', 'utf-8') # European CSVs
    ]

    for sep, encoding in candidates:
        try:
            if isinstance(source, str):
                 # It's a file path
                 df_try = pd.read_csv(source, sep=sep, encoding=encoding)
            else:
                 # Streamlit buffer
                 source.seek(0)
                 df_try = pd.read_csv(source, sep=sep, encoding=encoding)
            
            # Simple validation: Must have at least one column that looks like date or time or open
            # And more than 1 column
            if len(df_try.columns) > 1:
                # Check for messy headers often found in wrong encoding
                if not any(x in str(df_try.columns[0]) for x in ['<', 'Date', 'Time', 'date', 'time']):
                    # Header looks weird, maybe wrong encoding read successfully?
                    # e.g. reading utf-16 as utf-8 can result in garbage chars
                    continue
                    
                print(f"Success loading with sep='{sep}', encoding='{encoding}'")
                df = df_try
                break
        except Exception as e:
            last_error = e
            continue
            
    if df is None:
        print(f"Error loading data. Last error: {last_error}")
        return pd.DataFrame()

    try:
        # Normalization
        # 1. Standardize Column Names (remove < > and lower case)
        # MT5 Headers: <DATE>	<OPEN>	<HIGH>	<LOW>	<CLOSE>	<TICKVOL>	<VOL>	<SPREAD>
        df.columns = df.columns.str.replace('<', '').str.replace('>', '').str.lower()
        
        # 2. Combine Date and Time if separate (some MT5 formats), or use Date/Time column
        if 'date' in df.columns and 'time' in df.columns:
             # Check data types. If strings, combine.
             df['time'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
        elif 'date' in df.columns:
             df['time'] = pd.to_datetime(df['date'])
             
        # Drop original Date/Time split columns if they exist
        df = df.drop(columns=['date'], errors='ignore')
             
        # 3. Set Index
        if 'time' in df.columns:
            df = df.set_index('time')
            df.sort_index(inplace=True)
        else:
            print("Warning: No 'time' or 'date' column found after normalization.")
        
        # 4. Filter for required columns
        required = ['open', 'high', 'low', 'close', 'tickvol']
        # Check what we have
        available = [c for c in required if c in df.columns]
        
        if len(available) >= 4: # Assuming OHLC are there
            df = df[available]
            
            # CRITICAL FIX: Ensure all columns are numeric
            # MT5 might export numbers as strings or with wrong decimal separators if locale issues
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Drop any rows that failed conversion (became NaN)
            df.dropna(inplace=True)
            
            print(f"Data loaded successfully: {len(df)} rows. Columns: {list(df.columns)}")
            return df
        else:
             print(f"Missing required columns. Found: {df.columns}")
             return pd.DataFrame()

    except Exception as e:
        print(f"Error during normalization: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Test with local file if it exists
    test_path = "../data/EURUSD_M15.csv"
    if os.path.exists(test_path):
        df = load_data(test_path)
        print(df.head())
        print(df.info())
