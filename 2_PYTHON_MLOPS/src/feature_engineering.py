import pandas as pd
import numpy as np

# Manual Hurst Exponent Calculation (R/S Method)
# This bypasses the 'hurst' library which has issues with certain data.

def _manual_hurst(ts):
    """
    Calculates Hurst Exponent using Rescaled Range (R/S) analysis.
    Returns value between 0 and 1.
    H < 0.5: Mean-reverting
    H = 0.5: Random walk
    H > 0.5: Trending
    """
    ts = np.array(ts)
    n = len(ts)
    
    if n < 20:  # Need sufficient data points
        return np.nan
        
    # Ensure no NaNs or Infs
    if np.any(np.isnan(ts)) or np.any(np.isinf(ts)):
        return np.nan
    
    # Calculate returns (differences)
    returns = np.diff(ts)
    
    if len(returns) < 10 or np.std(returns) < 1e-10:
        return np.nan
    
    # Divide into sub-series and calculate R/S for each
    max_k = min(n // 2, 100)  # Maximum sub-series size
    min_k = 10  # Minimum sub-series size
    
    rs_list = []
    n_list = []
    
    for k in range(min_k, max_k + 1, max(1, (max_k - min_k) // 10)):
        # Number of sub-series of length k
        n_subseries = n // k
        if n_subseries < 1:
            continue
            
        rs_vals = []
        for i in range(n_subseries):
            subseries = ts[i * k : (i + 1) * k]
            
            # Mean-centered cumulative deviation
            mean_sub = np.mean(subseries)
            y = np.cumsum(subseries - mean_sub)
            
            # Range
            R = np.max(y) - np.min(y)
            
            # Standard deviation
            S = np.std(subseries, ddof=1)
            
            if S > 1e-10:
                rs_vals.append(R / S)
        
        if rs_vals:
            rs_list.append(np.mean(rs_vals))
            n_list.append(k)
    
    if len(rs_list) < 2:
        return np.nan
    
    # Log-log regression to find H
    log_n = np.log(n_list)
    log_rs = np.log(rs_list)
    
    # Linear regression: log(R/S) = H * log(n) + c
    try:
        slope, _ = np.polyfit(log_n, log_rs, 1)
        # Clamp H to valid range [0, 1]
        return max(0.0, min(1.0, slope))
    except:
        return np.nan

def calc_hurst(series, window=100):
    """
    Calculates rolling Hurst Exponent using manual R/S method.
    """
    values = series.values
    n = len(values)
    hurst_list = [np.nan] * n
    
    # Use every 10th point for speed (Hurst is slow to change)
    for i in range(window, n, 1):  # Calculate for all, but can increase step for speed
        window_data = values[i - window : i]
        hurst_list[i] = _manual_hurst(window_data)
        
    # Forward fill NaN gaps if using step > 1
    result = pd.Series(hurst_list, index=series.index)
    return result

def calc_atr(df, period=14):
    """
    Calculates Normalized Average True Range (ATR).
    Normalized ATR = ATR / Close Price.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Calculate True Range (TR)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate ATR (Wilder's Smoothing)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    
    # Normalize
    natr = atr / close
    return natr

def calc_adx(df, period=14):
    """
    Calculates Average Directional Index (ADX) using Pandas.
    """
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)
    
    # Calculate Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Calculate True Range (TR)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Smooth TR and DMs (Wilder's Smoothing -> EWM)
    tr_smooth = tr.ewm(alpha=1/period, adjust=False).mean()
    
    # Avoid division by zero in DI calculations
    tr_smooth_safe = tr_smooth.replace(0, np.nan)
    
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / tr_smooth_safe)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / tr_smooth_safe)
    
    # Calculate DX - handle division by zero
    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()
    
    # Where di_sum is 0 or NaN, set DX to 0 (no directional movement)
    dx = np.where(di_sum > 0, 100 * (di_diff / di_sum), 0.0)
    dx = pd.Series(dx, index=df.index)
    
    # Calculate ADX (Smoothed DX)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return adx

def build_feature_matrix(df):
    """
    Generates the feature matrix X for the GMM model.
    Columns: [hurst, volatility_atr, trend_adx]
    """
    # Ensure required columns exist
    required = ['close', 'high', 'low']
    if not all(col in df.columns for col in required):
        raise ValueError(f"Dataframe missing required columns: {required}")
        
    print(f"Calculating Features on {len(df)} rows...")
    print(f"Input df dtypes: {df.dtypes.to_dict()}")
    print(f"Sample input data (first 3 rows):\n{df.head(3)}")
    
    df_feat = pd.DataFrame(index=df.index)
    
    try:
        # 1. Hurst
        print("- Calculating Hurst (this may take time)...")
        close_prices = df['close'].astype(float)
        df_feat['hurst'] = calc_hurst(close_prices, window=100)
        hurst_valid = df_feat['hurst'].notna().sum()
        print(f"  Hurst: {hurst_valid} valid values, {len(df_feat) - hurst_valid} NaN")
        
        # 2. ATR
        print("- Calculating ATR...")
        df_feat['volatility_atr'] = calc_atr(df, period=14)
        atr_valid = df_feat['volatility_atr'].notna().sum()
        print(f"  ATR: {atr_valid} valid values, {len(df_feat) - atr_valid} NaN")
        
        # 3. ADX
        print("- Calculating ADX...")
        df_feat['trend_adx'] = calc_adx(df, period=14)
        adx_valid = df_feat['trend_adx'].notna().sum()
        print(f"  ADX: {adx_valid} valid values, {len(df_feat) - adx_valid} NaN")
        
        print(f"Features calculated. Shape before dropna: {df_feat.shape}")
        print(f"Sample features (rows 100-105):\n{df_feat.iloc[100:105] if len(df_feat) > 105 else df_feat.head()}")
        
        # Drop NaNs generated by rolling windows
        df_feat.dropna(inplace=True)
        
        print(f"Features final shape: {df_feat.shape}")
        if df_feat.empty:
            print("WARNING: All rows dropped after feature engineering.")
        
        return df_feat
    except Exception as e:
        import traceback
        print(f"Feature Calculation Error: {e}")
        traceback.print_exc()
        return pd.DataFrame()
