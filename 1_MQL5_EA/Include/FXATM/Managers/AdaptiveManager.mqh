//+------------------------------------------------------------------+
//|                                            AdaptiveManager.mqh   |
//|                                                     LAWRANCE KOH |
//|         Manages Adaptive Parameter Updates from ML Server / CSV  |
//+------------------------------------------------------------------+
//| PURPOSE:                                                         |
//|   Provides regime-based DCA parameters using:                    |
//|   - HTTP WebRequest to FastAPI server (Live/Demo mode)           |
//|   - CSV file lookup (Strategy Tester mode)                       |
//|                                                                  |
//| Note: WebRequest does NOT work in Strategy Tester, so we use a   |
//| pre-generated CSV cheatsheet for backtesting.                    |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, LAWRANCE KOH"
#property link      "lawrancekoh@outlook.com"
#property version   "3.00"

#include <FXATM/Managers/Settings.mqh>

//+------------------------------------------------------------------+
//| Regime Row Structure (for CSV data)                               |
//+------------------------------------------------------------------+
struct RegimeRow
{
    datetime    time;       // Bar timestamp
    int         id;         // Regime ID (0-3)
    double      dist;       // Distance multiplier
    double      lot;        // Lot multiplier
};

//+------------------------------------------------------------------+
//| Adaptive Parameters Structure                                     |
//+------------------------------------------------------------------+
struct AdaptiveParams
{
    int     regime_id;              // Current market regime (0-3)
    string  regime_label;           // Human-readable label
    double  distance_multiplier;    // DCA distance multiplier
    double  lot_multiplier;         // DCA lot multiplier
    bool    is_valid;               // Whether params were successfully retrieved
    string  error_message;          // Error message if any
};

//+------------------------------------------------------------------+
//| CAdaptiveManager Class                                            |
//+------------------------------------------------------------------+
class CAdaptiveManager
{
private:
    bool            m_is_enabled;
    bool            m_is_testing;           // True if running in Strategy Tester
    int             m_update_frequency_bars;
    int             m_bars_since_update;
    int             m_last_regime_id;
    string          m_api_url;              // FastAPI URL for live mode
    string          m_csv_file;             // CSV file for backtest mode
    int             m_ohlc_lookback;        // Number of bars to send for live inference
    
    // Backtest data storage
    RegimeRow       m_history[];            // Sorted array of regime rows
    int             m_history_count;        // Number of loaded rows
    
    // Private methods
    bool            LoadBacktestData();
    AdaptiveParams  GetParamsFromCSV(datetime current_time);
    AdaptiveParams  GetParamsFromAPI();
    int             BinarySearchTime(datetime target);
    string          BuildOHLCJson();        // New: Build JSON payload with OHLC data
    
public:
                    CAdaptiveManager();
                   ~CAdaptiveManager();
    
    // Lifecycle
    bool            Initialize(bool enabled, string api_url, string csv_file, int update_freq = 4);
    void            Shutdown();
    
    // Core method - call on each new bar
    void            OnNewBar();
    
    // Manual control
    void            ForceUpdate();
    int             GetCurrentRegime() { return m_last_regime_id; }
    bool            IsEnabled() { return m_is_enabled; }
    double          GetDistanceMultiplier() { return CSettings::DcaStepMultiplier; }
    double          GetLotMultiplier() { return CSettings::DcaLotMultiplier; }
};

//+------------------------------------------------------------------+
//| Constructor                                                       |
//+------------------------------------------------------------------+
CAdaptiveManager::CAdaptiveManager()
{
    m_is_enabled = false;
    m_is_testing = false;
    m_update_frequency_bars = 4;
    m_bars_since_update = 0;
    m_last_regime_id = 0;
    m_api_url = "http://localhost:8000/predict";
    m_csv_file = "backtest_cheatsheet.csv";
    m_history_count = 0;
    m_ohlc_lookback = 300;  // 300 bars = 3 days of M15 data (enough for Hurst)
}

//+------------------------------------------------------------------+
//| Destructor                                                        |
//+------------------------------------------------------------------+
CAdaptiveManager::~CAdaptiveManager()
{
    Shutdown();
}

//+------------------------------------------------------------------+
//| Initialize the adaptive manager                                   |
//+------------------------------------------------------------------+
bool CAdaptiveManager::Initialize(bool enabled, string api_url, string csv_file, int update_freq = 4)
{
    m_is_enabled = enabled;
    m_api_url = api_url;
    m_csv_file = csv_file;
    m_update_frequency_bars = update_freq;
    m_is_testing = (bool)MQLInfoInteger(MQL_TESTER);
    
    if(!m_is_enabled)
    {
        Print("[AdaptiveManager] Disabled - using static parameters");
        return true;
    }
    
    if(m_is_testing)
    {
        // Strategy Tester mode: Load CSV cheatsheet
        Print("[AdaptiveManager] Strategy Tester detected - loading CSV cheatsheet");
        if(!LoadBacktestData())
        {
            Print("[AdaptiveManager] Failed to load backtest CSV. Using static parameters.");
            m_is_enabled = false;
            return false;
        }
        Print("[AdaptiveManager] Initialized in BACKTEST mode with ", m_history_count, " rows");
    }
    else
    {
        // Live/Demo mode: Will use WebRequest with OHLC data
        Print("[AdaptiveManager] Live/Demo mode - will use API at ", m_api_url);
        Print("[AdaptiveManager] Will send ", m_ohlc_lookback, " bars of OHLC data for live inference");
        Print("[AdaptiveManager] NOTE: Ensure URL is allowed in Tools > Options > Expert Advisors");
    }
    
    // Get initial parameters
    ForceUpdate();
    
    return true;
}

//+------------------------------------------------------------------+
//| Shutdown                                                          |
//+------------------------------------------------------------------+
void CAdaptiveManager::Shutdown()
{
    ArrayFree(m_history);
    m_history_count = 0;
    m_is_enabled = false;
}

//+------------------------------------------------------------------+
//| Load backtest data from CSV file                                  |
//+------------------------------------------------------------------+
bool CAdaptiveManager::LoadBacktestData()
{
    // Open CSV file from MQL5/Files directory
    int file_handle = FileOpen(m_csv_file, FILE_READ | FILE_CSV | FILE_ANSI, ',');
    
    if(file_handle == INVALID_HANDLE)
    {
        Print("[AdaptiveManager] Cannot open file: ", m_csv_file, " Error: ", GetLastError());
        return false;
    }
    
    // Skip header line
    string header_line = "";
    if(!FileIsEnding(file_handle))
    {
        // Read header (Time,RegimeID,DistMult,LotMult)
        FileReadString(file_handle);  // Time
        FileReadString(file_handle);  // RegimeID
        FileReadString(file_handle);  // DistMult
        FileReadString(file_handle);  // LotMult
    }
    
    // Read data rows
    int count = 0;
    int array_size = 100000;  // Pre-allocate for ~4 years of M15 data
    ArrayResize(m_history, array_size);
    
    while(!FileIsEnding(file_handle) && count < array_size)
    {
        long epoch = StringToInteger(FileReadString(file_handle));
        if(epoch == 0) continue;  // Skip invalid rows
        
        m_history[count].time = (datetime)epoch;
        m_history[count].id = (int)StringToInteger(FileReadString(file_handle));
        m_history[count].dist = StringToDouble(FileReadString(file_handle));
        m_history[count].lot = StringToDouble(FileReadString(file_handle));
        
        count++;
    }
    
    FileClose(file_handle);
    
    // Resize to actual count
    ArrayResize(m_history, count);
    m_history_count = count;
    
    if(m_history_count > 0)
    {
        Print("[AdaptiveManager] Loaded ", m_history_count, " rows from ", m_csv_file);
        Print("[AdaptiveManager] Time range: ", m_history[0].time, " to ", m_history[m_history_count-1].time);
        return true;
    }
    
    Print("[AdaptiveManager] No valid data found in ", m_csv_file);
    return false;
}

//+------------------------------------------------------------------+
//| Binary search for closest time in sorted array                    |
//+------------------------------------------------------------------+
int CAdaptiveManager::BinarySearchTime(datetime target)
{
    if(m_history_count == 0)
        return -1;
    
    int left = 0;
    int right = m_history_count - 1;
    int result = 0;  // Default to first row
    
    while(left <= right)
    {
        int mid = (left + right) / 2;
        
        if(m_history[mid].time <= target)
        {
            result = mid;  // This is a valid candidate
            left = mid + 1;
        }
        else
        {
            right = mid - 1;
        }
    }
    
    return result;
}

//+------------------------------------------------------------------+
//| Get parameters from CSV (backtest mode)                           |
//+------------------------------------------------------------------+
AdaptiveParams CAdaptiveManager::GetParamsFromCSV(datetime current_time)
{
    AdaptiveParams params;
    params.is_valid = false;
    params.regime_id = 0;
    params.distance_multiplier = 1.5;
    params.lot_multiplier = 1.2;
    params.regime_label = "Default";
    
    int idx = BinarySearchTime(current_time);
    
    if(idx >= 0 && idx < m_history_count)
    {
        params.regime_id = m_history[idx].id;
        params.distance_multiplier = m_history[idx].dist;
        params.lot_multiplier = m_history[idx].lot;
        params.is_valid = true;
        
        // Set label based on regime ID
        switch(params.regime_id)
        {
            case 0: params.regime_label = "Trending"; break;
            case 1: params.regime_label = "Strong Trend"; break;
            case 2: params.regime_label = "Choppy"; break;
            case 3: params.regime_label = "Ranging"; break;
            default: params.regime_label = "Unknown"; break;
        }
    }
    else
    {
        params.error_message = "Time not found in cheatsheet";
    }
    
    return params;
}

//+------------------------------------------------------------------+
//| Build JSON payload with OHLC data for live inference              |
//+------------------------------------------------------------------+
string CAdaptiveManager::BuildOHLCJson()
{
    // Copy recent M15 rates
    MqlRates rates[];
    int copied = CopyRates(_Symbol, PERIOD_M15, 0, m_ohlc_lookback, rates);
    
    if(copied < 150)
    {
        Print("[AdaptiveManager] Not enough bars for inference: ", copied);
        // Return minimal request without OHLC data (will trigger fallback)
        return StringFormat(
            "{\"action\":\"GET_PARAMS\",\"symbol\":\"%s\",\"magic\":%d}",
            _Symbol, CSettings::EaMagicNumber
        );
    }
    
    // Build JSON with OHLC data array
    // Format: {"action":"GET_PARAMS","symbol":"EURUSD","magic":123456,"ohlc_data":[{...},{...},...]}
    string json = StringFormat(
        "{\"action\":\"GET_PARAMS\",\"symbol\":\"%s\",\"magic\":%d,\"ohlc_data\":[",
        _Symbol, CSettings::EaMagicNumber
    );
    
    for(int i = 0; i < copied; i++)
    {
        if(i > 0) json += ",";
        json += StringFormat(
            "{\"time\":%d,\"open\":%.5f,\"high\":%.5f,\"low\":%.5f,\"close\":%.5f}",
            (long)rates[i].time,
            rates[i].open,
            rates[i].high,
            rates[i].low,
            rates[i].close
        );
    }
    
    json += "]}";
    
    return json;
}

//+------------------------------------------------------------------+
//| Get parameters from API (live mode)                               |
//+------------------------------------------------------------------+
AdaptiveParams CAdaptiveManager::GetParamsFromAPI()
{
    AdaptiveParams params;
    params.is_valid = false;
    params.regime_id = 0;
    params.distance_multiplier = 1.5;
    params.lot_multiplier = 1.2;
    params.regime_label = "Default";
    
    // Build JSON request body with OHLC data
    string json_request = BuildOHLCJson();
    
    // Prepare WebRequest parameters
    char post_data[];
    char result_data[];
    string result_headers;
    
    StringToCharArray(json_request, post_data, 0, StringLen(json_request));
    ArrayResize(post_data, StringLen(json_request));  // Remove null terminator
    
    string headers = "Content-Type: application/json\r\n";
    
    // Make HTTP POST request (increased timeout for feature calculation)
    int timeout = 10000;  // 10 second timeout (feature calc takes ~150ms + network)
    int response_code = WebRequest(
        "POST",
        m_api_url,
        headers,
        timeout,
        post_data,
        result_data,
        result_headers
    );
    
    if(response_code == -1)
    {
        int error = GetLastError();
        params.error_message = StringFormat("WebRequest failed. Error %d. Add URL to allowed list.", error);
        Print("[AdaptiveManager] ", params.error_message);
        return params;
    }
    
    if(response_code != 200)
    {
        params.error_message = StringFormat("HTTP Error %d", response_code);
        Print("[AdaptiveManager] ", params.error_message);
        return params;
    }
    
    // Parse JSON response
    string response = CharArrayToString(result_data);
    
    // Simple JSON parsing (look for key fields)
    if(StringFind(response, "\"status\":\"OK\"") >= 0 || StringFind(response, "\"status\": \"OK\"") >= 0)
    {
        // Extract regime_id
        int pos = StringFind(response, "\"regime_id\":");
        if(pos >= 0)
        {
            string sub = StringSubstr(response, pos + 12, 10);
            params.regime_id = (int)StringToInteger(sub);
        }
        
        // Extract distance_multiplier
        pos = StringFind(response, "\"distance_multiplier\":");
        if(pos >= 0)
        {
            string sub = StringSubstr(response, pos + 22, 10);
            params.distance_multiplier = StringToDouble(sub);
        }
        
        // Extract lot_multiplier
        pos = StringFind(response, "\"lot_multiplier\":");
        if(pos >= 0)
        {
            string sub = StringSubstr(response, pos + 17, 10);
            params.lot_multiplier = StringToDouble(sub);
        }
        
        // Extract regime_label
        pos = StringFind(response, "\"regime_label\":\"");
        if(pos >= 0)
        {
            int end_pos = StringFind(response, "\"", pos + 16);
            if(end_pos > pos + 16)
            {
                params.regime_label = StringSubstr(response, pos + 16, end_pos - pos - 16);
            }
        }
        
        // Check inference mode
        pos = StringFind(response, "\"inference_mode\":\"live\"");
        if(pos >= 0)
        {
            Print("[AdaptiveManager] Live ML inference successful");
        }
        else
        {
            Print("[AdaptiveManager] Using fallback parameters (insufficient data or model error)");
        }
        
        params.is_valid = true;
    }
    else
    {
        params.error_message = "Invalid response from API: " + StringSubstr(response, 0, 100);
        Print("[AdaptiveManager] ", params.error_message);
    }
    
    return params;
}

//+------------------------------------------------------------------+
//| Called on each new bar - check if update needed                   |
//+------------------------------------------------------------------+
void CAdaptiveManager::OnNewBar()
{
    if(!m_is_enabled)
        return;
    
    m_bars_since_update++;
    
    if(m_bars_since_update >= m_update_frequency_bars)
    {
        ForceUpdate();
        m_bars_since_update = 0;
    }
}

//+------------------------------------------------------------------+
//| Force an immediate parameter update                               |
//+------------------------------------------------------------------+
void CAdaptiveManager::ForceUpdate()
{
    if(!m_is_enabled)
        return;
    
    AdaptiveParams params;
    
    if(m_is_testing)
    {
        // Backtest mode: Get from CSV
        datetime current_time = iTime(_Symbol, PERIOD_M15, 0);
        params = GetParamsFromCSV(current_time);
    }
    else
    {
        // Live mode: Get from API with OHLC data
        params = GetParamsFromAPI();
    }
    
    if(params.is_valid)
    {
        m_last_regime_id = params.regime_id;
        
        // Update CSettings with new multipliers
        CSettings::DcaStepMultiplier = params.distance_multiplier;
        CSettings::DcaLotMultiplier = params.lot_multiplier;
        
        Print("[AdaptiveManager] Updated to Regime ", params.regime_id, 
              " (", params.regime_label, ")",
              " - StepMult=", params.distance_multiplier,
              ", LotMult=", params.lot_multiplier);
    }
    else
    {
        Print("[AdaptiveManager] Failed to get params: ", params.error_message);
    }
}

//+------------------------------------------------------------------+
//| Global instance                                                   |
//+------------------------------------------------------------------+
CAdaptiveManager g_adaptive_manager;

//+------------------------------------------------------------------+
