//+------------------------------------------------------------------+
//|                                            AdaptiveManager.mqh   |
//|                                                     LAWRANCE KOH |
//|                 Manages Adaptive Parameter Updates from ML Server|
//+------------------------------------------------------------------+
//| PURPOSE:                                                         |
//|   Orchestrates the connection between ZMQClient and CSettings.   |
//|   Updates DCA multipliers based on ML-predicted market regime.   |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, LAWRANCE KOH"
#property link      "lawrancekoh@outlook.com"
#property version   "1.00"

#include <FXATM/Managers/ZMQClient.mqh>
#include <FXATM/Managers/Settings.mqh>

//+------------------------------------------------------------------+
//| Global Adaptive Manager Instance                                  |
//+------------------------------------------------------------------+
class CAdaptiveManager
{
private:
    CZMQClient      m_zmq_client;
    bool            m_is_enabled;
    int             m_update_frequency_bars;    // How often to request update
    int             m_bars_since_update;
    int             m_last_regime_id;
    string          m_server_address;
    
    // Feature calculation (simplified - real implementation would use full ATR/ADX)
    double          CalculateHurst();
    double          CalculateATR(int period = 14);
    double          CalculateADX(int period = 14);
    
public:
                    CAdaptiveManager();
                   ~CAdaptiveManager();
    
    // Lifecycle
    bool            Initialize(bool enabled = true, string server = "tcp://localhost:5555", int update_freq = 4);
    void            Shutdown();
    
    // Core method - call on each new bar
    void            OnNewBar();
    
    // Manual control
    void            ForceUpdate();
    int             GetCurrentRegime() { return m_last_regime_id; }
    bool            IsEnabled() { return m_is_enabled; }
};

//+------------------------------------------------------------------+
//| Constructor                                                       |
//+------------------------------------------------------------------+
CAdaptiveManager::CAdaptiveManager()
{
    m_is_enabled = false;
    m_update_frequency_bars = 4;  // Update every 4 bars (1 hour on M15)
    m_bars_since_update = 0;
    m_last_regime_id = 0;
    m_server_address = "tcp://localhost:5555";
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
bool CAdaptiveManager::Initialize(bool enabled = true, string server = "tcp://localhost:5555", int update_freq = 4)
{
    m_is_enabled = enabled;
    m_update_frequency_bars = update_freq;
    m_server_address = server;
    
    if(!m_is_enabled)
    {
        Print("[AdaptiveManager] Disabled - using static parameters");
        return true;
    }
    
    if(!m_zmq_client.Initialize(m_server_address))
    {
        Print("[AdaptiveManager] Failed to connect to ML server");
        m_is_enabled = false;
        return false;
    }
    
    Print("[AdaptiveManager] Initialized - connected to ", m_server_address);
    
    // Get initial parameters
    ForceUpdate();
    
    return true;
}

//+------------------------------------------------------------------+
//| Shutdown                                                          |
//+------------------------------------------------------------------+
void CAdaptiveManager::Shutdown()
{
    m_zmq_client.Shutdown();
    m_is_enabled = false;
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
    if(!m_is_enabled || !m_zmq_client.IsConnected())
        return;
    
    // Calculate current market features
    double hurst = CalculateHurst();
    double atr = CalculateATR(14);
    double adx = CalculateADX(14);
    
    // Request adaptive parameters from Python server
    AdaptiveParams params = m_zmq_client.GetAdaptiveParams(_Symbol, hurst, atr, adx);
    
    if(params.is_valid)
    {
        m_last_regime_id = params.regime_id;
        
        // Update CSettings with new multipliers
        // These are used by DCAManager for lot sizing and step calculation
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
//| Simplified Hurst calculation (placeholder)                        |
//| Real implementation would use the full R/S method                 |
//+------------------------------------------------------------------+
double CAdaptiveManager::CalculateHurst()
{
    // Placeholder: Return a middle value
    // In production, this would calculate H from price data
    // Or better: the Python server can calculate from sent OHLC data
    
    // Simple proxy: Use recent price volatility as a rough estimate
    double prices[];
    ArraySetAsSeries(prices, true);
    CopyClose(_Symbol, PERIOD_M15, 0, 100, prices);
    
    if(ArraySize(prices) < 100)
        return 0.5;
    
    // Very rough approximation based on price direction consistency
    int up_count = 0;
    for(int i = 1; i < 100; i++)
    {
        if(prices[i-1] > prices[i])
            up_count++;
    }
    
    // If mostly in one direction, H > 0.5 (trending)
    // If balanced, H ~ 0.5 (random)
    double ratio = (double)up_count / 99.0;
    double hurst = 0.5 + MathAbs(ratio - 0.5);  // 0.5 to 1.0
    
    return hurst;
}

//+------------------------------------------------------------------+
//| Calculate ATR (normalized)                                        |
//+------------------------------------------------------------------+
double CAdaptiveManager::CalculateATR(int period = 14)
{
    int atr_handle = iATR(_Symbol, PERIOD_M15, period);
    if(atr_handle == INVALID_HANDLE)
        return 0.001;
    
    double atr_buffer[];
    ArraySetAsSeries(atr_buffer, true);
    CopyBuffer(atr_handle, 0, 0, 1, atr_buffer);
    IndicatorRelease(atr_handle);
    
    if(ArraySize(atr_buffer) < 1)
        return 0.001;
    
    // Normalize by current price
    double close = iClose(_Symbol, PERIOD_M15, 0);
    if(close > 0)
        return atr_buffer[0] / close;
    
    return 0.001;
}

//+------------------------------------------------------------------+
//| Calculate ADX                                                     |
//+------------------------------------------------------------------+
double CAdaptiveManager::CalculateADX(int period = 14)
{
    int adx_handle = iADX(_Symbol, PERIOD_M15, period);
    if(adx_handle == INVALID_HANDLE)
        return 25.0;
    
    double adx_buffer[];
    ArraySetAsSeries(adx_buffer, true);
    CopyBuffer(adx_handle, 0, 0, 1, adx_buffer);  // Main ADX line
    IndicatorRelease(adx_handle);
    
    if(ArraySize(adx_buffer) < 1)
        return 25.0;
    
    return adx_buffer[0];
}

//+------------------------------------------------------------------+
//| Global instance                                                   |
//+------------------------------------------------------------------+
CAdaptiveManager g_adaptive_manager;

//+------------------------------------------------------------------+
