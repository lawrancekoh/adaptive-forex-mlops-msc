//+------------------------------------------------------------------+
//|                                                   ZMQClient.mqh  |
//|                                                     LAWRANCE KOH |
//|                        ZMQ Client for Python Inference Server    |
//+------------------------------------------------------------------+
//| PURPOSE:                                                         |
//|   Communicate with the Python ML Inference Server via ZeroMQ.    |
//|   Sends feature data, receives adaptive trading parameters.      |
//|                                                                  |
//| REQUIREMENTS:                                                    |
//|   1. Install ZMQ library for MQL5 (mql-zmq)                      |
//|   2. Python Inference Server running on tcp://localhost:5555     |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, LAWRANCE KOH"
#property link      "lawrancekoh@outlook.com"
#property version   "1.00"

// Include the ZMQ library for MQL5
// Note: Requires mql-zmq library to be installed
// https://github.com/dingmaotu/mql-zmq
#include <Zmq/Zmq.mqh>

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
//| CZMQClient Class                                                  |
//+------------------------------------------------------------------+
class CZMQClient
{
private:
    Context     m_context;
    Socket      m_socket;
    string      m_server_address;
    int         m_timeout_ms;
    bool        m_is_connected;
    
public:
                CZMQClient();
               ~CZMQClient();
    
    // Core methods
    bool        Initialize(string server_address = "tcp://localhost:5555", int timeout_ms = 3000);
    void        Shutdown();
    bool        IsConnected() { return m_is_connected; }
    
    // Request adaptive parameters from Python server
    AdaptiveParams  GetAdaptiveParams(string symbol, double hurst, double atr, double adx);
    
    // Request with just symbol (server uses last known features or default)
    AdaptiveParams  GetDefaultParams(string symbol);
    
    // Hot-reload model on server
    bool            ReloadServerModel();
};

//+------------------------------------------------------------------+
//| Constructor                                                       |
//+------------------------------------------------------------------+
CZMQClient::CZMQClient()
{
    m_is_connected = false;
    m_timeout_ms = 3000;
    m_server_address = "tcp://localhost:5555";
}

//+------------------------------------------------------------------+
//| Destructor                                                        |
//+------------------------------------------------------------------+
CZMQClient::~CZMQClient()
{
    Shutdown();
}

//+------------------------------------------------------------------+
//| Initialize connection to Python server                            |
//+------------------------------------------------------------------+
bool CZMQClient::Initialize(string server_address = "tcp://localhost:5555", int timeout_ms = 3000)
{
    m_server_address = server_address;
    m_timeout_ms = timeout_ms;
    
    // Create ZMQ context and socket
    m_context.setBlocky(false);
    m_socket.connect(m_server_address);
    
    // Set receive timeout
    m_socket.setReceiveTimeout(m_timeout_ms);
    m_socket.setSendTimeout(m_timeout_ms);
    
    m_is_connected = true;
    Print("[ZMQClient] Connected to ", m_server_address);
    
    return true;
}

//+------------------------------------------------------------------+
//| Shutdown connection                                               |
//+------------------------------------------------------------------+
void CZMQClient::Shutdown()
{
    if(m_is_connected)
    {
        m_socket.disconnect(m_server_address);
        m_is_connected = false;
        Print("[ZMQClient] Disconnected from ", m_server_address);
    }
}

//+------------------------------------------------------------------+
//| Get adaptive parameters with features                             |
//+------------------------------------------------------------------+
AdaptiveParams CZMQClient::GetAdaptiveParams(string symbol, double hurst, double atr, double adx)
{
    AdaptiveParams params;
    params.is_valid = false;
    params.regime_id = 0;
    params.distance_multiplier = 1.5;
    params.lot_multiplier = 1.2;
    
    if(!m_is_connected)
    {
        params.error_message = "Not connected to server";
        return params;
    }
    
    // Build JSON request
    string request = StringFormat(
        "{\"action\":\"GET_PARAMS\",\"symbol\":\"%s\",\"features\":{\"hurst\":%.4f,\"volatility_atr\":%.6f,\"trend_adx\":%.2f}}",
        symbol, hurst, atr, adx
    );
    
    // Send request
    ZmqMsg request_msg(request);
    if(!m_socket.send(request_msg))
    {
        params.error_message = "Failed to send request";
        Print("[ZMQClient] Error: ", params.error_message);
        return params;
    }
    
    // Receive response
    ZmqMsg response_msg;
    if(!m_socket.recv(response_msg))
    {
        params.error_message = "Timeout waiting for response";
        Print("[ZMQClient] Error: ", params.error_message);
        return params;
    }
    
    // Parse JSON response
    string response = response_msg.getData();
    
    // Simple JSON parsing (MQL5 doesn't have native JSON)
    // Expected format: {"status":"OK","regime_id":2,"regime_label":"Trending",...}
    if(StringFind(response, "\"status\":\"OK\"") >= 0)
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
        
        // Extract regime_label (simplified - just look for known labels)
        if(StringFind(response, "Ranging") >= 0)
            params.regime_label = "Ranging (Safe)";
        else if(StringFind(response, "Trending") >= 0)
            params.regime_label = "Trending";
        else if(StringFind(response, "Choppy") >= 0)
            params.regime_label = "Choppy / Weak Trend";
        else if(StringFind(response, "Strong") >= 0)
            params.regime_label = "Strong Trend / Breakout";
        else
            params.regime_label = "Unknown";
        
        params.is_valid = true;
        Print("[ZMQClient] Received: Regime ", params.regime_id, " (", params.regime_label, 
              ") - dist=", params.distance_multiplier, " lot=", params.lot_multiplier);
    }
    else
    {
        params.error_message = "Server returned error: " + response;
        Print("[ZMQClient] Error: ", params.error_message);
    }
    
    return params;
}

//+------------------------------------------------------------------+
//| Get default parameters (no features provided)                     |
//+------------------------------------------------------------------+
AdaptiveParams CZMQClient::GetDefaultParams(string symbol)
{
    // Use middle-of-the-road defaults for features
    return GetAdaptiveParams(symbol, 0.5, 0.001, 25.0);
}

//+------------------------------------------------------------------+
//| Request server to reload its model                                |
//+------------------------------------------------------------------+
bool CZMQClient::ReloadServerModel()
{
    if(!m_is_connected)
        return false;
    
    string request = "{\"action\":\"RELOAD_MODEL\"}";
    ZmqMsg request_msg(request);
    
    if(!m_socket.send(request_msg))
        return false;
    
    ZmqMsg response_msg;
    if(!m_socket.recv(response_msg))
        return false;
    
    string response = response_msg.getData();
    return (StringFind(response, "\"status\":\"OK\"") >= 0);
}

//+------------------------------------------------------------------+
