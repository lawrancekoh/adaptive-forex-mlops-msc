//+------------------------------------------------------------------+
//|                                                  CatrUtility.mqh |
//|                         ATR Utility for Volatility-Based Trading |
//|                                     Copyright 2025, LAWRANCE KOH |
//|                                          lawrancekoh@outlook.com |
//+------------------------------------------------------------------+
//| PURPOSE:                                                         |
//|   Reusable ATR (Average True Range) calculations for volatility  |
//|   analysis in trading systems. Provides ATR-based lot sizing     |
//|   and dynamic stop loss/take profit levels.                      |
//|                                                                  |
//| KEY FEATURES:                                                    |
//|   • ATR indicator management with automatic initialization       |
//|   • Volatility-adjusted lot sizing with configurable bounds      |
//|   • ATR-based price level calculations for SL/TP placement       |
//|   • Average ATR calculations for normalization                   |
//|                                                                  |
//| USAGE:                                                           |
//|   Instantiate CatrUtility, call Init() with period/timeframe,    |
//|   then use GetAtrMultiplierForLots() or GetAtrBasedLevel()       |
//|   methods for volatility-aware trading decisions.                |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, LAWRANCE KOH"
#property link      "lawrancekoh@outlook.com"
#property version   "1.00"

#include <Indicators\Indicators.mqh>

// ATR utility class for reusable ATR calculations and volatility analysis
class CatrUtility
{
private:
    int m_handle;
    int m_period;
    ENUM_TIMEFRAMES m_timeframe;
    double m_atr_buffer[];

public:
    CatrUtility(void) : m_handle(INVALID_HANDLE), m_period(14), m_timeframe(PERIOD_CURRENT) {}
    ~CatrUtility(void) { Deinit(); }

    // Initialize ATR indicator
    bool Init(int period = 14, ENUM_TIMEFRAMES timeframe = PERIOD_CURRENT)
    {
        m_period = period;
        m_timeframe = timeframe;

        // Validate parameters
        if (period <= 0 || period > 1000)
        {
            Print("CatrUtility::Init: Invalid ATR period: ", period);
            return false;
        }

        // Get ATR indicator handle
        m_handle = iATR(_Symbol, timeframe, period);
        if (m_handle == INVALID_HANDLE)
        {
            Print("CatrUtility::Init: Failed to create ATR indicator handle. Error: ", GetLastError());
            return false;
        }

        // Set buffer as series for easy access
        ArraySetAsSeries(m_atr_buffer, true);

        Print("CatrUtility::Init: ATR indicator initialized successfully. Period: ", period, ", Timeframe: ", timeframe);
        return true;
    }

    // Deinitialize ATR indicator
    void Deinit()
    {
        if (m_handle != INVALID_HANDLE)
        {
            IndicatorRelease(m_handle);
            m_handle = INVALID_HANDLE;
        }
    }

    // Get current ATR value for the last closed bar
    double GetCurrentAtr()
    {
        if (m_handle == INVALID_HANDLE)
        {
            Print("CatrUtility::GetCurrentAtr: ATR indicator not initialized");
            return 0.0;
        }

        // Copy ATR value for the last closed bar (shift 1)
        if (CopyBuffer(m_handle, 0, 1, 1, m_atr_buffer) <= 0)
        {
            Print("CatrUtility::GetCurrentAtr: Failed to copy ATR buffer. Error: ", GetLastError());
            return 0.0;
        }

        return m_atr_buffer[0];
    }

    // Calculate ATR multiplier for lot sizing (higher ATR = smaller lots)
    double GetAtrMultiplierForLots(double baseLot, double currentAtr)
    {
        if (currentAtr <= 0.0 || baseLot <= 0.0)
        {
            Print("CatrUtility::GetAtrMultiplierForLots: Invalid parameters. ATR: ", currentAtr, ", BaseLot: ", baseLot);
            return 1.0;
        }

        // Get average ATR over a longer period for normalization
        double avgAtr = GetAverageAtr(20); // 20-period average ATR
        if (avgAtr <= 0.0)
        {
            return 1.0; // Fallback to base lot size
        }

        // Calculate scaling factor: reduce lots when ATR is high (volatile)
        double volatilityRatio = currentAtr / avgAtr;
        double scalingFactor = 1.0 / MathSqrt(volatilityRatio); // Square root for smoother scaling

        // Clamp scaling factor to reasonable bounds
        if (scalingFactor < 0.1) scalingFactor = 0.1; // Minimum 10% of base lot
        if (scalingFactor > 2.0) scalingFactor = 2.0; // Maximum 200% of base lot

        return scalingFactor;
    }

    // Compute ATR-adjusted price levels for SL/TP
    double GetAtrBasedLevel(double entryPrice, double multiplier, bool isLong, double pipSize)
    {
        double currentAtr = GetCurrentAtr();
        if (currentAtr <= 0.0)
        {
            Print("CatrUtility::GetAtrBasedLevel: Invalid ATR value: ", currentAtr);
            return entryPrice;
        }

        double atrAdjustment = currentAtr * multiplier;

        if (isLong)
        {
            // For long positions, SL below entry, TP above entry
            return entryPrice - (atrAdjustment / pipSize) * pipSize;
        }
        else
        {
            // For short positions, SL above entry, TP below entry
            return entryPrice + (atrAdjustment / pipSize) * pipSize;
        }
    }

private:
    // Get average ATR over specified periods for normalization
    double GetAverageAtr(int periods)
    {
        if (m_handle == INVALID_HANDLE || periods <= 0)
        {
            return 0.0;
        }

        double atrValues[];
        ArraySetAsSeries(atrValues, true);

        if (CopyBuffer(m_handle, 0, 1, periods, atrValues) <= 0)
        {
            Print("CatrUtility::GetAverageAtr: Failed to copy ATR buffer. Error: ", GetLastError());
            return 0.0;
        }

        double sum = 0.0;
        for (int i = 0; i < periods; i++)
        {
            sum += atrValues[i];
        }

        return sum / periods;
    }
};
//+------------------------------------------------------------------+