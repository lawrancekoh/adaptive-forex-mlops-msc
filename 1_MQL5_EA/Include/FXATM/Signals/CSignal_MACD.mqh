//+------------------------------------------------------------------+
//|                                                    CSignal_MACD.mqh |
//|                                     Copyright 2025, LAWRANCE KOH |
//|                                          lawrancekoh@outlook.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, LAWRANCE KOH"
#property link      "lawrancekoh@outlook.com"
#property version   "1.00"

#include "ISignal.mqh"

/**
 * @brief MACD signal implementation.
 *
 * Generates buy/sell signals based on MACD main line crossing the signal line.
 */
class CSignal_MACD : public ISignal
{
private:
    int m_handle;                    // Indicator handle
    ENUM_TIMEFRAMES m_timeframe;     // Timeframe
    int m_fast_period;               // Fast EMA period
    int m_slow_period;               // Slow EMA period
    int m_signal_period;             // Signal line period
    ENUM_APPLIED_PRICE m_applied_price; // Applied price
    bool m_threshold_check;          // Threshold check enabled
    bool m_threshold_check_reverse;  // Reverse threshold logic
    int m_last_signal;               // Last signal for status
    ENUM_SIGNAL_ROLE m_role;         // Signal role (Bias or Entry)

public:
    /**
     * @brief Initializes the MACD signal.
     *
     * @param settings The signal settings.
     * @return bool true if successful, false otherwise.
     */
    virtual bool Init(const CSignalSettings &settings) override
    {
        // Map parameters from settings
        m_timeframe = settings.Timeframe;
        m_fast_period = settings.Params.IntParams[0];
        m_slow_period = settings.Params.IntParams[1];
        m_signal_period = settings.Params.IntParams[2];
        m_applied_price = settings.Params.Price;
        m_threshold_check = settings.Params.BoolParams[0];
        m_threshold_check_reverse = settings.Params.BoolParams[1];
        m_role = settings.Role;

        // Get indicator handle
        m_handle = iMACD(_Symbol, m_timeframe, m_fast_period, m_slow_period, m_signal_period, m_applied_price);

        // Validate handle
        if (m_handle == INVALID_HANDLE)
        {
            Print("CSignal_MACD: Failed to get MACD handle for ", _Symbol, " timeframe ", EnumToString(m_timeframe));
            return false;
        }

        m_last_signal = SIGNAL_NONE;
        return true;
    }

    /**
     * @brief Gets the current trading signal.
     *
     * @return int SIGNAL_BUY, SIGNAL_SELL, or SIGNAL_NONE.
     */
    virtual int GetSignal() override
    {
        double main_buffer[3];
        double signal_buffer[3];
        double histogram_buffer[3] = {0, 0, 0};

        // Get data from last closed bar (shift 1)
        if (CopyBuffer(m_handle, 0, 1, 2, main_buffer) != 2 ||
            CopyBuffer(m_handle, 1, 1, 2, signal_buffer) != 2)
        {
            return SIGNAL_NONE;
        }

        // Histogram is optional for logging
        CopyBuffer(m_handle, 2, 1, 2, histogram_buffer);

        // Implement crossover logic
        bool buy_cross = main_buffer[0] > signal_buffer[0] && main_buffer[1] <= signal_buffer[1];
        bool sell_cross = main_buffer[0] < signal_buffer[0] && main_buffer[1] >= signal_buffer[1];

        // Apply threshold filter if enabled
        if (m_threshold_check)
        {
            if (m_threshold_check_reverse)
            {
                buy_cross = buy_cross && main_buffer[0] > 0;  // Buy cross must be above zero
                sell_cross = sell_cross && main_buffer[0] < 0; // Sell cross must be below zero
            }
            else
            {
                buy_cross = buy_cross && main_buffer[0] < 0;  // Buy cross must be below zero
                sell_cross = sell_cross && main_buffer[0] > 0; // Sell cross must be above zero
            }
        }

        // Return signal
        if (buy_cross)
        {
            m_last_signal = SIGNAL_BUY;
            return SIGNAL_BUY;
        }
        if (sell_cross)
        {
            m_last_signal = SIGNAL_SELL;
            return SIGNAL_SELL;
        }

        m_last_signal = SIGNAL_NONE;
        return SIGNAL_NONE;
    }

    /**
     * @brief Gets the status string.
     *
     * @return string Status message.
     */
    virtual string GetStatus() override
    {
        string status = StringFormat("MACD(%d,%d,%d)", m_fast_period, m_slow_period, m_signal_period);

        if (m_last_signal == SIGNAL_BUY)
            status += " [BUY]";
        else if (m_last_signal == SIGNAL_SELL)
            status += " [SELL]";
        else
            status += " [NEUTRAL]";

        return status;
    }

    /**
     * @brief Gets the role of the signal.
     *
     * @return ENUM_SIGNAL_ROLE The role.
     */
    virtual ENUM_SIGNAL_ROLE GetRole() override
    {
        return m_role;
    }

    /**
     * @brief Gets the timeframe of the signal.
     *
     * @return ENUM_TIMEFRAMES The timeframe.
     */
    virtual ENUM_TIMEFRAMES GetTimeframe() const override
    {
        return m_timeframe;
    }

    /**
     * @brief Draws visual indicators on the chart when a signal triggers.
     *
     * @param barTime The time of the bar where the signal occurred.
     * @param signal The signal type (SIGNAL_BUY, SIGNAL_SELL).
     * @param signalIndex The index of the signal slot (0-2) for vertical stacking.
     */
    virtual void DrawSignal(datetime barTime, int signal, int signalIndex) override
    {
        // Implementation for chart drawing if needed
        // For now, leave as stub or implement basic arrow drawing
    }
};
//+------------------------------------------------------------------+