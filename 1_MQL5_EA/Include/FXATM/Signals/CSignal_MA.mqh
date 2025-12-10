//+------------------------------------------------------------------+
//|                                                   CSignal_MA.mqh |
//|                                     Copyright 2025, LAWRANCE KOH |
//|                                          lawrancekoh@outlook.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, LAWRANCE KOH"
#property link      "lawrancekoh@outlook.com"
#property version   "1.00"

#include "ISignal.mqh"

/**
 * @brief Moving Average Crossover signal implementation.
 *
 * Generates buy/sell signals based on Fast MA crossing Slow MA.
 */
class CSignal_MA : public ISignal
{
private:
    int m_handle_fast;               // Fast MA handle
    int m_handle_slow;               // Slow MA handle
    ENUM_TIMEFRAMES m_timeframe;     // Timeframe
    int m_fast_period;               // Fast MA period
    int m_slow_period;               // Slow MA period
    int m_shift;                     // Shift
    ENUM_MA_METHOD m_ma_method;      // MA Method
    ENUM_APPLIED_PRICE m_applied_price; // Applied price
    int m_last_signal;               // Last signal for status
    ENUM_SIGNAL_ROLE m_role;         // Signal role (Bias or Entry)

public:
    /**
     * @brief Initializes the MA Crossover signal.
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
        m_shift = settings.Params.IntParams[2];
        m_ma_method = settings.Params.MaMethod1;
        m_applied_price = settings.Params.Price;
        m_role = settings.Role;

        // Get indicator handles
        m_handle_fast = iMA(_Symbol, m_timeframe, m_fast_period, m_shift, m_ma_method, m_applied_price);
        m_handle_slow = iMA(_Symbol, m_timeframe, m_slow_period, m_shift, m_ma_method, m_applied_price);

        // Validate handles
        if (m_handle_fast == INVALID_HANDLE || m_handle_slow == INVALID_HANDLE)
        {
            Print("CSignal_MA: Failed to get MA handles for ", _Symbol, " timeframe ", EnumToString(m_timeframe));
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
        double fast_buffer[3];
        double slow_buffer[3];

        // Get data from last closed bar (index 1) and the one before (index 2)
        // We request 2 elements starting from index 1.
        if (CopyBuffer(m_handle_fast, 0, 1, 2, fast_buffer) != 2 ||
            CopyBuffer(m_handle_slow, 0, 1, 2, slow_buffer) != 2)
        {
            return SIGNAL_NONE;
        }

        // Arrays are ordered by time, so [0] is oldest (index 2 in reverse), [1] is newest (index 1 in reverse)
        // Wait, CopyBuffer behavior:
        // "The order of elements in the copied array is from present to past." -> NO.
        // CopyBuffer: "The elements in the target array are indexed starting from 0. The order of elements ... depends on the array series flag."
        // BUT if we use a dynamic array without SetAsSeries, index 0 is the OLDEST.
        // Here I used static arrays `double fast_buffer[3]`. They are not series by default.
        // So fast_buffer[0] is the oldest copied element (bar index 2).
        // fast_buffer[1] is the newest copied element (bar index 1).

        double fast_prev = fast_buffer[0]; // Index 2
        double fast_curr = fast_buffer[1]; // Index 1
        double slow_prev = slow_buffer[0]; // Index 2
        double slow_curr = slow_buffer[1]; // Index 1

        // Implement crossover logic
        // Buy: Fast crosses Slow upwards
        bool buy_cross = fast_curr > slow_curr && fast_prev <= slow_prev;

        // Sell: Fast crosses Slow downwards
        bool sell_cross = fast_curr < slow_curr && fast_prev >= slow_prev;

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
        string status = StringFormat("MA(%d,%d)", m_fast_period, m_slow_period);

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
        // Stub
    }
};
//+------------------------------------------------------------------+
