//+------------------------------------------------------------------+
//|                                       CSignal_BollingerBands.mqh |
//|                                     Copyright 2025, LAWRANCE KOH |
//|                                          lawrancekoh@outlook.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, LAWRANCE KOH"
#property link      "lawrancekoh@outlook.com"
#property version   "1.00"

#include "ISignal.mqh"

/**
 * @brief Bollinger Bands signal implementation.
 *
 * Generates signals when price crosses back inside the bands:
 * - Buy: Price crosses up back inside the lower band (Close[1] > Lower[1] && Close[2] <= Lower[2]).
 * - Sell: Price crosses down back inside the upper band (Close[1] < Upper[1] && Close[2] >= Upper[2]).
 */
class CSignal_BollingerBands : public ISignal
{
private:
    int m_handle;                    // Indicator handle
    ENUM_TIMEFRAMES m_timeframe;     // Timeframe
    int m_period;                    // Period
    int m_shift;                     // Shift
    double m_deviation;              // Deviation
    ENUM_APPLIED_PRICE m_applied_price; // Applied price
    int m_last_signal;               // Last signal for status
    ENUM_SIGNAL_ROLE m_role;         // Signal role

public:
    /**
     * @brief Initializes the Bollinger Bands signal.
     *
     * @param settings The signal settings.
     * @return bool true if successful, false otherwise.
     */
    virtual bool Init(const CSignalSettings &settings) override
    {
        // Map parameters from settings
        // IntParams[0]: Period
        // IntParams[1]: Shift
        // DoubleParams[0]: Deviation
        m_timeframe = settings.Timeframe;
        m_period = settings.Params.IntParams[0];
        m_shift = settings.Params.IntParams[1];
        m_deviation = settings.Params.DoubleParams[0];
        m_applied_price = settings.Params.Price;
        m_role = settings.Role;

        // Get indicator handle
        m_handle = iBands(_Symbol, m_timeframe, m_period, m_shift, m_deviation, m_applied_price);

        // Validate handle
        if (m_handle == INVALID_HANDLE)
        {
            Print("CSignal_BollingerBands: Failed to get handle for ", _Symbol, " timeframe ", EnumToString(m_timeframe));
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
        double upper_buffer[3]; // Buffer 1
        double lower_buffer[3]; // Buffer 2
        double close_buffer[3]; // Close price

        // Get data from last closed bar (shift 1) and the one before (shift 2)
        // Buffers: 0 - Base, 1 - Upper, 2 - Lower
        if (CopyBuffer(m_handle, 1, 1, 2, upper_buffer) != 2 ||
            CopyBuffer(m_handle, 2, 1, 2, lower_buffer) != 2)
        {
            return SIGNAL_NONE;
        }

        // Get Close prices
        if (CopyClose(_Symbol, m_timeframe, 1, 2, close_buffer) != 2)
        {
            return SIGNAL_NONE;
        }

        // Arrays: [0] = shift 1 (previous bar), [1] = shift 2 (bar before previous)

        double upper_shift1 = upper_buffer[0]; // Upper band at shift 1
        double upper_shift2 = upper_buffer[1]; // Upper band at shift 2
        double lower_shift1 = lower_buffer[0]; // Lower band at shift 1
        double lower_shift2 = lower_buffer[1]; // Lower band at shift 2
        double close_shift1 = close_buffer[0]; // Close at shift 1
        double close_shift2 = close_buffer[1]; // Close at shift 2

        // Logic: Cross Back Inside Bands
        // Buy: Close[1] > Lower_Band[1] AND Close[2] <= Lower_Band[2] (crossed up back inside lower band)
        bool buy_signal = close_shift1 > lower_shift1 && close_shift2 <= lower_shift2;

        // Sell: Close[1] < Upper_Band[1] AND Close[2] >= Upper_Band[2] (crossed down back inside upper band)
        bool sell_signal = close_shift1 < upper_shift1 && close_shift2 >= upper_shift2;

        if (buy_signal)
        {
            m_last_signal = SIGNAL_BUY;
            return SIGNAL_BUY;
        }
        if (sell_signal)
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
        string status = StringFormat("BB(%d, %.1f)", m_period, m_deviation);

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
