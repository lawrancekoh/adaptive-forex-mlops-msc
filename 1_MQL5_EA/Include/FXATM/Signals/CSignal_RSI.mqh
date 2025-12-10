//+------------------------------------------------------------------+
//|                                                  CSignal_RSI.mqh |
//|                                     Copyright 2025, LAWRANCE KOH |
//|                                          lawrancekoh@outlook.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, LAWRANCE KOH"
#property link      "lawrancekoh@outlook.com"
#property version   "1.00"

#include "ISignal.mqh"

/**
 * @brief RSI signal implementation.
 *
 * Generates buy/sell signals based on RSI level cross out of oversold/overbought levels.
 */
class CSignal_RSI : public ISignal
{
private:
    int m_handle;                    // Indicator handle
    ENUM_TIMEFRAMES m_timeframe;     // Timeframe
    int m_period;                    // RSI period
    ENUM_APPLIED_PRICE m_applied_price; // Applied price
    double m_lvl_dn;                 // Oversold level
    double m_lvl_up;                 // Overbought level
    int m_last_signal;               // Last signal for status
    ENUM_SIGNAL_ROLE m_role;         // Signal role (Bias or Entry)

public:
    /**
     * @brief Initializes the RSI signal.
     *
     * @param settings The signal settings.
     * @return bool true if successful, false otherwise.
     */
    virtual bool Init(const CSignalSettings &settings) override
    {
        // Map parameters from settings
        m_timeframe = settings.Timeframe;
        m_period = settings.Params.IntParams[0];
        m_applied_price = settings.Params.Price;
        m_lvl_dn = settings.Params.DoubleParams[0];
        m_lvl_up = settings.Params.DoubleParams[1];
        m_role = settings.Role;

        // Get indicator handle
        m_handle = iRSI(_Symbol, m_timeframe, m_period, m_applied_price);

        // Validate handle
        if (m_handle == INVALID_HANDLE)
        {
            Print("CSignal_RSI: Failed to get RSI handle for ", _Symbol, " timeframe ", EnumToString(m_timeframe));
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
        double rsi_buffer[3];

        // Get data from last closed bar (shift 1) and previous (shift 2)
        if (CopyBuffer(m_handle, 0, 1, 2, rsi_buffer) != 2)
        {
            return SIGNAL_NONE;
        }

        // Implement level cross out logic
        bool buy_signal = rsi_buffer[0] > m_lvl_dn && rsi_buffer[1] <= m_lvl_dn; // Crossed up out of oversold
        bool sell_signal = rsi_buffer[0] < m_lvl_up && rsi_buffer[1] >= m_lvl_up; // Crossed down out of overbought

        // Return signal
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
        string status = StringFormat("RSI(%d,%.1f,%.1f)", m_period, m_lvl_dn, m_lvl_up);

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