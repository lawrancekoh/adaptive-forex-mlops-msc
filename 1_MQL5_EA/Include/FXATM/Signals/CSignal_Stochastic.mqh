//+------------------------------------------------------------------+
//|                                           CSignal_Stochastic.mqh |
//|                                     Copyright 2025, LAWRANCE KOH |
//|                                          lawrancekoh@outlook.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, LAWRANCE KOH"
#property link      "lawrancekoh@outlook.com"
#property version   "1.00"

#include "ISignal.mqh"

/**
 * @brief Stochastic Oscillator signal implementation.
 *
 * Generates buy/sell signals based on %K crossing %D with Overbought/Oversold filtering.
 */
class CSignal_Stochastic : public ISignal
{
private:
    int m_handle;                    // Indicator handle
    ENUM_TIMEFRAMES m_timeframe;     // Timeframe
    int m_k_period;                  // K period
    int m_d_period;                  // D period
    int m_slowing;                   // Slowing
    ENUM_MA_METHOD m_ma_method;      // MA Method
    ENUM_STO_PRICE m_price_field;    // Price field
    double m_overbought;             // Overbought level (e.g., 80)
    double m_oversold;               // Oversold level (e.g., 20)
    int m_last_signal;               // Last signal for status
    ENUM_SIGNAL_ROLE m_role;         // Signal role (Bias or Entry)

public:
    /**
     * @brief Initializes the Stochastic signal.
     *
     * @param settings The signal settings.
     * @return bool true if successful, false otherwise.
     */
    virtual bool Init(const CSignalSettings &settings) override
    {
        // Map parameters from settings
        m_timeframe = settings.Timeframe;
        m_k_period = settings.Params.IntParams[0];
        m_d_period = settings.Params.IntParams[1];
        m_slowing = settings.Params.IntParams[2];
        m_ma_method = settings.Params.MaMethod1;
        m_price_field = settings.Params.PriceField;
        m_overbought = settings.Params.DoubleParams[0];
        m_oversold = settings.Params.DoubleParams[1];
        m_role = settings.Role;

        // Get indicator handle
        m_handle = iStochastic(_Symbol, m_timeframe, m_k_period, m_d_period, m_slowing, m_ma_method, m_price_field);

        // Validate handle
        if (m_handle == INVALID_HANDLE)
        {
            Print("CSignal_Stochastic: Failed to get Stochastic handle for ", _Symbol, " timeframe ", EnumToString(m_timeframe));
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
        double main_buffer[2];   // %K
        double signal_buffer[2]; // %D

        // Get data from last closed bar (index 1) and the one before (index 2)
        // CopyBuffer returns elements ordered from oldest to newest (for default arrays).
        // Request start 1, count 2.
        // buffer[0] = index 2 (older)
        // buffer[1] = index 1 (newer, last closed)
        if (CopyBuffer(m_handle, 0, 1, 2, main_buffer) != 2 ||
            CopyBuffer(m_handle, 1, 1, 2, signal_buffer) != 2)
        {
            return SIGNAL_NONE;
        }

        double main_prev = main_buffer[0];
        double main_curr = main_buffer[1];
        double signal_prev = signal_buffer[0];
        double signal_curr = signal_buffer[1];

        // Implement crossover logic with filters
        // Buy: %K crosses %D upwards AND (%K < Oversold OR %K was < Oversold recently)
        // Standard conservative: Cross UP happened while below Oversold (or crossing out of it)
        // Here we implement: Cross UP AND Main line is below Oversold (or just crossed out)

        // Let's assume standard "Cross from Oversold zone":
        // The cross itself must happen.
        // And generally we want the cross to happen when the price is "cheap" (oversold).
        // Some prefer "Cross happened AND both lines were below 20".
        // Let's go with: Main line crosses Signal line UP, and Main line (current) is < Oversold OR Main (prev) < Oversold.
        // Actually, safer is "Cross UP" + "Main < Oversold" is too restrictive because if it crosses UP out of 20, it might be > 20 now.
        // So usually: Cross UP AND (Main_Prev < Oversold OR Main_Curr < Oversold).

        // However, user just asked for "Stochastic". I'll use standard cross + level check.
        // Buy if CrossUp AND Main_Curr < Oversold (Deep oversold buy) OR
        // Buy if CrossUp AND Main_Prev < Oversold (Breakout from oversold)

        bool cross_up = main_curr > signal_curr && main_prev <= signal_prev;
        bool cross_down = main_curr < signal_curr && main_prev >= signal_prev;

        // Apply OB/OS filters
        // Buy only if we are in or coming out of Oversold area.
        // Let's relax it slightly: Buy if CrossUp AND Main_Curr <= Oversold + 5? No, keep it strict or standard.
        // Standard: Buy when lines cross upward from below 20.
        // i.e. CrossUp AND (Main_Prev < Oversold)

        bool buy_signal = cross_up && (main_prev < m_oversold);
        bool sell_signal = cross_down && (main_prev > m_overbought);

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
        string status = StringFormat("Stoch(%d,%d,%d)", m_k_period, m_d_period, m_slowing);

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
