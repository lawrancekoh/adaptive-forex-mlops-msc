//+------------------------------------------------------------------+
//|                                                SignalManager.mqh |
//|                                     Copyright 2025, LAWRANCE KOH |
//|                                          lawrancekoh@outlook.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, LAWRANCE KOH"
#property link      "lawrancekoh@outlook.com"
#property version   "1.01" // Updated version

#include "Settings.mqh"
#include "../Signals/ISignal.mqh"
#include <Arrays/ArrayObj.mqh>
// --- Include all signal implementations that will be created
// #include "../Signals/CSignal_RSI.mqh"
// #include "../Signals/CSignal_MACD.mqh"

/**
 * @class CSignalManager
 * @brief Manages signal aggregation with persistent bias mechanism.
 *
 * This class implements a "sticky" bias system where bias signals (ROLE_BIAS) persist
 * across multiple bars until overridden or timed out. The persistence helps maintain
 * trend direction even when bias signals temporarily disappear.
 *
 * Key Features:
 * - Bias Persistence: Once set by a bias signal, the bias holds until:
 *   1. A conflicting bias signal appears (disagreement resets to NONE).
 *   2. No bias signal reinforces it for a configurable number of bars (timeout).
 * - Timeout Mechanism: Uses a counter that increments on each GetFinalSignal() call
 *   (tied to EA heartbeat timeframe). Resets after CSettings::BiasPersistenceBars calls.
 *   Example: With M15 heartbeat and 20 bars setting, bias resets after ~5 hours.
 * - Signal Aggregation: Follows "All Must Agree" logic, but persistent bias allows
 *   entry signals to trigger trades even if no current bias signal is present.
 *
 * Usage Notes:
 * - Bias signals set the direction; entry signals trigger trades within that direction.
 * - Timeout prevents stale bias from influencing decisions indefinitely.
 * - Counter resets when bias is updated or cleared.
 */
class CSignalManager
  {
private:
    CArrayObj         m_signals;
    // --- Persistent bias state ---
    int               m_current_bias;          // Current persistent bias (BUY/SELL/NONE)
    int               m_bias_timeout_counter;  // Counts calls since bias was last set/updated

public:
    CSignalManager(void) : m_current_bias(SIGNAL_NONE), m_bias_timeout_counter(0) // Initialize bias to NONE and counter to 0
      {
      };

    ~CSignalManager(void)
      {
       for(int i = 0; i < m_signals.Total(); i++)
         {
          delete m_signals.At(i);
         }
       m_signals.Clear();
      };

    void AddSignal(ISignal* signal)
      {
       if (signal == NULL)
         {
          Print("SignalManager: Attempted to add null signal pointer");
          return;
         }
       m_signals.Add(signal);
      }

    /**
     * @brief Gets the final, combined trading signal, now with persistent bias.
     *
     * 1. It first checks all signals on the CURRENT bar for new triggers.
     * 2. It checks for disagreements (e.g., two bias signals fighting).
     * 3. If a new, non-conflicting bias signal appears, it UPDATES the
     * persistent 'm_current_bias'.
     * 4. If no new bias signal appears, 'm_current_bias' KEEPS its old value.
     * 5. Finally, it checks the 'm_current_bias' against any ENTRY triggers.
     *
     * @return int The final trade signal (SIGNAL_BUY, SIGNAL_SELL, SIGNAL_NONE).
     */
    int GetFinalSignal()
      {
       // --- Bias Timeout Check ---
       // The persistent bias times out after a configurable number of bars (calls to this method).
       // This prevents stale bias from influencing decisions forever.
       // - Counter increments each time bias is active.
       // - Resets when bias is updated or when no bias is present.
       // - Tied to EA heartbeat timeframe (e.g., M15), so "bars" here mean heartbeat intervals.
       if (m_current_bias != SIGNAL_NONE)
         {
          m_bias_timeout_counter++;
          if (m_bias_timeout_counter >= CSettings::BiasPersistenceBars)
            {
             m_current_bias = SIGNAL_NONE;
             m_bias_timeout_counter = 0;
             Print("SignalManager: Bias timed out after ", CSettings::BiasPersistenceBars, " bars. Reset to NONE.");
            }
         }
       else
         {
          m_bias_timeout_counter = 0;  // Reset counter when no bias
         }

       // --- STEP 1: Check all signals for CURRENT bar triggers ---
       bool biasConfigured = false;
       bool biasBuy = false;   // Represents a NEW bias signal THIS BAR
       bool biasSell = false;  // Represents a NEW bias signal THIS BAR
       bool entryConfigured = false;
       bool entryBuy = false;
       bool entrySell = false;

       // Iterate through all signals
       for(int i = 0; i < m_signals.Total(); i++)
         {
          ISignal* sig = m_signals.At(i);
          if(sig == NULL) continue;

          int signal = sig.GetSignal();
          ENUM_SIGNAL_ROLE role = sig.GetRole();

          if(role == ROLE_BIAS)
            {
             biasConfigured = true;
             if(signal == SIGNAL_BUY) biasBuy = true;
             else if(signal == SIGNAL_SELL) biasSell = true;
            }
          else if(role == ROLE_ENTRY)
            {
             entryConfigured = true;
             if(signal == SIGNAL_BUY) entryBuy = true;
             else if(signal == SIGNAL_SELL) entrySell = true;
            }
         }

       // --- STEP 2: Update the persistent 'm_current_bias' ---

       // --- Check for Bias Disagreements ---
       // If multiple bias signals disagree (e.g., one BUY, one SELL), reset the persistent bias
       // to NONE immediately. This prevents conflicting bias from persisting.
       if(biasBuy && biasSell)
         {
          Print("SignalManager: Bias signal disagreement on current bar. Bias reset to NONE.");
          m_current_bias = SIGNAL_NONE;  // Explicit reset on disagreement
          return SIGNAL_NONE;
         }

       // --- Update Persistent Bias ---
       // Set or reinforce the persistent bias if a clear signal appears.
       // Reset the timeout counter to start fresh persistence period.
       if(biasBuy)
         {
          m_current_bias = SIGNAL_BUY;
          m_bias_timeout_counter = 0;  // Reset counter on bias update
         }
       else if(biasSell)
         {
          m_current_bias = SIGNAL_SELL;
          m_bias_timeout_counter = 0;  // Reset counter on bias update
         }
       // Note: If no new bias signal, m_current_bias persists from previous bars.


       // --- STEP 3: Final decision matrix using persistent bias ---

       // Check for entry-level disagreements
       if(entryBuy && entrySell)
         {
          Print("SignalManager: Entry signal disagreement. No action taken.");
          return SIGNAL_NONE;
         }

       // Check for a BUY signal
       if((m_current_bias == SIGNAL_BUY || !biasConfigured) &&  // Bias is BUY (or no bias is set)
          (entryBuy || !entryConfigured) &&                     // Entry is BUY (or no entry is set)
          (m_current_bias == SIGNAL_BUY || entryBuy))           // At least one of them MUST be BUY
         {
          return SIGNAL_BUY;
         }

       // Check for a SELL signal
       if((m_current_bias == SIGNAL_SELL || !biasConfigured) && // Bias is SELL (or no bias is set)
          (entrySell || !entryConfigured) &&                    // Entry is SELL (or no entry is set)
          (m_current_bias == SIGNAL_SELL || entrySell))         // At least one of them MUST be SELL
         {
          return SIGNAL_SELL;
         }

       // Default: No signal
       return SIGNAL_NONE;
      }

    /**
     * @brief Gets the status of all signals AND the current persistent bias.
     */
    string GetStatus()
      {
       string status = StringFormat("Current Bias: %s | ", GetSignalString(m_current_bias));
       for(int i = 0; i < m_signals.Total(); i++)
         {
          ISignal* sig = m_signals.At(i);
          if(sig == NULL) continue;
          string roleStr = (sig.GetRole() == ROLE_BIAS) ? "[Bias]" : "[Entry]";
          string tfStr = GetTimeframeString(sig.GetTimeframe());
          status += roleStr + " " + sig.GetStatus() + " " + tfStr + " | ";
         }
       // Remove trailing " | "
       if(StringLen(status) > 3)
          status = StringSubstr(status, 0, StringLen(status) - 3);
       return status;
      }

private:
    string GetSignalString(int signal)
       {
        switch(signal)
          {
           case SIGNAL_BUY: return "BUY";
           case SIGNAL_SELL: return "SELL";
           default: return "NONE";
          }
       }

    string GetTimeframeString(ENUM_TIMEFRAMES timeframe)
       {
        switch(timeframe)
          {
           case PERIOD_M1: return "M1";
           case PERIOD_M5: return "M5";
           case PERIOD_M15: return "M15";
           case PERIOD_M30: return "M30";
           case PERIOD_H1: return "H1";
           case PERIOD_H4: return "H4";
           case PERIOD_D1: return "D1";
           case PERIOD_W1: return "W1";
           case PERIOD_MN1: return "MN1";
           default: return EnumToString(timeframe);
          }
       }
  };
//+------------------------------------------------------------------+