//+------------------------------------------------------------------+
//|                                                      ISignal.mqh |
//|                                     Copyright 2025, LAWRANCE KOH |
//|                                          lawrancekoh@outlook.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, LAWRANCE KOH"
#property link      "lawrancekoh@outlook.com"
#property version   "1.00"

#include <Object.mqh>
#include "../Managers/Settings.mqh"

/**
 * @brief Defines the contract for all signal-generating classes.
 *
 * This interface ensures that every signal, regardless of its underlying
 * indicator or logic, provides a consistent way for the SignalManager
 * to initialize it, retrieve trading signals, and check its status.
 */
class ISignal : public CObject
   {
public:
     /**
      * @brief Initializes the signal with the required parameters.
      *
      * This method should be called once before any other methods are used.
      * It sets up the indicator handles and any other necessary configurations.
      *
      * @param settings The signal settings struct containing all parameters.
      * @return bool true if initialization is successful, false otherwise.
      */
     virtual bool Init(const CSignalSettings &settings) { return false; }

     /**
      * @brief Gets the latest trading signal.
      *
      * This is the core method that the SignalManager will call on every tick
      * or bar to determine if a trading opportunity exists.
      *
      * @return int A signal from the ENUM_TRADE_SIGNAL enumeration
      *         (e.g., SIGNAL_BUY, SIGNAL_SELL, SIGNAL_NONE).
      */
     virtual int GetSignal() { return SIGNAL_NONE; }

     /**
      * @brief Gets the current status of the signal.
      *
      * This can be used for debugging or displaying status information on the UI.
      * For example, it could return "RSI(14) is Overbought" or "MACD Cross Occurred".
      *
      * @return string A human-readable status message.
      */
     virtual string GetStatus() { return ""; }

     /**
      * @brief Gets the role of the signal (Bias or Entry).
      *
      * @return ENUM_SIGNAL_ROLE The role assigned to this signal.
      */
     virtual ENUM_SIGNAL_ROLE GetRole() { return ROLE_BIAS; }

     /**
      * @brief Gets the timeframe this signal operates on.
      *
      * @return ENUM_TIMEFRAMES The timeframe (e.g., PERIOD_M15, PERIOD_H1).
      */
     virtual ENUM_TIMEFRAMES GetTimeframe() const { return PERIOD_CURRENT; }

     /**
      * @brief Draws visual indicators on the chart when a signal triggers.
      *
      * @param barTime The time of the bar where the signal occurred.
      * @param signal The signal type (SIGNAL_BUY, SIGNAL_SELL).
      * @param signalIndex The index of the signal slot (0-2) for vertical stacking.
      */
     virtual void DrawSignal(datetime barTime, int signal, int signalIndex) = 0;
    };
//+------------------------------------------------------------------+
