//+------------------------------------------------------------------+
//|                                                 MoneyManager.mqh |
//|                                     Copyright 2025, LAWRANCE KOH |
//|                                          lawrancekoh@outlook.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, LAWRANCE KOH"
#property link      "lawrancekoh@outlook.com"
#property version   "1.00"

#include "Settings.mqh"
#include "Basket.mqh"
#include "CatrUtility.mqh"

class CMoneyManager
   {
private:
   CatrUtility* m_atr_utility;

public:
   CMoneyManager(void) : m_atr_utility(NULL) {};
   ~CMoneyManager(void) { if (m_atr_utility != NULL) delete m_atr_utility; };

   void Init()
     {
      // Nothing to do here for now
     }

   // Set ATR utility for volatility-adjusted lot sizing
   void SetAtrUtility(CatrUtility* atr_utility)
     {
      m_atr_utility = atr_utility;
     }

   //+------------------------------------------------------------------+
   //| Validates the lot size against broker limits (min, max, step).  |
   //+------------------------------------------------------------------+
   double ValidateLotSize(double lot)
     {
      string symbol = CSettings::Symbol;
      double min_lot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
      double max_lot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
      double step_lot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);

      // Apply limits
      lot = MathMin(lot, max_lot);
      lot = MathMax(lot, min_lot);

      // Normalize to the nearest valid step
      if(step_lot > 0)
        {
         lot = MathRound(lot / step_lot) * step_lot;
        }

      return lot;
     }

   //+------------------------------------------------------------------+
   //| Returns the pip size for the given symbol (standard 10 points). |
   //+------------------------------------------------------------------+
   static double GetPipSize(string symbol = NULL)
     {
      if(symbol == NULL) symbol = CSettings::Symbol;
      double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
      return point * 10; // Standard pip size is 10 points
     }

   //+------------------------------------------------------------------+
   //| Returns the value of one tick in account currency for pricing.  |
   //+------------------------------------------------------------------+
   static double GetTickValueInAccountCurrency(string symbol = NULL)
     {
      if(symbol == NULL) symbol = CSettings::Symbol;
      return SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
     }

   //+------------------------------------------------------------------+
   //| Calculates the monetary risk of SL pips for one lot in account currency. |
   //+------------------------------------------------------------------+
   static double GetSlValuePerLotInAccountCurrency(int sl_pips, string symbol = NULL)
     {
      double pip_size = GetPipSize(symbol);
      double tick_value = GetTickValueInAccountCurrency(symbol);
      if(symbol == NULL) symbol = CSettings::Symbol;
      double tick_size = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);

      if(tick_size == 0) return 0;

      // Calculate using tick size to handle instruments where tick size != point
      return (sl_pips * pip_size / tick_size) * tick_value;
     }

   //+------------------------------------------------------------------+
   //| Converts pip value to monetary value in account currency for given lot size. |
   //+------------------------------------------------------------------+
   static double GetMoneyFromPips(double pips, double lot_size, string symbol = NULL)
     {
      if(symbol == NULL) symbol = CSettings::Symbol;

      double pip_size = GetPipSize(symbol);
      double tick_value = GetTickValueInAccountCurrency(symbol);
      double tick_size = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);

      if(tick_size == 0) return 0;

      // Calculate using tick size to handle instruments where tick size != point
      return (pips * pip_size / tick_size) * tick_value * lot_size;
     }

   //+------------------------------------------------------------------+
   //| Converts monetary value in account currency to pip value for given lot size. |
   //+------------------------------------------------------------------+
   static double GetPipsFromMoney(double money, double lot_size, string symbol = NULL)
     {
      if(symbol == NULL) symbol = CSettings::Symbol;

      double pip_size = GetPipSize(symbol);
      double tick_value = GetTickValueInAccountCurrency(symbol);
      double tick_size = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);

      if(tick_value == 0 || lot_size == 0 || pip_size == 0) return 0;

      // Reverse the calculation: Money = (Pips * PipSize / TickSize) * TickValue * LotSize
      // Pips = Money / ( (PipSize / TickSize) * TickValue * LotSize )
      return money / ((pip_size / tick_size) * tick_value * lot_size);
     }

   //+------------------------------------------------------------------+
   //| Calculates the initial lot size based on the selected mode.      |
   //+------------------------------------------------------------------+
   double GetInitialLotSize()
     {
      double calculated_lot = 0.0;

      switch(CSettings::LotSizingMode)
        {
         case MODE_FIXED_LOT:
            calculated_lot = CSettings::LotFixed;
            break;

         case MODE_LOTS_PER_THOUSAND_BALANCE:
            calculated_lot = (AccountInfoDouble(ACCOUNT_BALANCE) / 1000.0) * CSettings::LotsPerThousand;
            break;

         case MODE_LOTS_PER_THOUSAND_EQUITY:
            calculated_lot = (AccountInfoDouble(ACCOUNT_EQUITY) / 1000.0) * CSettings::LotsPerThousand;
            break;

         case MODE_RISK_PERCENT_BALANCE:
            if(CSettings::SlPips <= 0)
              {
               Print("Risk modes require SlPips > 0. Falling back to fixed lot.");
               calculated_lot = CSettings::LotFixed;
              }
            else
              {
               double risk_amount = AccountInfoDouble(ACCOUNT_BALANCE) * (CSettings::LotRiskPercent / 100.0);
               double sl_value_per_lot = GetSlValuePerLotInAccountCurrency(CSettings::SlPips);
               if(sl_value_per_lot > 0)
                 {
                  calculated_lot = risk_amount / sl_value_per_lot;
                 }
               else
                 {
                  Print("Unable to calculate SL value. Falling back to fixed lot.");
                  calculated_lot = CSettings::LotFixed;
                 }
              }
            break;

         case MODE_RISK_PERCENT_EQUITY:
            if(CSettings::SlPips <= 0)
              {
               Print("Risk modes require SlPips > 0. Falling back to fixed lot.");
               calculated_lot = CSettings::LotFixed;
              }
            else
              {
               double risk_amount = AccountInfoDouble(ACCOUNT_EQUITY) * (CSettings::LotRiskPercent / 100.0);
               double sl_value_per_lot = GetSlValuePerLotInAccountCurrency(CSettings::SlPips);
               if(sl_value_per_lot > 0)
                 {
                  calculated_lot = risk_amount / sl_value_per_lot;
                 }
               else
                 {
                  Print("Unable to calculate SL value. Falling back to fixed lot.");
                  calculated_lot = CSettings::LotFixed;
                 }
              }
            break;

         case MODE_VOLATILITY_ADJUSTED:
            if(m_atr_utility == NULL)
              {
               Print("MODE_VOLATILITY_ADJUSTED requires ATR utility to be set. Falling back to fixed lot.");
               calculated_lot = CSettings::LotFixed;
              }
            else
              {
               // Get base lot size using fixed mode as baseline
               double base_lot = CSettings::LotFixed;
               if(base_lot <= 0)
                 {
                  base_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
                 }

               // Get current ATR and calculate scaling factor
               double current_atr = m_atr_utility.GetCurrentAtr();
               double scaling_factor = m_atr_utility.GetAtrMultiplierForLots(base_lot, current_atr);

               // Apply volatility adjustment
               calculated_lot = base_lot * scaling_factor;

               Print("Volatility-adjusted lot sizing: Base lot: ", base_lot,
                     ", Current ATR: ", current_atr,
                     ", Scaling factor: ", scaling_factor,
                     ", Final lot: ", calculated_lot);
              }
            break;
        }

      return ValidateLotSize(calculated_lot);
     }

   //+------------------------------------------------------------------+
   //| Calculates the lot size for stacking trades based on the selected mode. |
   //+------------------------------------------------------------------+
   double GetStackingLotSize(const CBasket &basket)
     {
      double calculated_lot = 0.0;

      switch(CSettings::StackingLotMode)
        {
         case MODE_FIXED:
            calculated_lot = CSettings::StackingLotSize;
            break;

         case MODE_LAST_TRADE:
            calculated_lot = basket.LastTradeLots;
            break;

         case MODE_BASKET_TOTAL:
            calculated_lot = basket.TotalVolume;
            break;

         case MODE_ENTRY_BASED:
            calculated_lot = GetInitialLotSize();
            break;
        }

      return ValidateLotSize(calculated_lot);
     }

   //+------------------------------------------------------------------+
   //| Checks if current account drawdown exceeds the threshold.       |
   //| Returns true if drawdown < MaxDrawdownPercent (trading allowed).|
   //+------------------------------------------------------------------+
   bool CheckDrawdown()
     {
      double balance = AccountInfoDouble(ACCOUNT_BALANCE);
      double equity = AccountInfoDouble(ACCOUNT_EQUITY);
      if(balance <= 0) return true; // Avoid division by zero

      double drawdown_percent = ((balance - equity) / balance) * 100.0;
      return drawdown_percent < CSettings::MaxDrawdownPercent;
     }

private:
  };
//+------------------------------------------------------------------+
