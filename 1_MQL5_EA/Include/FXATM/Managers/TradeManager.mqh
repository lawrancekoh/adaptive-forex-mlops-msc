//+------------------------------------------------------------------+
//|                                                 TradeManager.mqh |
//|                                     Copyright 2025, LAWRANCE KOH |
//|                                          lawrancekoh@outlook.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, LAWRANCE KOH"
#property link      "lawrancekoh@outlook.com"
#property version   "1.00"

#include <Trade\Trade.mqh>
#include "Settings.mqh"
#include "MoneyManager.mqh"
#include "Basket.mqh"

//--- Signal definitions
#define SIGNAL_BUY  1
#define SIGNAL_SELL -1
#define SIGNAL_NONE 0


class CTradeManager
  {
private:
    CTrade   m_trade;
    string   m_symbol;

    // Basket caching for performance optimization
    CBasket  m_buy_basket_cache;
    CBasket  m_sell_basket_cache;
    bool     m_cache_valid;

    //+------------------------------------------------------------------+
    //| Generates structured comment for trades                          |
    //+------------------------------------------------------------------+
    string GenerateComment(int signal, string trade_type, int serial_number)
      {
       string base = CSettings::EaName;
       if(base == "") base = "FXATM";
       string direction = (signal == SIGNAL_BUY) ? "BUY" : "SELL";
       return base + " " + direction + " " + trade_type + " " + IntegerToString(serial_number);
      }

public:
    CTradeManager(void) {};
    ~CTradeManager(void) {};

   void Init()
     {
      m_symbol = Symbol();
      m_trade.SetExpertMagicNumber(CSettings::EaMagicNumber);
      m_trade.SetDeviationInPoints(CSettings::MaxSlippagePoints);
      m_trade.SetTypeFillingBySymbol(m_symbol);
     }

   //+------------------------------------------------------------------+
//| Opens a trade based on the signal.                               |
//+------------------------------------------------------------------+
   bool OpenTrade(const int signal, const double lots, const int sl_pips, const int tp_pips, string tradeType, int serial)
     {
      if(signal == SIGNAL_NONE)
         return false;

      // Generate comment using the new parameters
      string comment = GenerateComment(signal, tradeType, serial);

      //--- Determine Order Type
      ENUM_ORDER_TYPE order_type = (signal == SIGNAL_BUY) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;

      //--- Get Current Price
      double price = SymbolInfoDouble(_Symbol, (order_type == ORDER_TYPE_BUY) ? SYMBOL_ASK : SYMBOL_BID);

      //--- Calculate SL/TP Prices
      double pip_size = CMoneyManager::GetPipSize();
      double sl_price = 0;
      if(sl_pips > 0)
        {
         sl_price = (order_type == ORDER_TYPE_BUY) ? price - sl_pips * pip_size : price + sl_pips * pip_size;
        }

      double tp_price = 0;
      if(tp_pips > 0)
        {
         tp_price = (order_type == ORDER_TYPE_BUY) ? price + tp_pips * pip_size : price - tp_pips * pip_size;
        }

      //--- Execute Trade
      if(order_type == ORDER_TYPE_BUY)
        {
         return m_trade.Buy(lots, _Symbol, price, sl_price, tp_price, comment);
        }
      else
        {
         return m_trade.Sell(lots, _Symbol, price, sl_price, tp_price, comment);
        }
     }

   //+------------------------------------------------------------------+
   //| Checks if a basket of trades is already open for this symbol and direction. |
   //+------------------------------------------------------------------+
   bool HasOpenBasket(ENUM_POSITION_TYPE direction = POSITION_TYPE_BUY)
     {
      for(int i = PositionsTotal() - 1; i >= 0; i--)
        {
         ulong ticket = PositionGetTicket(i);
         if(ticket == 0) continue;
         if(!PositionSelectByTicket(ticket)) continue;
         if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;

         // In backtest mode, skip magic number check due to Strategy Tester limitations
         if(!CSettings::IsBacktestMode)
           {
            if(PositionGetInteger(POSITION_MAGIC) != CSettings::EaMagicNumber) continue;
           }

         if((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) == direction) return true;
        }
      return false;
     }

   //+------------------------------------------------------------------+
   //| Gets the current basket state by scanning open positions for the specified direction. |
   //+------------------------------------------------------------------+
   CBasket GetBasket(ENUM_POSITION_TYPE direction)
     {
      CBasket basket;
      datetime latestTime = D'1970.01.01 00:00:00';
      datetime earliestTime = D'2030.01.01 00:00:00';
      int count = 0;
      int stacking_count = 0;
      double total_volume = 0.0;
      double weighted_price_sum = 0.0;
      double total_profit = 0.0;
      double total_costs = 0.0;

      for(int i = PositionsTotal() - 1; i >= 0; i--)
        {
         ulong ticket = PositionGetTicket(i);
         if(ticket == 0) continue;
         if(!PositionSelectByTicket(ticket)) continue;
         if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;

         // In backtest mode, skip magic number check due to Strategy Tester limitations
         if(!CSettings::IsBacktestMode)
           {
            if(PositionGetInteger(POSITION_MAGIC) != CSettings::EaMagicNumber) continue;
           }

         if((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) != direction) continue;

         // Add ticket to basket
         basket.AddTicket(ticket);

         count++;
         double volume = PositionGetDouble(POSITION_VOLUME);
         double entry_price = PositionGetDouble(POSITION_PRICE_OPEN);
         total_volume += volume;
         weighted_price_sum += volume * entry_price;
         total_profit += PositionGetDouble(POSITION_PROFIT);
         total_costs += PositionGetDouble(POSITION_SWAP); // POSITION_COMMISSION deprecated

         // Optimized parsing: check for PTP flag first
         string comment = PositionGetString(POSITION_COMMENT);
         bool has_ptp = (StringLen(comment) == 0 || StringFind(comment, "[PTP]") != -1);

         // Parse comment for flags and basket info
         if (has_ptp) {
            basket.HasPartialTPExecuted = true;
         }

         // Parse base comment by stripping flags (anything in brackets)
         string base_comment = comment;
         int flag_pos = StringFind(base_comment, " [");
         if (flag_pos != -1) {
            base_comment = StringSubstr(base_comment, 0, flag_pos);
         }

         // Parse comment format: base direction type serial (e.g., FXATMv4 BUY INIT 1)
         // Updated to handle EA names with spaces by parsing from the end
         string parts[];
         int split_count = StringSplit(base_comment, ' ', parts);
         if (split_count >= 4) {
            if (parts[split_count-2] == "STACK") {
               stacking_count++;
            }
         } else if (StringLen(base_comment) > 0) {
            // Malformed base comment, log warning but continue
            Print("Warning: Malformed base comment '", base_comment, "' in position ", ticket);
         }

         datetime posTime = (datetime)PositionGetInteger(POSITION_TIME);
         if(posTime > latestTime)
           {
            latestTime = posTime;
            basket.Ticket = (int)ticket;
            basket.LastTradePrice = entry_price;
            basket.LastTradeLots = volume;
            basket.BasketDirection = direction;

            // Only update basket type and serial from the latest trade
            if (split_count >= 4) {
               basket.BasketType = parts[split_count-2];
               basket.SerialNumber = (int)StringToInteger(parts[split_count-1]);
            }
           }
         if(posTime < earliestTime)
           {
            earliestTime = posTime;
            basket.InitialTradePrice = entry_price;
           }
        }

      basket.TradeCount = count;
      basket.StackingCount = stacking_count;
      basket.HasStacked = stacking_count > 0;
      basket.TotalVolume = total_volume;
      basket.AvgEntryPrice = (total_volume > 0.0) ? weighted_price_sum / total_volume : 0.0;
      basket.TotalProfit = total_profit;
      basket.TotalCosts = total_costs;
      return basket;
     }

   //+------------------------------------------------------------------+
   //| Sets the same Stop Loss price for all positions in the basket   |
   //+------------------------------------------------------------------+
   void SetBasketSL(ENUM_POSITION_TYPE direction, double sl_price)
     {
      // Get broker's minimum stops level
      double stops_level = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
      double stops_distance = stops_level * _Point;
      double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      
      Print("SetBasketSL Debug: Direction: ", direction, " Target SL: ", sl_price, " Bid: ", bid, " Ask: ", ask);

      for(int i = PositionsTotal() - 1; i >= 0; i--)
        {
         ulong ticket = PositionGetTicket(i);
         if(ticket == 0) continue;
         if(!PositionSelectByTicket(ticket)) continue;
         if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;

         // In backtest mode, skip magic number check due to Strategy Tester limitations
         if(!CSettings::IsBacktestMode)
           {
            if(PositionGetInteger(POSITION_MAGIC) != CSettings::EaMagicNumber) continue;
           }

         if((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) != direction) continue;

         // Validate stops level before modifying
         bool is_valid_sl = true;
         if(direction == POSITION_TYPE_BUY)
           {
            // For BUY: SL must be below BID by at least stops_distance
            if(sl_price >= bid - stops_distance)
              {
               // Set to minimum allowed SL (small loss) to ensure SL is set
               sl_price = bid - stops_distance - _Point * 10;
               Print("SetBasketSL Debug: Adjusted BUY SL to ", sl_price);
              }
           }
         else // POSITION_TYPE_SELL
           {
            // For SELL: SL must be above ASK by at least stops_distance
            if(sl_price <= ask + stops_distance)
              {
               // Set to minimum allowed SL (small loss) to ensure SL is set
               sl_price = ask + stops_distance + _Point * 10;
               Print("SetBasketSL Debug: Adjusted SELL SL to ", sl_price);
              }
           }

         double current_sl = PositionGetDouble(POSITION_SL);
         double current_tp = PositionGetDouble(POSITION_TP);
         double norm_current_sl = NormalizeDouble(current_sl, _Digits);
         double norm_sl_price = NormalizeDouble(sl_price, _Digits);
         if(norm_current_sl != norm_sl_price) // Modify only if SL differs
           {
            Print("SetBasketSL Debug: Modifying ticket ", ticket, " SL from ", current_sl, " to ", sl_price);
            if(!m_trade.PositionModify(ticket, sl_price, current_tp)) {
                Print("SetBasketSL Debug: Failed to modify ticket ", ticket, " Error: ", m_trade.ResultRetcode());
            }
           }
        }
     }

   //+------------------------------------------------------------------+
   //| Calculates True Break-Even price for remaining volume            |
   //+------------------------------------------------------------------+
   double CalculateTrueBreakEvenPrice(const CBasket &basket, double remaining_vol)
   {
       double total_costs = basket.TotalCosts;
       double desired_profit = CMoneyManager::GetMoneyFromPips(CSettings::BeOffsetPips, remaining_vol);
       double total_money_needed = desired_profit - total_costs;
       double pips_needed = CMoneyManager::GetPipsFromMoney(total_money_needed, remaining_vol);
       if (pips_needed < 0) pips_needed = 0;  // Prevent setting SL to a loss level; use entry price as BE
       double pip_size = CMoneyManager::GetPipSize();
       if (basket.BasketDirection == POSITION_TYPE_BUY)
       {
           return basket.AvgEntryPrice + pips_needed * pip_size;  // SL above entry for BUY (Profit)
       }
       else
       {
           return basket.AvgEntryPrice - pips_needed * pip_size;  // SL below entry for SELL (Profit)
       }
   }

   //+------------------------------------------------------------------+
   //| Calculates Basket TP price based on average entry price          |
   //+------------------------------------------------------------------+
   double CalculateBasketTpPrice(const CBasket &basket, int tp_pips)
   {
       double pip_size = CMoneyManager::GetPipSize();
       if (basket.BasketDirection == POSITION_TYPE_BUY)
       {
           return basket.AvgEntryPrice + tp_pips * pip_size;
       }
       else
       {
           return basket.AvgEntryPrice - tp_pips * pip_size;
       }
   }

   //+------------------------------------------------------------------+
   //| Manages Basket TP by setting TP on all positions when basket expands |
   //+------------------------------------------------------------------+
   void ManageBasketTP(const CBasket &basket)
   {
       if (basket.TradeCount <= 1 || CSettings::BasketTpPips <= 0) return;
       double tp_price = CalculateBasketTpPrice(basket, CSettings::BasketTpPips);
       SetBasketTP(basket.BasketDirection, tp_price);
   }

   //+------------------------------------------------------------------+
   //| Manages Partial Take Profit with proportional distribution      |
   //+------------------------------------------------------------------+
   void ManagePartialTP(const CBasket &basket)
   {
       // Guard clauses
       if (CSettings::PartialTpTriggerPips <= 0 || basket.HasPartialTPExecuted || basket.ProfitPips() < CSettings::PartialTpTriggerPips) return;

       // Calculate target volume to close
       double target_volume_to_close = basket.TotalVolume * (CSettings::PartialTpClosePercent / 100.0);
       double actual_closed_volume = 0.0;
       double min_vol = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
       double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);

       // Proportional distribution across all positions
       for (int i = 0; i < ArraySize(basket.Tickets); i++) {
           ulong ticket = basket.Tickets[i];
           if (!PositionSelectByTicket(ticket)) continue;

           double pos_volume = PositionGetDouble(POSITION_VOLUME);
           double proportional_close = pos_volume * (target_volume_to_close / basket.TotalVolume);
           proportional_close = MathFloor(proportional_close / step) * step;  // Round down

           if (proportional_close >= min_vol) {
               bool close_success = m_trade.PositionClosePartial(ticket, proportional_close);
               if (close_success) {
                   actual_closed_volume += proportional_close;
               } else {
                   // Print("PTP: Failed to close position ", ticket, " volume ", proportional_close);
               }
           } else {
               // Print("PTP: Skipping position ", ticket, " as calculated partial close volume ", proportional_close, " is below min volume ", min_vol);
           }
       }

       if (actual_closed_volume == 0.0) return;  // No closes succeeded

       // Update basket after partial closes to get correct AvgEntryPrice
       CBasket updated_basket = GetBasket(basket.BasketDirection);

       // Set True BE SL on remaining positions if enabled
       if (CSettings::PartialTpSetBe) {
           double be_price = CalculateTrueBreakEvenPrice(updated_basket, updated_basket.TotalVolume);

           // Validate BE price against current market price and stops level
           double stops_level = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
           double stops_distance = stops_level * _Point;
           double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
           double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
           bool is_safe = false;

           if (updated_basket.BasketDirection == POSITION_TYPE_BUY) {
               // For BUY, SL must be < Bid - Stops
               if (be_price < bid - stops_distance) is_safe = true;
           } else {
               // For SELL, SL must be > Ask + Stops
               if (be_price > ask + stops_distance) is_safe = true;
           }

           if (is_safe) {
               Print("PTP Debug: Setting BE SL to ", be_price);
               SetBasketSL(updated_basket.BasketDirection, be_price);
           } else {
               Print("PTP Debug: Skipping BE SL - Price too close or in loss. BE: ", be_price, " Bid: ", bid, " Ask: ", ask);
           }
       }


   }

       
   //+------------------------------------------------------------------+
   void SetBasketTP(ENUM_POSITION_TYPE direction, double tp_price)
     {
      for(int i = PositionsTotal() - 1; i >= 0; i--)
        {
         ulong ticket = PositionGetTicket(i);
         if(ticket == 0) continue;
         if(!PositionSelectByTicket(ticket)) continue;
         if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;

         // In backtest mode, skip magic number check due to Strategy Tester limitations
         if(!CSettings::IsBacktestMode)
           {
            if(PositionGetInteger(POSITION_MAGIC) != CSettings::EaMagicNumber) continue;
           }

         if((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) != direction) continue;

         double current_sl = PositionGetDouble(POSITION_SL);
         double current_tp = PositionGetDouble(POSITION_TP);
         
         double norm_current_tp = NormalizeDouble(current_tp, _Digits);
         double norm_tp_price = NormalizeDouble(tp_price, _Digits);
         
         if(norm_current_tp != norm_tp_price) // Modify only if TP differs
           {
            m_trade.PositionModify(ticket, current_sl, tp_price);
           }
        }
     }

   void CloseTrades(ENUM_ORDER_TYPE type)
     {
      // Logic to close trades of a certain type.
     }

   int GetOpenTradesCount()
     {
      // Logic to count open trades.
      return 0;
     }

   //+------------------------------------------------------------------+
   //| Refresh basket cache once per tick for performance optimization |
   //+------------------------------------------------------------------+
   void Refresh()
     {
      // Reset baskets
      m_buy_basket_cache = CBasket();
      m_sell_basket_cache = CBasket();

      // Variables for weighted average calculation
      double buy_weighted_sum = 0.0;
      double sell_weighted_sum = 0.0;
      datetime buy_latest = D'1970.01.01 00:00:00';
      datetime buy_earliest = D'2030.01.01 00:00:00';
      datetime sell_latest = D'1970.01.01 00:00:00';
      datetime sell_earliest = D'2030.01.01 00:00:00';

      // Single loop to populate both baskets simultaneously
      for(int i = PositionsTotal() - 1; i >= 0; i--)
        {
         ulong ticket = PositionGetTicket(i);
         if(ticket == 0 || !PositionSelectByTicket(ticket)) continue;
         if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;

         // In backtest mode, skip magic number check due to Strategy Tester limitations
         if(!CSettings::IsBacktestMode)
           {
            if(PositionGetInteger(POSITION_MAGIC) != CSettings::EaMagicNumber) continue;
           }

         ENUM_POSITION_TYPE direction = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);

         // Update basket statistics based on direction
         double volume = PositionGetDouble(POSITION_VOLUME);
         double entry_price = PositionGetDouble(POSITION_PRICE_OPEN);

         if(direction == POSITION_TYPE_BUY)
           {
            // Update buy basket
            m_buy_basket_cache.AddTicket(ticket);
            m_buy_basket_cache.TradeCount++;
            m_buy_basket_cache.TotalVolume += volume;
            buy_weighted_sum += volume * entry_price;
            m_buy_basket_cache.TotalProfit += PositionGetDouble(POSITION_PROFIT);
            m_buy_basket_cache.TotalCosts += PositionGetDouble(POSITION_SWAP);

            // Optimized parsing: check for PTP flag first
            string comment = PositionGetString(POSITION_COMMENT);
            bool has_ptp = (StringLen(comment) == 0 || StringFind(comment, "[PTP]") != -1);

            // Parse comment for flags and basket info (optimized)
            if (has_ptp) {
               m_buy_basket_cache.HasPartialTPExecuted = true;
            }

            // Parse base comment by stripping flags (anything in brackets)
            string base_comment = comment;
            int flag_pos = StringFind(base_comment, " [");
            if (flag_pos != -1) {
               base_comment = StringSubstr(base_comment, 0, flag_pos);
            }

            // Parse comment format: base direction type serial (e.g., FXATMv4 BUY INIT 1)
            // Updated to handle EA names with spaces by parsing from the end
            string parts[];
            int split_count = StringSplit(base_comment, ' ', parts);
            if (split_count >= 4) {
               if (parts[split_count-2] == "STACK") {
                  m_buy_basket_cache.StackingCount++;
               }
            } else if (StringLen(base_comment) > 0) {
               // Malformed base comment, log warning but continue
               Print("Warning: Malformed base comment '", base_comment, "' in position ", ticket);
            }

            // Track latest and earliest times
            datetime posTime = (datetime)PositionGetInteger(POSITION_TIME);
            if(posTime > buy_latest)
              {
               buy_latest = posTime;
               m_buy_basket_cache.Ticket = (int)ticket;
               m_buy_basket_cache.LastTradePrice = entry_price;
               m_buy_basket_cache.LastTradeLots = volume;
               m_buy_basket_cache.BasketDirection = direction;

               // Only update basket type and serial from the latest trade
               if (split_count >= 4) {
                  m_buy_basket_cache.BasketType = parts[split_count-2];
                  m_buy_basket_cache.SerialNumber = (int)StringToInteger(parts[split_count-1]);
               }
              }
            if(posTime < buy_earliest)
              {
               buy_earliest = posTime;
               m_buy_basket_cache.InitialTradePrice = entry_price;
              }
           }
         else // POSITION_TYPE_SELL
           {
            // Update sell basket
            m_sell_basket_cache.AddTicket(ticket);
            m_sell_basket_cache.TradeCount++;
            m_sell_basket_cache.TotalVolume += volume;
            sell_weighted_sum += volume * entry_price;
            m_sell_basket_cache.TotalProfit += PositionGetDouble(POSITION_PROFIT);
            m_sell_basket_cache.TotalCosts += PositionGetDouble(POSITION_SWAP);

            // Optimized parsing: check for PTP flag first
            string comment = PositionGetString(POSITION_COMMENT);
            bool has_ptp = (StringLen(comment) == 0 || StringFind(comment, "[PTP]") != -1);

            // Parse comment for flags and basket info (optimized)
            if (has_ptp) {
               m_sell_basket_cache.HasPartialTPExecuted = true;
            }

            // Parse base comment by stripping flags (anything in brackets)
            string base_comment = comment;
            int flag_pos = StringFind(base_comment, " [");
            if (flag_pos != -1) {
               base_comment = StringSubstr(base_comment, 0, flag_pos);
            }

            // Parse comment format: base direction type serial (e.g., FXATMv4 BUY INIT 1)
            // Updated to handle EA names with spaces by parsing from the end
            string parts[];
            int split_count = StringSplit(base_comment, ' ', parts);
            if (split_count >= 4) {
               if (parts[split_count-2] == "STACK") {
                  m_sell_basket_cache.StackingCount++;
               }
            } else if (StringLen(base_comment) > 0) {
               // Malformed base comment, log warning but continue
               Print("Warning: Malformed base comment '", base_comment, "' in position ", ticket);
            }

            // Track latest and earliest times
            datetime posTime = (datetime)PositionGetInteger(POSITION_TIME);
            if(posTime > sell_latest)
              {
               sell_latest = posTime;
               m_sell_basket_cache.Ticket = (int)ticket;
               m_sell_basket_cache.LastTradePrice = entry_price;
               m_sell_basket_cache.LastTradeLots = volume;
               m_sell_basket_cache.BasketDirection = direction;

               // Only update basket type and serial from the latest trade
               if (split_count >= 4) {
                  m_sell_basket_cache.BasketType = parts[split_count-2];
                  m_sell_basket_cache.SerialNumber = (int)StringToInteger(parts[split_count-1]);
               }
              }
            if(posTime < sell_earliest)
              {
               sell_earliest = posTime;
               m_sell_basket_cache.InitialTradePrice = entry_price;
              }
           }
        }

      // Calculate final statistics for both baskets
      if(m_buy_basket_cache.TradeCount > 0)
        {
         m_buy_basket_cache.AvgEntryPrice = (m_buy_basket_cache.TotalVolume > 0.0) ? buy_weighted_sum / m_buy_basket_cache.TotalVolume : 0.0;
         m_buy_basket_cache.HasStacked = m_buy_basket_cache.StackingCount > 0;
        }

      if(m_sell_basket_cache.TradeCount > 0)
        {
         m_sell_basket_cache.AvgEntryPrice = (m_sell_basket_cache.TotalVolume > 0.0) ? sell_weighted_sum / m_sell_basket_cache.TotalVolume : 0.0;
         m_sell_basket_cache.HasStacked = m_sell_basket_cache.StackingCount > 0;
        }

      m_cache_valid = true;
     }

   //+------------------------------------------------------------------+
   //| Get cached basket state (call Refresh() first)                   |
   //+------------------------------------------------------------------+
   CBasket GetCachedBasket(ENUM_POSITION_TYPE direction)
     {
      if(!m_cache_valid) Refresh();
      return (direction == POSITION_TYPE_BUY) ? m_buy_basket_cache : m_sell_basket_cache;
     }

   //+------------------------------------------------------------------+
   //| Check if cached basket exists (call Refresh() first)             |
   //+------------------------------------------------------------------+
   bool HasCachedBasket(ENUM_POSITION_TYPE direction)
     {
      if(!m_cache_valid) Refresh();
      CBasket basket = (direction == POSITION_TYPE_BUY) ? m_buy_basket_cache : m_sell_basket_cache;
      return basket.Ticket > 0;
     }

   //+------------------------------------------------------------------+
   //| Get the current Stop Loss price for the basket                   |
   //+------------------------------------------------------------------+
   double GetBasketSL(ENUM_POSITION_TYPE direction)
     {
      for(int i = PositionsTotal() - 1; i >= 0; i--)
        {
         ulong ticket = PositionGetTicket(i);
         if(ticket == 0) continue;
         if(!PositionSelectByTicket(ticket)) continue;
         if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;

         // In backtest mode, skip magic number check due to Strategy Tester limitations
         if(!CSettings::IsBacktestMode)
           {
            if(PositionGetInteger(POSITION_MAGIC) != CSettings::EaMagicNumber) continue;
           }

         if((ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE) != direction) continue;

         // Return SL of the first matching position (they should all be the same)
         return PositionGetDouble(POSITION_SL);
        }
      return 0.0; // No positions found
     }

   //+------------------------------------------------------------------+
   //| Check if the basket's stop loss is in a profitable position      |
   //+------------------------------------------------------------------+
   bool IsStopLossProfitable(const CBasket &basket)
     {
      return basket.ProfitPips() > 0;
     }

  };
//+------------------------------------------------------------------+