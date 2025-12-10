//+------------------------------------------------------------------+
//|                                              StackingManager.mqh |
//|                                     Copyright 2025, LAWRANCE KOH |
//|                                          lawrancekoh@outlook.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, LAWRANCE KOH"
#property link      "lawrancekoh@outlook.com"
#property version   "1.00"

#include "Settings.mqh"
#include "MoneyManager.mqh"
#include "TradeManager.mqh"

class CStackingManager
  {
private:
   CMoneyManager* m_money_manager;
   CTradeManager* m_trade_manager;

public:
   CStackingManager(void) {};
   ~CStackingManager(void) {};

   void SetMoneyManager(CMoneyManager* mm) { m_money_manager = mm; }
   void SetTradeManager(CTradeManager* tm) { m_trade_manager = tm; }

   void Init()
     {
        // Nothing to do here for now
     }

   void ManageStacking(ENUM_POSITION_TYPE direction, const CBasket &basket)
     {
      // Guard clauses
      if (CSettings::StackingMaxTrades <= 0 || basket.Ticket == 0) return;
      if (basket.StackingCount >= CSettings::StackingMaxTrades) return;

      // Risk check: high drawdown blocks stacking
      if (!m_money_manager.CheckDrawdown()) return;

      // Profit-based trigger check
      if (basket.ProfitPips() < CSettings::StackingTriggerPips) return;

      // Calculate Stacking Lot
      double stack_lot = m_money_manager.GetStackingLotSize(basket);

      // Capture existing basket SL before opening new trade
      double previous_basket_sl = m_trade_manager.GetBasketSL(direction);

      // Execute Stacking Trade
      int signal = (basket.BasketDirection == POSITION_TYPE_BUY) ? SIGNAL_BUY : SIGNAL_SELL;
      m_trade_manager.OpenTrade(signal, stack_lot, CSettings::SlPips, 0, "STACK", basket.StackingCount + 1);

      // Refresh basket cache to include the new stacking position
      m_trade_manager.Refresh();
      CBasket updated_basket = m_trade_manager.GetCachedBasket(direction);

      // Set uniform SL on all basket positions to ensure consistency
      // We must determine whether to use the new trade's SL or the existing (potentially tighter) SL
      double new_trade_sl = m_trade_manager.GetBasketSL(updated_basket.BasketDirection);
      double sl_to_apply = new_trade_sl;

      if(previous_basket_sl > 0)
        {
         if(updated_basket.BasketDirection == POSITION_TYPE_BUY)
           {
            // For BUY, higher SL is better (tighter)
            if(previous_basket_sl > new_trade_sl)
               sl_to_apply = previous_basket_sl;
           }
         else
           {
            // For SELL, lower SL is better (tighter)
            if(previous_basket_sl < new_trade_sl && previous_basket_sl > 0)
               sl_to_apply = previous_basket_sl;
           }
        }

      if(sl_to_apply > 0)
        {
         m_trade_manager.SetBasketSL(updated_basket.BasketDirection, sl_to_apply);
         // Update current_basket_sl for the subsequent BE check
         // Note: We don't declare double current_basket_sl here as it was used below,
         // but the original code declared it locally. We need to make sure subsequent code uses sl_to_apply or we re-declare.
        }

      double current_basket_sl = sl_to_apply;

      // Always move SL to true breakeven when stacking triggers to account for costs
      double be_price = m_trade_manager.CalculateTrueBreakEvenPrice(updated_basket, updated_basket.TotalVolume);

      // Validate BE price against market price and stops level
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

      // Only set if breakeven is better than current SL and safe
      if (is_safe && ((updated_basket.BasketDirection == POSITION_TYPE_BUY && be_price > current_basket_sl) ||
          (updated_basket.BasketDirection == POSITION_TYPE_SELL && be_price < current_basket_sl && current_basket_sl != 0.0))) {
         m_trade_manager.SetBasketSL(updated_basket.BasketDirection, be_price);
      }
     }
  };
//+------------------------------------------------------------------+