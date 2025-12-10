//+------------------------------------------------------------------+
//|                                          TrailingStopManager.mqh |
//|                                     Copyright 2025, LAWRANCE KOH |
//|                                          lawrancekoh@outlook.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, LAWRANCE KOH"
#property link      "lawrancekoh@outlook.com"
#property version   "1.00"

#include "Settings.mqh"
#include <Trade/Trade.mqh>
#include "MoneyManager.mqh"
#include "CatrUtility.mqh"

class CTrailingStopManager : public CObject
{
private:
     CTrade m_trade;
     CatrUtility m_atr_utility;

     void HandleSteppedTSL(const CBasket &basket);
     void HandleAtrTsl(const CBasket &basket);

public:
     CTrailingStopManager(void) {};
     ~CTrailingStopManager(void) {};

     void Init()
     {
         m_trade.SetExpertMagicNumber(CSettings::EaMagicNumber);

         // Initialize ATR utility for ATR-based trailing stops
         if(!m_atr_utility.Init(CSettings::TslAtrPeriod, PERIOD_CURRENT))
         {
             Print("TrailingStopManager::Init: Failed to initialize ATR utility");
         }
     }

    void ManageBasketTSL(const ENUM_POSITION_TYPE direction, const CBasket &basket)
    {
        if (CSettings::TslMode == MODE_TSL_NONE)
        {
            return;
        }

        switch(CSettings::TslMode)
        {
            case MODE_TSL_STEP:
                HandleSteppedTSL(basket);
                break;
            case MODE_TSL_ATR:
                HandleAtrTsl(basket);
                break;
            // Other cases will be added in a later phase
        }
    }
};

void CTrailingStopManager::HandleSteppedTSL(const CBasket &basket)
{
    if(basket.Ticket == 0) return;

    double market_price = (basket.BasketDirection == POSITION_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK);

     // Use cached basket data for performance optimization
     double total_profit_money = basket.TotalProfit;
     double total_costs = basket.TotalCosts;

    // Calculate average profit in pips per lot
    double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
    double pip_value_per_lot = tick_value * (CMoneyManager::GetPipSize() / tick_size); // Value of 1 pip for 1 lot

    double average_profit_pips = 0.0;
    if(pip_value_per_lot > 0 && basket.TotalVolume > 0)
    {
        average_profit_pips = total_profit_money / (basket.TotalVolume * pip_value_per_lot);
    }

    // Debug removed

    if(average_profit_pips < CSettings::TslBeTriggerPips) return;

    double pip_size = CMoneyManager::GetPipSize();
    double breakeven_price;
    if(!CSettings::BreakevenIncludesCosts)
    {
        breakeven_price = basket.AvgEntryPrice;
        if(basket.BasketDirection == POSITION_TYPE_BUY)
            breakeven_price += CSettings::BeOffsetPips * pip_size;
        else
            breakeven_price -= CSettings::BeOffsetPips * pip_size;
    }
    else
    {
        // Account for estimated commission on close
        total_costs -= basket.TotalVolume * CSettings::CommissionPerLot;
        double desired_profit = CMoneyManager::GetMoneyFromPips(CSettings::BeOffsetPips, basket.TotalVolume);
        double total_money_to_cover = desired_profit - total_costs;
        double total_pips_to_cover = CMoneyManager::GetPipsFromMoney(total_money_to_cover, basket.TotalVolume);
        if (total_pips_to_cover < 0) total_pips_to_cover = 0;  // Prevent loss SL
        double offset = total_pips_to_cover * pip_size;
        breakeven_price = basket.AvgEntryPrice;
        if(basket.BasketDirection == POSITION_TYPE_BUY)
            breakeven_price += offset;
        else
            breakeven_price -= offset;
    }

    double profit_beyond_trigger = average_profit_pips - CSettings::TslBeTriggerPips;
    int steps = 0;
    if(profit_beyond_trigger > 0 && CSettings::TslStepPips > 0)
        steps = (int)floor(profit_beyond_trigger / CSettings::TslStepPips);

    double new_sl_price = 0;
    if(basket.BasketDirection == POSITION_TYPE_BUY)
        new_sl_price = breakeven_price + (steps * CSettings::TslStepPips * CMoneyManager::GetPipSize());
    else
        new_sl_price = breakeven_price - (steps * CSettings::TslStepPips * CMoneyManager::GetPipSize());
    
    double stop_level_dist = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * _Point;

    for(int i = 0; i < ArraySize(basket.Tickets); i++)
    {
        ulong ticket = basket.Tickets[i];
        if(!PositionSelectByTicket(ticket)) continue;
        
        double current_sl = PositionGetDouble(POSITION_SL);

        // Debug: Print values for troubleshooting


        if(basket.BasketDirection == POSITION_TYPE_BUY && new_sl_price >= market_price - stop_level_dist)
        {

            continue;
        }
        if(basket.BasketDirection == POSITION_TYPE_SELL && new_sl_price <= market_price + stop_level_dist)
        {

            continue;
        }

        double current_tp = PositionGetDouble(POSITION_TP);
        double new_tp = CSettings::TslRemoveTp ? 0 : current_tp;

        // Skip modification if both SL and TP are essentially the same (prevent invalid stops on no-change)
        if (MathAbs(new_sl_price - current_sl) < 0.00001 && MathAbs(new_tp - current_tp) < 0.00001) {
            continue;
        }

        // Check for minimum meaningful difference (prevent floating-point precision issues)
        double min_diff = _Point * 2; // Minimum 2 points difference
        if(MathAbs(new_sl_price - current_sl) < min_diff) {
            continue;
        }

        if(basket.BasketDirection == POSITION_TYPE_BUY && new_sl_price <= current_sl) {
            continue;
        }
        if(basket.BasketDirection == POSITION_TYPE_SELL && (new_sl_price >= current_sl && current_sl != 0.0)) {
            continue;
        }

        m_trade.PositionModify(ticket, new_sl_price, new_tp);
    }
}

void CTrailingStopManager::HandleAtrTsl(const CBasket &basket)
{
    if(basket.Ticket == 0) return;

    double market_price = (basket.BasketDirection == POSITION_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK);

    // Calculate ATR-based trailing stop level using current market price for true trailing
    double pip_size = CMoneyManager::GetPipSize();
    double new_sl_price = m_atr_utility.GetAtrBasedLevel(market_price, CSettings::TslAtrMultiplier, basket.BasketDirection == POSITION_TYPE_BUY, pip_size);

    double stop_level_dist = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * _Point;

    // Validate ATR-based SL against broker requirements
    bool is_valid_sl = true;
    if(basket.BasketDirection == POSITION_TYPE_BUY)
    {
        // For BUY: SL must be below BID by at least stops_distance
        if(new_sl_price >= market_price - stop_level_dist)
        {
            Print("ATR TSL: Invalid SL for BUY basket. SL: ", new_sl_price, " would be too close to market price: ", market_price);
            return;
        }
    }
    else // POSITION_TYPE_SELL
    {
        // For SELL: SL must be above ASK by at least stops_distance
        if(new_sl_price <= market_price + stop_level_dist)
        {
            Print("ATR TSL: Invalid SL for SELL basket. SL: ", new_sl_price, " would be too close to market price: ", market_price);
            return;
        }
    }

    // Apply ATR-based trailing stop to all positions in basket
    for(int i = 0; i < ArraySize(basket.Tickets); i++)
    {
        ulong ticket = basket.Tickets[i];
        if(!PositionSelectByTicket(ticket)) continue;

        double current_sl = PositionGetDouble(POSITION_SL);
        double current_tp = PositionGetDouble(POSITION_TP);
        double new_tp = CSettings::TslRemoveTp ? 0 : current_tp;

        // Skip modification if both SL and TP are essentially the same (prevent invalid stops on no-change)
        if (MathAbs(new_sl_price - current_sl) < 0.00001 && MathAbs(new_tp - current_tp) < 0.00001) {
            continue;
        }

        // Check for minimum meaningful difference (prevent floating-point precision issues)
        double min_diff = _Point * 2; // Minimum 2 points difference
        if(MathAbs(new_sl_price - current_sl) < min_diff) {
            continue;
        }

        // Apply "Better Price" rule: only modify if new SL improves current SL
        if(basket.BasketDirection == POSITION_TYPE_BUY && new_sl_price <= current_sl) {
            continue;
        }
        if(basket.BasketDirection == POSITION_TYPE_SELL && (new_sl_price >= current_sl && current_sl != 0.0)) {
            continue;
        }

        Print("ATR TSL: Modifying position ", ticket, " SL from ", current_sl, " to ", new_sl_price, " (ATR multiplier: ", CSettings::TslAtrMultiplier, ")");
        if(!m_trade.PositionModify(ticket, new_sl_price, new_tp))
        {
            Print("ATR TSL: Failed to modify position ", ticket, " Error: ", m_trade.ResultRetcode());
        }
    }
}
//+------------------------------------------------------------------+