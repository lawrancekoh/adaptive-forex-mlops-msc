    //+------------------------------------------------------------------+
    //|                                                   DCAManager.mqh |
    //|                                     Copyright 2025, LAWRANCE KOH |
    //|                                          lawrancekoh@outlook.com |
    //+------------------------------------------------------------------+
    #property copyright "Copyright 2025, LAWRANCE KOH"
    #property link      "lawrancekoh@outlook.com"
    #property version   "1.00"

    #include "Settings.mqh"
    #include "TradeManager.mqh"
    #include "MoneyManager.mqh"

    class CDCAManager
    {
    private:
        CTradeManager* m_trade_manager;
        CMoneyManager* m_money_manager;

    public:
        CDCAManager(void){};
        ~CDCAManager(void){};

        void SetTradeManager(CTradeManager* tm) { m_trade_manager = tm; }
        void SetMoneyManager(CMoneyManager* mm) { m_money_manager = mm; }

        void Init()
        {
            // Nothing to do here for now
        }

        void ManageDCA(ENUM_POSITION_TYPE direction, const CBasket &basket)
        {
            // DCA Guard Clauses - check first before any modifications
            if (CSettings::DcaMaxTrades <= 0 || basket.Ticket == 0) return; // DCA disabled or no basket
            if (!m_money_manager.CheckDrawdown()) return; // Risk check: high drawdown blocks DCA
            if (basket.TradeCount >= CSettings::DcaMaxTrades) return; // Max trades reached

            double pip_size = CMoneyManager::GetPipSize();

            // Get current market price for drawdown calculation
            double market_price = (basket.BasketDirection == POSITION_TYPE_BUY) ?
                                SymbolInfoDouble(_Symbol, SYMBOL_BID) :
                                SymbolInfoDouble(_Symbol, SYMBOL_ASK);

            // Calculate required drawdown pips for this DCA level (increases with each trade)
            double required_drawdown_pips = CSettings::DcaTriggerPips;
            for(int i = 1; i < basket.TradeCount; i++)
            {
                required_drawdown_pips *= CSettings::DcaStepMultiplier;
            }

            // Calculate actual drawdown in pips from last trade
            double drawdown_pips = 0;
            if (basket.BasketDirection == POSITION_TYPE_BUY)
                drawdown_pips = (basket.LastTradePrice - market_price) / pip_size;
            else
                drawdown_pips = (market_price - basket.LastTradePrice) / pip_size;

            // Check if drawdown meets DCA trigger threshold
            if (drawdown_pips < required_drawdown_pips) return;

            // Calculate DCA Lot Size (apply multiplier after certain trades)
            double dca_lot;
            if (basket.TradeCount >= CSettings::DcaLotMultiplierStart)
            {
                dca_lot = basket.LastTradeLots * CSettings::DcaLotMultiplier; // Increase lot size
            }
            else
            {
                dca_lot = basket.LastTradeLots; // Same lot size as previous trade
            }
            dca_lot = m_money_manager.ValidateLotSize(dca_lot); // Ensure broker compliance

            // Execute DCA Trade in same direction as basket
            int signal = (basket.BasketDirection == POSITION_TYPE_BUY) ? SIGNAL_BUY : SIGNAL_SELL;
            m_trade_manager.OpenTrade(signal, dca_lot, CSettings::SlPips, 0, "DCA", basket.TradeCount + 1);

            // Refresh basket cache to include the new DCA position
            m_trade_manager.Refresh();

            // Fetch updated basket to ensure we use the new AvgEntryPrice
            CBasket updated_basket = m_trade_manager.GetCachedBasket(direction);

            // Update basket SL/TP after successful DCA trade (uniform risk management)
            // Use updated_basket to be consistent, though InitialTradePrice should be invariant
            double basket_sl_price = updated_basket.InitialTradePrice + (direction == POSITION_TYPE_BUY ? -CSettings::SlPips * pip_size : CSettings::SlPips * pip_size);
            m_trade_manager.SetBasketSL(direction, basket_sl_price); // Set SL for entire basket

            // Set uniform basket TP if enabled
            if(CSettings::BasketTpPips > 0)
            {
                // Use updated_basket.AvgEntryPrice which includes the new DCA trade
                double basket_tp_price = updated_basket.AvgEntryPrice + (direction == POSITION_TYPE_BUY ? CSettings::BasketTpPips * pip_size : -CSettings::BasketTpPips * pip_size);
                m_trade_manager.SetBasketTP(direction, basket_tp_price); // Set TP for entire basket
            }
        }
    };
    //+------------------------------------------------------------------+