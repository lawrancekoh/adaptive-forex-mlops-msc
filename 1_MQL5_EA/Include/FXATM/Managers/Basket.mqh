//+------------------------------------------------------------------+
//|                                                       Basket.mqh |
//|                                     Copyright 2025, LAWRANCE KOH |
//|                                          lawrancekoh@outlook.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, LAWRANCE KOH"
#property link      "lawrancekoh@outlook.com"
#property version   "1.00"

// Basket structure representing the state of a trade basket
struct CBasket
    {
     int               Ticket;           // Ticket of the most recent position
     int               TradeCount;       // Total number of positions in the basket
     int               StackingCount;    // Number of stacking trades added
     double            LastTradePrice;   // Entry price of the most recent position
     double            LastTradeLots;    // Lot size of the most recent position
     double            InitialTradePrice;// Entry price of the first position
     double            TotalVolume;      // Total volume of all positions
     double            AvgEntryPrice;    // Volume-weighted average entry price
     double            TotalProfit;      // Total profit of the basket
     double            TotalCosts;       // Total costs (swap + commission)
     ENUM_POSITION_TYPE BasketDirection; // Direction of the basket (BUY/SELL)
     string            BasketType;       // Type of basket (e.g., INITIAL, DCA, STACK)
     int               SerialNumber;     // Serial number of the basket
     bool              HasPartialTPExecuted; // Flag for partial TP execution
     bool              HasStacked;       // Flag indicating if stacking has occurred
     long              Tickets[];        // Array of all ticket IDs in the basket

    // Constructor
    CBasket() : Ticket(0), TradeCount(0), StackingCount(0), LastTradePrice(0.0),
                LastTradeLots(0.0), InitialTradePrice(0.0), TotalVolume(0.0),
                AvgEntryPrice(0.0), TotalProfit(0.0), TotalCosts(0.0),
                BasketDirection(POSITION_TYPE_BUY), BasketType(""), SerialNumber(0),
                HasPartialTPExecuted(false), HasStacked(false) {
        ArrayResize(Tickets, 0);
    }

    // Method to add a ticket to the array
    void AddTicket(long ticket) {
        int size = ArraySize(Tickets);
        ArrayResize(Tickets, size + 1);
        Tickets[size] = ticket;
    }

   // Calculate profit in pips
   double ProfitPips() const
     {
      double current_price = (BasketDirection == POSITION_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double pip_size = CMoneyManager::GetPipSize();
      if (BasketDirection == POSITION_TYPE_BUY)
        {
         return (current_price - AvgEntryPrice) / pip_size;
        }
      else
        {
         return (AvgEntryPrice - current_price) / pip_size;
        }
     }
  };
//+------------------------------------------------------------------+