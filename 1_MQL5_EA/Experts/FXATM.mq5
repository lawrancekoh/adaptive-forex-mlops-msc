//+------------------------------------------------------------------+
//|                                                        FXATM.mq5 |
//|                                FX Automated Trading Manager v4.0 |
//|                             Advanced Multi-Signal Expert Advisor |
//|                                     Copyright 2025, LAWRANCE KOH |
//|                                          lawrancekoh@outlook.com |
//+------------------------------------------------------------------+
//| PURPOSE:                                                         |
//|   Comprehensive automated trading system for MetaTrader 5        |
//|   featuring multi-signal aggregation, advanced risk management,  |
//|   and adaptive position sizing.                                  |
//|                                                                  |
//| KEY FEATURES:                                                    |
//|   • 3-Slot Polymorphic Signal System (MACD, RSI, ATR, etc.)      |
//|   • 6 Lot Sizing Modes (Fixed, Risk%, ATR-Volatility Adjusted)   |
//|   • 5 Trailing Stop Loss Modes (Step, ATR, MA, High/Low)         |
//|   • DCA & Stacking for basket expansion                          |
//|   • Partial Take Profit with True Break-Even                     |
//|   • News & Time filtering for risk control                       |
//|   • Chart UI with manual trading controls                        |
//|                                                                  |
//| REQUIREMENTS:                                                    |
//|   • MetaTrader 5 build 3280+                                     |
//|   • Allow WebRequest for news filtering                          |
//|   • Add https://nfs.forexfactory.net to allowed URLs             |
//|                                                                  |
//| VERSION HISTORY:                                                 |
//|   4.00 - ATR-based features (TSL, lot sizing)                    |
//|   3.00 - Multi-signal system, advanced basket management         |
//|   2.00 - DCA and trailing stop implementation                    |
//|   1.00 - Initial release with basic signal processing            |
//+------------------------------------------------------------------+
#property copyright   "Copyright 2025, LAWRANCE KOH"
#property link        "lawrancekoh@outlook.com"
#property version     "4.00"
#property description "FXATM v4.0 - Advanced Multi-Signal Expert Advisor with ATR-based features"

#include <FXATM/Managers/Settings.mqh>
#include <FXATM/Managers/TradeManager.mqh>
#include <FXATM/Managers/MoneyManager.mqh>
#include <FXATM/Managers/SignalManager.mqh>
#include <FXATM/Signals/CSignal_MACD.mqh>
#include <FXATM/Signals/CSignal_RSI.mqh>
#include <FXATM/Signals/CSignal_MA.mqh>
#include <FXATM/Signals/CSignal_Stochastic.mqh>
#include <FXATM/Signals/CSignal_BollingerBands.mqh>
#include <FXATM/Managers/DCAManager.mqh>
#include <FXATM/Managers/TrailingStopManager.mqh>
#include <FXATM/Managers/TimeManager.mqh>
#include <FXATM/Managers/NewsManager.mqh>
#include <FXATM/Managers/StackingManager.mqh>
#include <FXATM/Managers/UIManager.mqh>
#include <FXATM/Managers/CatrUtility.mqh>

// --- Manager Instances ---
CTradeManager*             g_trade_manager;
CMoneyManager*             g_money_manager;
CSignalManager*            g_signal_manager;
CDCAManager*               g_dca_manager;
CTrailingStopManager*      g_tsl_manager;
CTimeManager*              g_time_manager;
CNewsManager*              g_news_manager;
CStackingManager*          g_stacking_manager;
CUIManager*                g_ui_manager;
CatrUtility*               g_atr_utility;

// A. GENERAL SETTINGS
input group "******** GENERAL SETTINGS ********";
input string   InpEaName = "FXATMv4";                                  // EA name for display purposes
input long     InpEaMagicNumber = 123456;                              // Unique ID for EA's trades
input int      InpMaxSpreadPoints = 40;                                // Max allowed spread in POINTS for new trades
input int      InpMaxSlippagePoints = 10;                              // Max allowed slippage in POINTS for all trades
input double   InpMaxDrawdownPercent = 50.0;                           // Max drawdown % before stopping new trades (set to 100 or higher to disable)
input ENUM_TIMEFRAMES InpEaHeartbeatTimeframe = PERIOD_M15;            // Timeframe for heartbeat (new bar check)
input bool     InpAllowLongTrades = true;                              // Allow BUY (long) trades
input bool     InpAllowShortTrades = true;                             // Allow SELL (short) trades

// B. POSITION MANAGEMENT SETTINGS
input group "******** POSITION MANAGEMENT SETTINGS ********";
input ENUM_LOT_SIZING_MODE InpLotSizingMode = MODE_FIXED_LOT;          // Lot sizing calculation method
input double   InpLotFixed = 0.04;                                     // Lot size for Fixed Lot mode
input double   InpLotsPerThousand = 0.01;                              // Lots per 1000 units of balance/equity
input double   InpLotRiskPercent = 1.0;                                // Risk % for Balance/Equity modes
input int      InpSlPips = 500;                                        // Initial SL pips (0 = no SL, disables risk modes)
input int      InpInitialTpPips = 42;                                  // Initial TP in pips (0 = no TP)
// Basket TP/SL moved here for complete position lifecycle management
input int      InpBasketTpPips = 26;                                   // Basket TP in pips when basket has >1 position (0 = disabled)

// C. LOSS MANAGEMENT SETTINGS
input group "******** LOSS MANAGEMENT SETTINGS ********";
input int      InpDcaMaxTrades = 10;                                   // Max number of DCA trades allowed (0 = disabled)
input int      InpDcaTriggerPips = 21;                                 // Initial pips in drawdown to add first DCA trade
input double   InpDcaStepMultiplier = 1.1;                             // Step multiplier for subsequent DCA trades
input double   InpDcaLotMultiplier = 1.5;                              // Lot multiplier for next DCA trade
input int      InpDcaLotMultiplierStart = 2;                           // Multiplier starts from this trade number (e.g., 3rd trade)

// D. PROFIT MANAGEMENT SETTINGS
input group "******** PROFIT MANAGEMENT SETTINGS ********";
input ENUM_TSL_MODE InpTslMode = MODE_TSL_STEP;                        // Trailing stop loss mode

// E1. Trigger and Steps Settings
input int      InpTslBeTriggerPips = 13;                               // Pips in profit to trigger break-even
input int      InpBeOffsetPips = 3;                                    // Pips *past* entry to set SL for BE
input int      InpTslStepPips = 10;                                    // TSL Step in pips
input bool     InpTslRemoveTp = true;                                  // Remove TP when TSL triggers
input bool     InpBreakevenIncludesCosts = true;                       // 'True BE' accounts for swap & commission
input double   InpCommissionPerLot = 0.0;                              // Commission per lot for True BE calculations

// E2. ATR Settings
input int      InpTslAtrPeriod = 14;                                   // ATR period for TSL
input double   InpTslAtrMultiplier = 2.5;                              // ATR multiplier for TSL distance

// E3. Moving Average Settings
input int      InpTslMaPeriod = 20;                                    // Moving Average period for TSL
input ENUM_MA_METHOD InpTslMaMethod = MODE_SMA;                        // Moving Average method for TSL
input ENUM_APPLIED_PRICE InpTslMaPrice = PRICE_CLOSE;                  // Price to apply MA to for TSL

// E4. High/Low Bar Settings
input int      InpTslHiLoPeriod = 10;                                  // Period to look back for High/Low TSL

// E5. Stacking Settings
// Stacking settings moved here as part of profit management
input int      InpStackingMaxTrades = 3;                               // Max number of Stacking trades (0 = disabled)
input int      InpStackingTriggerPips = 50;                            // Fixed pips trigger for stacking trades
input double   InpStackingLotSize = 0.01;                              // Lot size for stacking trades
input ENUM_STACKING_LOT_MODE InpStackingLotMode = MODE_FIXED;          // Stacking lot sizing mode

// E. ADVANCED EXIT SETTINGS
input group "******** ADVANCED EXIT SETTINGS ********";
input int      InpPartialTpTriggerPips = 13;                           // Pips in profit to trigger partial close (0 = disabled)
input double   InpPartialTpClosePercent = 50.0;                        // Percentage of volume to close
input bool     InpPartialTpSetBe = true;                               // Set remaining position to BE after partial close?

// F. FILTER SETTINGS
input group "******** TIME FILTER SETTINGS ********";
input string   InpEaTradingDays = "1,2,3,4,5";                         // Allowed trading days (Mon=1...Fri=5)
input string   InpEaTradingTimeStart = "00:00";                        // Trading start time (Broker time)
input string   InpEaTradingTimeEnd = "23:59";                          // Trading end time (Broker time)

// G. NEWS FILTER SETTINGS
input group "******** NEWS FILTER SETTINGS ********";
input ENUM_NEWS_SOURCE InpNewsSourceMode = MODE_DISABLED;              // Source for news event data (MODE_DISABLED = off)
input string   InpNewsCalendarURL = "https://nfs.forexfactory.net/ffcal_week_this.csv"; // URL for web request mode
input int      InpNewsMinsBefore = 30;                                 // Block trading X minutes before news
input int      InpNewsMinsAfter = 30;                                  // Block trading X minutes after news
input bool     InpNewsFilterHighImpact = true;                         // Filter high-impact news
input bool     InpNewsFilterMedImpact = false;                         // Filter medium-impact news
input bool     InpNewsFilterLowImpact = false;                         // Filter low-impact news
input string   InpNewsFilterCurrencies = "USD,EUR,GBP,JPY,CAD,AUD,NZD,CHF"; // Currencies to monitor for news

// J. SIGNAL SETTINGS
input group "******** SIGNAL DEFINITIONS ********";
input group "******** SIGNAL SLOT 1 ********";
input ENUM_SIGNAL_TYPE      InpSignal1_Type = SIGNAL_MACD;              // Signal type for slot 1
input ENUM_SIGNAL_ROLE      InpSignal1_Role = ROLE_ENTRY;               // Role for signal 1
input ENUM_TIMEFRAMES       InpSignal1_Timeframe = PERIOD_M15;          // Timeframe for signal 1
input int                   InpSignal1_IntParam0 = 12;                  // Int param 0 (e.g., MACD Fast)
input int                   InpSignal1_IntParam1 = 26;                  // Int param 1 (e.g., MACD Slow)
input int                   InpSignal1_IntParam2 = 9;                   // Int param 2 (e.g., MACD Signal)
input int                   InpSignal1_IntParam3 = 0;                   // Int param 3 (reserved)
input double                InpSignal1_DoubleParam0 = 0.0;              // Double param 0 (reserved)
input double                InpSignal1_DoubleParam1 = 0.0;              // Double param 1 (reserved)
input double                InpSignal1_DoubleParam2 = 0.0;              // Double param 2 (reserved)
input double                InpSignal1_DoubleParam3 = 0.0;              // Double param 3 (reserved)
input ENUM_APPLIED_PRICE    InpSignal1_Price = PRICE_CLOSE;             // Applied price
input ENUM_MA_METHOD        InpSignal1_MaMethod1 = MODE_SMA;            // MA method 1
input ENUM_MA_METHOD        InpSignal1_MaMethod2 = MODE_SMA;            // MA method 2
input ENUM_STO_PRICE        InpSignal1_PriceField = STO_LOWHIGH;        // Stochastic price field
input bool                  InpSignal1_BoolParam0 = false;              // Bool param 0 (e.g., Threshold check)
input bool                  InpSignal1_BoolParam1 = false;              // Bool param 1 (e.g., Threshold reverse)
input bool                  InpSignal1_BoolParam2 = false;              // Bool param 2 (reserved)
input bool                  InpSignal1_BoolParam3 = false;              // Bool param 3 (reserved)
input group "******** SIGNAL SLOT 2 ********";
input ENUM_SIGNAL_TYPE      InpSignal2_Type = SIGNAL_MACD;              // Signal type for slot 2
input ENUM_SIGNAL_ROLE      InpSignal2_Role = ROLE_BIAS;                // Role for signal 2
input ENUM_TIMEFRAMES       InpSignal2_Timeframe = PERIOD_H1;           // Timeframe for signal 2
input int                   InpSignal2_IntParam0 = 12;                  // Int param 0
input int                   InpSignal2_IntParam1 = 26;                  // Int param 1
input int                   InpSignal2_IntParam2 = 9;                   // Int param 2
input int                   InpSignal2_IntParam3 = 0;                   // Int param 3
input double                InpSignal2_DoubleParam0 = 0.0;              // Double param 0
input double                InpSignal2_DoubleParam1 = 0.0;              // Double param 1
input double                InpSignal2_DoubleParam2 = 0.0;              // Double param 2
input double                InpSignal2_DoubleParam3 = 0.0;              // Double param 3
input ENUM_APPLIED_PRICE    InpSignal2_Price = PRICE_CLOSE;             // Applied price
input ENUM_MA_METHOD        InpSignal2_MaMethod1 = MODE_SMA;            // MA method 1
input ENUM_MA_METHOD        InpSignal2_MaMethod2 = MODE_SMA;            // MA method 2
input ENUM_STO_PRICE        InpSignal2_PriceField = STO_LOWHIGH;        // Stochastic price field
input bool                  InpSignal2_BoolParam0 = false;              // Bool param 0
input bool                  InpSignal2_BoolParam1 = false;              // Bool param 1
input bool                  InpSignal2_BoolParam2 = false;              // Bool param 2
input bool                  InpSignal2_BoolParam3 = false;              // Bool param 3
input group "******** SIGNAL SLOT 3 ********";
input ENUM_SIGNAL_TYPE      InpSignal3_Type = SIGNAL_RSI;               // Signal type for slot 3
input ENUM_SIGNAL_ROLE      InpSignal3_Role = ROLE_ENTRY;               // Role for signal 3
input ENUM_TIMEFRAMES       InpSignal3_Timeframe = PERIOD_M15;          // Timeframe for signal 3
input int                   InpSignal3_IntParam0 = 14;                  // Int param 0
input int                   InpSignal3_IntParam1 = 0;                   // Int param 1
input int                   InpSignal3_IntParam2 = 0;                   // Int param 2
input int                   InpSignal3_IntParam3 = 0;                   // Int param 3
input double                InpSignal3_DoubleParam0 = 30.0;             // Double param 0
input double                InpSignal3_DoubleParam1 = 70.0;             // Double param 1
input double                InpSignal3_DoubleParam2 = 0.0;              // Double param 2
input double                InpSignal3_DoubleParam3 = 0.0;              // Double param 3
input ENUM_APPLIED_PRICE    InpSignal3_Price = PRICE_CLOSE;             // Applied price
input ENUM_MA_METHOD        InpSignal3_MaMethod1 = MODE_SMA;            // MA method 1
input ENUM_MA_METHOD        InpSignal3_MaMethod2 = MODE_SMA;            // MA method 2
input ENUM_STO_PRICE        InpSignal3_PriceField = STO_LOWHIGH;        // Stochastic price field
input bool                  InpSignal3_BoolParam0 = false;              // Bool param 0
input bool                  InpSignal3_BoolParam1 = false;              // Bool param 1
input bool                  InpSignal3_BoolParam2 = false;              // Bool param 2
input bool                  InpSignal3_BoolParam3 = false;              // Bool param 3

// M. SIGNAL MANAGER SETTINGS
input group "******** SIGNAL MANAGER SETTINGS ********";
input int     InpBiasPersistenceBars = 24;                             // Bars (EA heartbeat intervals) bias persists before auto-reset

// K. CHART UI SETTINGS
input group "******** CHART UI SETTINGS ********";
input bool     InpChartShowPanels = true;                              // Show/hide the chart UI panel
input ENUM_BASE_CORNER InpChartPanelCorner = CORNER_RIGHT_LOWER;       // Corner to display the UI panel
input color    InpChartColorBackground = clrBlack;                     // Background color of the UI panel
input color    InpChartColorTextMain = clrWhite;                       // Main text color for the UI
input color    InpChartColorBuy = clrDodgerBlue;                       // Color for BUY status/buttons
input color    InpChartColorSell = clrRed;                             // Color for SELL status/buttons
input color    InpChartColorNeutral = clrGray;                         // Color for NEUTRAL status

//+------------------------------------------------------------------+
//| Helper Functions                                                 |
//+------------------------------------------------------------------+
bool IsNewBar(const ENUM_TIMEFRAMES timeframe)
{
    static datetime previousTime = 0;
    datetime currentTime = iTime(_Symbol, timeframe, 0);
    if(previousTime != currentTime)
    {
        previousTime = currentTime;
        return true;
    }
    return false;
}

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
    //--- Create manager instances
    g_trade_manager = new CTradeManager();
    g_money_manager = new CMoneyManager();
    g_signal_manager = new CSignalManager();
    g_dca_manager = new CDCAManager();
    g_tsl_manager = new CTrailingStopManager();
    g_time_manager = new CTimeManager();
    g_news_manager = new CNewsManager();
    g_stacking_manager = new CStackingManager();
    g_ui_manager = new CUIManager();
    g_atr_utility = new CatrUtility();

    //--- Copy input values to CSettings (reordered to match new logical grouping)
    // A. General Settings
    CSettings::EaName = InpEaName;
    CSettings::EaMagicNumber = InpEaMagicNumber;
    CSettings::MaxSpreadPoints = InpMaxSpreadPoints;
    CSettings::MaxSlippagePoints = InpMaxSlippagePoints;
    CSettings::MaxDrawdownPercent = InpMaxDrawdownPercent;
    CSettings::EaHeartbeatTimeframe = InpEaHeartbeatTimeframe;
    CSettings::AllowLongTrades = InpAllowLongTrades;
    CSettings::AllowShortTrades = InpAllowShortTrades;
    CSettings::Symbol = _Symbol;

    // B. Position Management Settings (Lot sizing + Basket TP/SL)
    CSettings::LotSizingMode = InpLotSizingMode;
    CSettings::LotFixed = InpLotFixed;
    CSettings::LotsPerThousand = InpLotsPerThousand;
    CSettings::LotRiskPercent = InpLotRiskPercent;
    CSettings::SlPips = InpSlPips;
    CSettings::InitialTpPips = InpInitialTpPips;
    CSettings::BasketTpPips = InpBasketTpPips;

    // C. Loss Management Settings (DCA)
    CSettings::DcaMaxTrades = InpDcaMaxTrades;
    CSettings::DcaTriggerPips = InpDcaTriggerPips;
    CSettings::DcaStepMultiplier = InpDcaStepMultiplier;
    CSettings::DcaLotMultiplier = InpDcaLotMultiplier;
    CSettings::DcaLotMultiplierStart = InpDcaLotMultiplierStart;

    // D. Profit Management Settings (TSL + Stacking)
    CSettings::TslMode = InpTslMode;
    CSettings::TslBeTriggerPips = InpTslBeTriggerPips;
    CSettings::BeOffsetPips = InpBeOffsetPips;
    CSettings::TslStepPips = InpTslStepPips;
    CSettings::TslRemoveTp = InpTslRemoveTp;
    CSettings::BreakevenIncludesCosts = InpBreakevenIncludesCosts;
    CSettings::CommissionPerLot = InpCommissionPerLot;
    CSettings::TslAtrPeriod = InpTslAtrPeriod;
    CSettings::TslAtrMultiplier = InpTslAtrMultiplier;
    CSettings::TslMaPeriod = InpTslMaPeriod;
    CSettings::TslMaMethod = InpTslMaMethod;
    CSettings::TslMaPrice = InpTslMaPrice;
    CSettings::TslHiLoPeriod = InpTslHiLoPeriod;
    CSettings::StackingMaxTrades = InpStackingMaxTrades;
    CSettings::StackingTriggerPips = InpStackingTriggerPips;
    CSettings::StackingLotSize = InpStackingLotSize;
    CSettings::StackingLotMode = InpStackingLotMode;

    // E. Advanced Exit Settings (Partial TP)
    CSettings::PartialTpTriggerPips = InpPartialTpTriggerPips;
    CSettings::PartialTpClosePercent = InpPartialTpClosePercent;
    CSettings::PartialTpSetBe = InpPartialTpSetBe;

    // F. Filter Settings (Time + News)
    CSettings::EaTradingDays = InpEaTradingDays;
    CSettings::EaTradingTimeStart = InpEaTradingTimeStart;
    CSettings::EaTradingTimeEnd = InpEaTradingTimeEnd;
    CSettings::NewsSourceMode = InpNewsSourceMode;
    CSettings::NewsCalendarURL = InpNewsCalendarURL;
    CSettings::NewsMinsBefore = InpNewsMinsBefore;
    CSettings::NewsMinsAfter = InpNewsMinsAfter;
    CSettings::NewsFilterHighImpact = InpNewsFilterHighImpact;
    CSettings::NewsFilterMedImpact = InpNewsFilterMedImpact;
    CSettings::NewsFilterLowImpact = InpNewsFilterLowImpact;
    CSettings::NewsFilterCurrencies = InpNewsFilterCurrencies;

    CSettings::Signal1.Type = InpSignal1_Type;
    CSettings::Signal1.Role = InpSignal1_Role;
    CSettings::Signal1.Timeframe = InpSignal1_Timeframe;
    CSettings::Signal1.Params.IntParams[0] = InpSignal1_IntParam0;
    CSettings::Signal1.Params.IntParams[1] = InpSignal1_IntParam1;
    CSettings::Signal1.Params.IntParams[2] = InpSignal1_IntParam2;
    CSettings::Signal1.Params.IntParams[3] = InpSignal1_IntParam3;
    CSettings::Signal1.Params.DoubleParams[0] = InpSignal1_DoubleParam0;
    CSettings::Signal1.Params.DoubleParams[1] = InpSignal1_DoubleParam1;
    CSettings::Signal1.Params.DoubleParams[2] = InpSignal1_DoubleParam2;
    CSettings::Signal1.Params.DoubleParams[3] = InpSignal1_DoubleParam3;
    CSettings::Signal1.Params.Price = InpSignal1_Price;
    CSettings::Signal1.Params.MaMethod1 = InpSignal1_MaMethod1;
    CSettings::Signal1.Params.MaMethod2 = InpSignal1_MaMethod2;
    CSettings::Signal1.Params.PriceField = InpSignal1_PriceField;
    CSettings::Signal1.Params.BoolParams[0] = InpSignal1_BoolParam0;
    CSettings::Signal1.Params.BoolParams[1] = InpSignal1_BoolParam1;
    CSettings::Signal1.Params.BoolParams[2] = InpSignal1_BoolParam2;
    CSettings::Signal1.Params.BoolParams[3] = InpSignal1_BoolParam3;

    CSettings::Signal2.Type = InpSignal2_Type;
    CSettings::Signal2.Role = InpSignal2_Role;
    CSettings::Signal2.Timeframe = InpSignal2_Timeframe;
    CSettings::Signal2.Params.IntParams[0] = InpSignal2_IntParam0;
    CSettings::Signal2.Params.IntParams[1] = InpSignal2_IntParam1;
    CSettings::Signal2.Params.IntParams[2] = InpSignal2_IntParam2;
    CSettings::Signal2.Params.IntParams[3] = InpSignal2_IntParam3;
    CSettings::Signal2.Params.DoubleParams[0] = InpSignal2_DoubleParam0;
    CSettings::Signal2.Params.DoubleParams[1] = InpSignal2_DoubleParam1;
    CSettings::Signal2.Params.DoubleParams[2] = InpSignal2_DoubleParam2;
    CSettings::Signal2.Params.DoubleParams[3] = InpSignal2_DoubleParam3;
    CSettings::Signal2.Params.Price = InpSignal2_Price;
    CSettings::Signal2.Params.MaMethod1 = InpSignal2_MaMethod1;
    CSettings::Signal2.Params.MaMethod2 = InpSignal2_MaMethod2;
    CSettings::Signal2.Params.PriceField = InpSignal2_PriceField;
    CSettings::Signal2.Params.BoolParams[0] = InpSignal2_BoolParam0;
    CSettings::Signal2.Params.BoolParams[1] = InpSignal2_BoolParam1;
    CSettings::Signal2.Params.BoolParams[2] = InpSignal2_BoolParam2;
    CSettings::Signal2.Params.BoolParams[3] = InpSignal2_BoolParam3;

    CSettings::Signal3.Type = InpSignal3_Type;
    CSettings::Signal3.Role = InpSignal3_Role;
    CSettings::Signal3.Timeframe = InpSignal3_Timeframe;
    CSettings::Signal3.Params.IntParams[0] = InpSignal3_IntParam0;
    CSettings::Signal3.Params.IntParams[1] = InpSignal3_IntParam1;
    CSettings::Signal3.Params.IntParams[2] = InpSignal3_IntParam2;
    CSettings::Signal3.Params.IntParams[3] = InpSignal3_IntParam3;
    CSettings::Signal3.Params.DoubleParams[0] = InpSignal3_DoubleParam0;
    CSettings::Signal3.Params.DoubleParams[1] = InpSignal3_DoubleParam1;
    CSettings::Signal3.Params.DoubleParams[2] = InpSignal3_DoubleParam2;
    CSettings::Signal3.Params.DoubleParams[3] = InpSignal3_DoubleParam3;
    CSettings::Signal3.Params.Price = InpSignal3_Price;
    CSettings::Signal3.Params.MaMethod1 = InpSignal3_MaMethod1;
    CSettings::Signal3.Params.MaMethod2 = InpSignal3_MaMethod2;
    CSettings::Signal3.Params.PriceField = InpSignal3_PriceField;
    CSettings::Signal3.Params.BoolParams[0] = InpSignal3_BoolParam0;
    CSettings::Signal3.Params.BoolParams[1] = InpSignal3_BoolParam1;
    CSettings::Signal3.Params.BoolParams[2] = InpSignal3_BoolParam2;
    CSettings::Signal3.Params.BoolParams[3] = InpSignal3_BoolParam3;

    CSettings::BiasPersistenceBars = InpBiasPersistenceBars;

    CSettings::ChartShowPanels = InpChartShowPanels;
    CSettings::ChartPanelCorner = InpChartPanelCorner;
    CSettings::ChartColorBackground = InpChartColorBackground;
    CSettings::ChartColorTextMain = InpChartColorTextMain;
    CSettings::ChartColorBuy = InpChartColorBuy;
    CSettings::ChartColorSell = InpChartColorSell;
    CSettings::ChartColorNeutral = InpChartColorNeutral;

    //--- Detect backtest mode for magic number handling
    CSettings::IsBacktestMode = (MQLInfoInteger(MQL_TESTER) || MQLInfoInteger(MQL_OPTIMIZATION));
    Print("FXATM: Backtest mode detected: ", CSettings::IsBacktestMode);

    //--- Initialize managers (after CSettings is set)
    g_trade_manager.Init();
    g_tsl_manager.Init();
    g_time_manager.Init();
    g_news_manager.Init();
    g_dca_manager.SetTradeManager(g_trade_manager);
    g_dca_manager.SetMoneyManager(g_money_manager);
    g_stacking_manager.SetMoneyManager(g_money_manager);
    g_stacking_manager.SetTradeManager(g_trade_manager);

    // Initialize ATR utility and inject into dependent managers
    if (!g_atr_utility.Init(CSettings::TslAtrPeriod, PERIOD_CURRENT))
    {
        Print("Failed to initialize ATR utility");
        return INIT_FAILED;
    }
    g_money_manager.SetAtrUtility(g_atr_utility);

    //--- Signal Instantiation (up to 3 configurable signals)
    // Signal1: Instantiate based on type
    switch(CSettings::Signal1.Type)
    {
        case SIGNAL_MACD:
        {
            ISignal* signal = new CSignal_MACD();
            if(signal.Init(CSettings::Signal1))
            {
                g_signal_manager.AddSignal(signal);
            }
            else
            {
                Print("Failed to initialize MACD signal for Signal1");
                delete signal;
            }
            break;
        }
        case SIGNAL_RSI:
        {
            ISignal* signal = new CSignal_RSI();
            if(signal.Init(CSettings::Signal1))
            {
                g_signal_manager.AddSignal(signal);
            }
            else
            {
                Print("Failed to initialize RSI signal for Signal1");
                delete signal;
            }
            break;
        }
        case SIGNAL_MA_CROSS:
        {
            ISignal* signal = new CSignal_MA();
            if(signal.Init(CSettings::Signal1))
            {
                g_signal_manager.AddSignal(signal);
            }
            else
            {
                Print("Failed to initialize MA signal for Signal1");
                delete signal;
            }
            break;
        }
        case SIGNAL_STOCHASTIC:
        {
            ISignal* signal = new CSignal_Stochastic();
            if(signal.Init(CSettings::Signal1))
            {
                g_signal_manager.AddSignal(signal);
            }
            else
            {
                Print("Failed to initialize Stochastic signal for Signal1");
                delete signal;
            }
            break;
        }
        case SIGNAL_BOLLINGER_BANDS:
        {
            ISignal* signal = new CSignal_BollingerBands();
            if(signal.Init(CSettings::Signal1))
            {
                g_signal_manager.AddSignal(signal);
            }
            else
            {
                Print("Failed to initialize Bollinger Bands signal for Signal1");
                delete signal;
            }
            break;
        }
        default:
            Print("Unsupported signal type for Signal1");
            break;
    }

    // Signal2
    switch(CSettings::Signal2.Type)
    {
        case SIGNAL_MACD:
        {
            ISignal* signal = new CSignal_MACD();
            if(signal.Init(CSettings::Signal2))
            {
                g_signal_manager.AddSignal(signal);
            }
            else
            {
                Print("Failed to initialize MACD signal for Signal2");
                delete signal;
            }
            break;
        }
        case SIGNAL_RSI:
        {
            ISignal* signal = new CSignal_RSI();
            if(signal.Init(CSettings::Signal2))
            {
                g_signal_manager.AddSignal(signal);
            }
            else
            {
                Print("Failed to initialize RSI signal for Signal2");
                delete signal;
            }
            break;
        }
        case SIGNAL_MA_CROSS:
        {
            ISignal* signal = new CSignal_MA();
            if(signal.Init(CSettings::Signal2))
            {
                g_signal_manager.AddSignal(signal);
            }
            else
            {
                Print("Failed to initialize MA signal for Signal2");
                delete signal;
            }
            break;
        }
        case SIGNAL_STOCHASTIC:
        {
            ISignal* signal = new CSignal_Stochastic();
            if(signal.Init(CSettings::Signal2))
            {
                g_signal_manager.AddSignal(signal);
            }
            else
            {
                Print("Failed to initialize Stochastic signal for Signal2");
                delete signal;
            }
            break;
        }
        case SIGNAL_BOLLINGER_BANDS:
        {
            ISignal* signal = new CSignal_BollingerBands();
            if(signal.Init(CSettings::Signal2))
            {
                g_signal_manager.AddSignal(signal);
            }
            else
            {
                Print("Failed to initialize Bollinger Bands signal for Signal2");
                delete signal;
            }
            break;
        }
        case SIGNAL_TYPE_NONE:
            // No signal to instantiate
            break;
        default:
            Print("Unsupported signal type for Signal2");
            break;
    }

    // Signal3
    switch(CSettings::Signal3.Type)
    {
        case SIGNAL_MACD:
        {
            ISignal* signal = new CSignal_MACD();
            if(signal.Init(CSettings::Signal3))
            {
                g_signal_manager.AddSignal(signal);
            }
            else
            {
                Print("Failed to initialize MACD signal for Signal3");
                delete signal;
            }
            break;
        }
        case SIGNAL_RSI:
        {
            ISignal* signal = new CSignal_RSI();
            if(signal.Init(CSettings::Signal3))
            {
                g_signal_manager.AddSignal(signal);
            }
            else
            {
                Print("Failed to initialize RSI signal for Signal3");
                delete signal;
            }
            break;
        }
        case SIGNAL_MA_CROSS:
        {
            ISignal* signal = new CSignal_MA();
            if(signal.Init(CSettings::Signal3))
            {
                g_signal_manager.AddSignal(signal);
            }
            else
            {
                Print("Failed to initialize MA signal for Signal3");
                delete signal;
            }
            break;
        }
        case SIGNAL_STOCHASTIC:
        {
            ISignal* signal = new CSignal_Stochastic();
            if(signal.Init(CSettings::Signal3))
            {
                g_signal_manager.AddSignal(signal);
            }
            else
            {
                Print("Failed to initialize Stochastic signal for Signal3");
                delete signal;
            }
            break;
        }
        case SIGNAL_BOLLINGER_BANDS:
        {
            ISignal* signal = new CSignal_BollingerBands();
            if(signal.Init(CSettings::Signal3))
            {
                g_signal_manager.AddSignal(signal);
            }
            else
            {
                Print("Failed to initialize Bollinger Bands signal for Signal3");
                delete signal;
            }
            break;
        }
        case SIGNAL_TYPE_NONE:
            // No signal to instantiate
            break;
        default:
            Print("Unsupported signal type for Signal3");
            break;
    }

    //---
    return(INIT_SUCCEEDED);
}
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    //--- Delete manager instances
    delete g_trade_manager;
    delete g_money_manager;
    delete g_signal_manager;
    delete g_dca_manager;
    delete g_tsl_manager;
    delete g_time_manager;
    delete g_news_manager;
    delete g_stacking_manager;
    delete g_ui_manager;
    delete g_atr_utility;
    //---
}
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
    // --- Refresh basket cache once per tick for performance ---
    g_trade_manager.Refresh();

    // --- MANAGEMENT LOGIC (runs on every tick for open baskets) ---
    // Manage BUY basket: PTP, TSL, then Stacking or DCA
    if (g_trade_manager.HasCachedBasket(POSITION_TYPE_BUY))
    {
        CBasket buy_basket = g_trade_manager.GetCachedBasket(POSITION_TYPE_BUY);
        if(buy_basket.Ticket > 0)
        {
            g_trade_manager.ManagePartialTP(buy_basket);
            // Refresh basket after PTP (positions may have been partially closed)
            g_trade_manager.Refresh();
            buy_basket = g_trade_manager.GetCachedBasket(POSITION_TYPE_BUY);
            g_tsl_manager.ManageBasketTSL(POSITION_TYPE_BUY, buy_basket);
            if (g_trade_manager.IsStopLossProfitable(buy_basket))
            {
                g_stacking_manager.ManageStacking(POSITION_TYPE_BUY, buy_basket);
                g_trade_manager.Refresh(); // Refresh cache after stacking to include new position
                buy_basket = g_trade_manager.GetCachedBasket(POSITION_TYPE_BUY);
                g_tsl_manager.ManageBasketTSL(POSITION_TYPE_BUY, buy_basket); // update TSL for expanded basket
            }
            else if (!buy_basket.HasStacked)
            {
                g_dca_manager.ManageDCA(POSITION_TYPE_BUY, buy_basket);
                g_trade_manager.Refresh(); // Refresh cache after DCA to include new position
                buy_basket = g_trade_manager.GetCachedBasket(POSITION_TYPE_BUY);
                g_tsl_manager.ManageBasketTSL(POSITION_TYPE_BUY, buy_basket); // update TSL for expanded basket
            }
            g_trade_manager.ManageBasketTP(buy_basket); // Set basket TP if expanded
        }
    }

    // Manage SELL basket: PTP, TSL, then Stacking or DCA
    if (g_trade_manager.HasCachedBasket(POSITION_TYPE_SELL))
    {
        CBasket sell_basket = g_trade_manager.GetCachedBasket(POSITION_TYPE_SELL);
        if(sell_basket.Ticket > 0)
        {
            g_trade_manager.ManagePartialTP(sell_basket);
            // Refresh basket after PTP (positions may have been partially closed)
            g_trade_manager.Refresh();
            sell_basket = g_trade_manager.GetCachedBasket(POSITION_TYPE_SELL);
            g_tsl_manager.ManageBasketTSL(POSITION_TYPE_SELL, sell_basket);
            if (g_trade_manager.IsStopLossProfitable(sell_basket))
            {
                g_stacking_manager.ManageStacking(POSITION_TYPE_SELL, sell_basket);
                g_trade_manager.Refresh(); // Refresh cache after stacking to include new position
                sell_basket = g_trade_manager.GetCachedBasket(POSITION_TYPE_SELL);
                g_tsl_manager.ManageBasketTSL(POSITION_TYPE_SELL, sell_basket); // update TSL for expanded basket
            }
            else if (!sell_basket.HasStacked)
            {
                g_dca_manager.ManageDCA(POSITION_TYPE_SELL, sell_basket);
                g_trade_manager.Refresh(); // Refresh cache after DCA to include new position
                sell_basket = g_trade_manager.GetCachedBasket(POSITION_TYPE_SELL);
                g_tsl_manager.ManageBasketTSL(POSITION_TYPE_SELL, sell_basket); // update TSL for expanded basket
            }
            g_trade_manager.ManageBasketTP(sell_basket); // Set basket TP if expanded
        }
    }

    // --- ENTRY LOGIC (runs on new bar only) ---
    if (!IsNewBar(CSettings::EaHeartbeatTimeframe)) return; // Throttle to heartbeat timeframe
    if (!g_money_manager.CheckDrawdown()) return; // Risk check: stop if drawdown too high
    if (!g_time_manager.IsTradeTimeAllowed()) return; // Time filter
    if (g_news_manager.IsNewsBlockActive()) return; // News filter
    if (SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) > CSettings::MaxSpreadPoints) return; // Spread filter
    int signal = g_signal_manager.GetFinalSignal(); // Aggregate signal from all sources

    // Check for BUY entry: signal, permissions, no existing basket
    if (signal == SIGNAL_BUY && CSettings::AllowLongTrades && !g_trade_manager.HasCachedBasket(POSITION_TYPE_BUY))
    {
        double lots = g_money_manager.GetInitialLotSize(); // Calculate lot size based on mode
        g_trade_manager.OpenTrade(signal, lots, CSettings::SlPips, CSettings::InitialTpPips, "INIT", 1);
    }

    // Check for SELL entry: signal, permissions, no existing basket
    if (signal == SIGNAL_SELL && CSettings::AllowShortTrades && !g_trade_manager.HasCachedBasket(POSITION_TYPE_SELL))
    {
        double lots = g_money_manager.GetInitialLotSize(); // Calculate lot size based on mode
        g_trade_manager.OpenTrade(signal, lots, CSettings::SlPips, CSettings::InitialTpPips, "INIT", 1);
    }
}
//+------------------------------------------------------------------+
