//+------------------------------------------------------------------+
//|                                                     Settings.mqh |
//|                            FXATM Configuration Management System |
//|                                     Copyright 2025, LAWRANCE KOH |
//|                                          lawrancekoh@outlook.com |
//+------------------------------------------------------------------+
//| PURPOSE:                                                         |
//|   Central configuration repository for FXATM Expert Advisor      |
//|   Manages all input parameters, enums, and static settings       |
//|                                                                  |
//| KEY COMPONENTS:                                                  |
//|   • 6 Lot sizing modes (Fixed, Risk%, ATR-Volatility Adjusted)   |
//|   • 5 Trailing Stop Loss modes (Step, ATR, MA, High/Low)         |
//|   • Signal configuration structures for polymorphic signals      |
//|   • Static settings class with global parameter access           |
//|                                                                  |
//| USAGE:                                                           |
//|   Include this file to access CSettings class and enumerations   |
//|   All EA parameters are centralized here for consistency         |
//+------------------------------------------------------------------+
#property link      "lawrancekoh@outlook.com"

#include <Object.mqh>

// Generic Trade Signals
enum ENUM_TRADE_SIGNAL
{
    SIGNAL_NONE,
    SIGNAL_BUY,
    SIGNAL_SELL,
    SIGNAL_CLOSE_BUY,
    SIGNAL_CLOSE_SELL
};

// B. LOT SIZING SETTINGS
enum ENUM_LOT_SIZING_MODE
{
    MODE_FIXED_LOT,
    MODE_LOTS_PER_THOUSAND_BALANCE,
    MODE_LOTS_PER_THOUSAND_EQUITY,
    MODE_RISK_PERCENT_BALANCE,
    MODE_RISK_PERCENT_EQUITY,
    MODE_VOLATILITY_ADJUSTED
};

// E. TRAILING STOP SETTINGS
enum ENUM_TSL_MODE
{
    MODE_TSL_NONE,
    MODE_TSL_STEP,
    MODE_TSL_ATR,
    MODE_TSL_MOVING_AVERAGE,
    MODE_TSL_HIGH_LOW_BAR
};

// H. NEWS FILTER SETTINGS
enum ENUM_NEWS_SOURCE
{
    MODE_DISABLED,
    MODE_MT5_BUILT_IN,
    MODE_WEB_REQUEST
};

// STACKING LOT MODE SETTINGS
enum ENUM_STACKING_LOT_MODE
{
    MODE_FIXED,
    MODE_LAST_TRADE,
    MODE_BASKET_TOTAL,
    MODE_ENTRY_BASED
};

// I. SIGNAL SETTINGS
enum ENUM_SIGNAL_TYPE
{
    SIGNAL_TYPE_NONE,
    SIGNAL_RSI,
    SIGNAL_MACD,
    SIGNAL_MA_CROSS,
    SIGNAL_STOCHASTIC,
    SIGNAL_BOLLINGER_BANDS
};
enum ENUM_SIGNAL_ROLE
{
    ROLE_BIAS,
    ROLE_ENTRY
};
struct CSignalParams
{
    int    IntParams[4];
    double DoubleParams[4];
    bool   BoolParams[4];
    ENUM_APPLIED_PRICE Price;
    ENUM_MA_METHOD     MaMethod1;
    ENUM_MA_METHOD     MaMethod2;
    ENUM_STO_PRICE     PriceField;
};
struct CSignalSettings
{
    ENUM_SIGNAL_TYPE   Type;
    ENUM_SIGNAL_ROLE   Role;
    ENUM_TIMEFRAMES    Timeframe;
    CSignalParams Params;
};

//+------------------------------------------------------------------+
//| CSettings class                                                  |
//| A static-like class to hold all EA settings.                     |
//+------------------------------------------------------------------+
class CSettings
{
public:
    // A. GENERAL SETTINGS
    static string   EaName;
    static long     EaMagicNumber;
    static int      MaxSpreadPoints;
    static int      MaxSlippagePoints;
    static double   MaxDrawdownPercent;
    static ENUM_TIMEFRAMES EaHeartbeatTimeframe;
    static bool     AllowLongTrades;
    static bool     AllowShortTrades;
    static string   Symbol;

    // B. POSITION MANAGEMENT SETTINGS (Lot sizing + Basket TP/SL)
    static ENUM_LOT_SIZING_MODE LotSizingMode;
    static double   LotFixed;
    static double   LotsPerThousand;
    static double   LotRiskPercent;
    static int      SlPips;
    static int      InitialTpPips;
    static int      BasketTpPips;

    // C. LOSS MANAGEMENT SETTINGS (DCA)
    static int      DcaMaxTrades;
    static int      DcaTriggerPips;
    static double   DcaStepMultiplier;
    static double   DcaLotMultiplier;
    static int      DcaLotMultiplierStart;

    // D. PROFIT MANAGEMENT SETTINGS (TSL + Stacking)
    static ENUM_TSL_MODE TslMode;
    static int      TslBeTriggerPips;
    static int      BeOffsetPips;
    static int      TslStepPips;
    static bool     TslRemoveTp;
    static bool     BreakevenIncludesCosts;
    static double   CommissionPerLot;
    static int      TslAtrPeriod;
    static double   TslAtrMultiplier;
    static int      TslMaPeriod;
    static ENUM_MA_METHOD TslMaMethod;
    static ENUM_APPLIED_PRICE TslMaPrice;
    static int      TslHiLoPeriod;
    static int      StackingMaxTrades;
    static int      StackingTriggerPips;
    static double   StackingLotSize;
    static ENUM_STACKING_LOT_MODE StackingLotMode;

    // E. ADVANCED EXIT SETTINGS (Partial TP)
    static int      PartialTpTriggerPips;
    static double   PartialTpClosePercent;
    static bool     PartialTpSetBe;

    // F. FILTER SETTINGS (Time + News)
    static string   EaTradingDays;
    static string   EaTradingTimeStart;
    static string   EaTradingTimeEnd;
    static ENUM_NEWS_SOURCE NewsSourceMode;
    static string   NewsCalendarURL;
    static int      NewsMinsBefore;
    static int      NewsMinsAfter;
    static bool     NewsFilterHighImpact;
    static bool     NewsFilterMedImpact;
    static bool     NewsFilterLowImpact;
    static string   NewsFilterCurrencies;

    // G. SIGNAL SETTINGS
    static CSignalSettings Signal1;
    static CSignalSettings Signal2;
    static CSignalSettings Signal3;

    // H. SIGNAL MANAGER SETTINGS
    static int BiasPersistenceBars;

    // I. CHART UI SETTINGS
    static bool     ChartShowPanels;
    static ENUM_BASE_CORNER ChartPanelCorner;
    static color    ChartColorBackground;
    static color    ChartColorTextMain;
    static color    ChartColorBuy;
    static color    ChartColorSell;
    static color    ChartColorNeutral;

    // J. BACKTEST MODE DETECTION
    static bool     IsBacktestMode;
};

//+------------------------------------------------------------------+
//| Static member initialization                                     |
//+------------------------------------------------------------------+
string   CSettings::EaName;
long     CSettings::EaMagicNumber;
int      CSettings::MaxSpreadPoints = 0;
int      CSettings::MaxSlippagePoints = 0;
double   CSettings::MaxDrawdownPercent;
ENUM_TIMEFRAMES CSettings::EaHeartbeatTimeframe;
bool     CSettings::AllowLongTrades;
bool     CSettings::AllowShortTrades;
string   CSettings::Symbol;

ENUM_LOT_SIZING_MODE CSettings::LotSizingMode;
double   CSettings::LotFixed;
double   CSettings::LotsPerThousand;
double   CSettings::LotRiskPercent;
int      CSettings::SlPips = 0;
int      CSettings::InitialTpPips = 0;
int      CSettings::BasketTpPips = 0;

int      CSettings::DcaMaxTrades = 0;
int      CSettings::DcaTriggerPips = 0;
double   CSettings::DcaStepMultiplier;
double   CSettings::DcaLotMultiplier;
int      CSettings::DcaLotMultiplierStart = 0;

ENUM_TSL_MODE CSettings::TslMode;
int      CSettings::TslBeTriggerPips = 0;
int      CSettings::BeOffsetPips = 0;
int      CSettings::TslStepPips = 0;
bool     CSettings::TslRemoveTp;
bool     CSettings::BreakevenIncludesCosts;
double   CSettings::CommissionPerLot;
int      CSettings::TslAtrPeriod = 0;
double   CSettings::TslAtrMultiplier;
int      CSettings::TslMaPeriod = 0;
ENUM_MA_METHOD CSettings::TslMaMethod;
ENUM_APPLIED_PRICE CSettings::TslMaPrice;
int      CSettings::TslHiLoPeriod = 0;
int      CSettings::StackingMaxTrades = 0;
int      CSettings::StackingTriggerPips = 0;
double   CSettings::StackingLotSize;
ENUM_STACKING_LOT_MODE CSettings::StackingLotMode;

int      CSettings::PartialTpTriggerPips = 0;
double   CSettings::PartialTpClosePercent;
bool     CSettings::PartialTpSetBe;

string   CSettings::EaTradingDays;
string   CSettings::EaTradingTimeStart;
string   CSettings::EaTradingTimeEnd;
ENUM_NEWS_SOURCE CSettings::NewsSourceMode;
string   CSettings::NewsCalendarURL;
int      CSettings::NewsMinsBefore = 0;
int      CSettings::NewsMinsAfter = 0;
bool     CSettings::NewsFilterHighImpact;
bool     CSettings::NewsFilterMedImpact;
bool     CSettings::NewsFilterLowImpact;
string   CSettings::NewsFilterCurrencies;

CSignalSettings CSettings::Signal1;
CSignalSettings CSettings::Signal2;
CSignalSettings CSettings::Signal3;

int      CSettings::BiasPersistenceBars = 0;

bool     CSettings::ChartShowPanels;
ENUM_BASE_CORNER CSettings::ChartPanelCorner;
color    CSettings::ChartColorBackground;
color    CSettings::ChartColorTextMain;
color    CSettings::ChartColorBuy;
color    CSettings::ChartColorSell;
color    CSettings::ChartColorNeutral;

bool     CSettings::IsBacktestMode;
//+------------------------------------------------------------------+
