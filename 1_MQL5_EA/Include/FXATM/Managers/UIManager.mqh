//+------------------------------------------------------------------+
//|                                                    UIManager.mqh |
//|                                     Copyright 2025, LAWRANCE KOH |
//|                                          lawrancekoh@outlook.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, LAWRANCE KOH"
#property link      "lawrancekoh@outlook.com"
#property version   "1.00"

#include "Settings.mqh"

class CUIManager
  {
public:
   CUIManager(void) {};
   ~CUIManager(void) {};

   void Init()
     {
        // Nothing to do here for now
     }

   void Update()
     {
      if (!CSettings::ChartShowPanels)
        {
         return;
        }
      // UI update logic to be implemented here.
      // This will involve drawing objects on the chart.
     }
  };
//+------------------------------------------------------------------+