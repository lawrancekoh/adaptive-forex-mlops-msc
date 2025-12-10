//+------------------------------------------------------------------+
//|                                                  TimeManager.mqh |
//|                                     Copyright 2025, LAWRANCE KOH |
//|                                          lawrancekoh@outlook.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, LAWRANCE KOH"
#property link      "lawrancekoh@outlook.com"
#property version   "1.00"

#include "Settings.mqh"

class CTimeManager
  {
private:
    int m_start_mins;
    int m_end_mins;
    bool m_allowed_days[7]; // 0=Sunday, 1=Monday, ..., 6=Saturday

public:
    CTimeManager(void) {};
    ~CTimeManager(void) {};

    void Init()
      {
         // Pre-calculate start/end times in minutes from midnight
         string start_time = CSettings::EaTradingTimeStart;
         m_start_mins = (int)StringToInteger(StringSubstr(start_time, 0, 2)) * 60 +
                        (int)StringToInteger(StringSubstr(start_time, 3, 2));

         string end_time = CSettings::EaTradingTimeEnd;
         m_end_mins = (int)StringToInteger(StringSubstr(end_time, 0, 2)) * 60 +
                      (int)StringToInteger(StringSubstr(end_time, 3, 2));

         // Pre-calculate allowed days as boolean array
         string days_str = "," + CSettings::EaTradingDays + ",";
         StringReplace(days_str, " ", ""); // Handle spaces in input
         for(int i = 0; i < 7; i++)
         {
            string day_check = "," + IntegerToString(i) + ",";
            m_allowed_days[i] = (StringFind(days_str, day_check) != -1);
         }
      }

   bool IsTradeTimeAllowed()
     {
      //--- Initial Check
      if(CSettings::EaTradingDays == "")
         return true;

      //--- Get Current Time
      MqlDateTime current_time;
      TimeCurrent(current_time);

      //--- Day of Week Check using pre-calculated boolean array
      if(!m_allowed_days[current_time.day_of_week])
         return false;

      //--- Time of Day Check using pre-calculated minutes
      long current_mins = current_time.hour * 60 + current_time.min;

      //--- Handle Overnight Sessions (e.g., Start 22:00, End 06:00)
      if(m_start_mins > m_end_mins)
        {
         return (current_mins >= m_start_mins || current_mins <= m_end_mins);
        }
      //--- Handle Normal Day Sessions
      else
        {
         return (current_mins >= m_start_mins && current_mins <= m_end_mins);
        }
     }
  };
//+------------------------------------------------------------------+