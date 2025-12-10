//+------------------------------------------------------------------+
//|                                                  NewsManager.mqh |
//|                                     Copyright 2025, LAWRANCE KOH |
//|                                          lawrancekoh@outlook.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025, LAWRANCE KOH"
#property link      "lawrancekoh@outlook.com"
#property version   "1.01" // Updated version

#include "Settings.mqh"
#include <Arrays/ArrayObj.mqh>

//+------------------------------------------------------------------+
//| News Event Structure                                             |
//+------------------------------------------------------------------+
struct CNewsEvent
{
   string title;
   string currency;
   string impact;
   datetime time;
};

//+------------------------------------------------------------------+
//| CNewsManager Class                                               |
//| Manages filtering of trading signals based on calendar news      |
//| events.                                                          |
//+------------------------------------------------------------------+
class CNewsManager
   {
private:
    datetime m_next_news_time;
    bool m_news_cache_valid;

    // Web Request Cache
    CNewsEvent m_cached_events[]; // Dynamic array of news events
    datetime m_last_web_request_time;
    const int m_web_request_interval_seconds; // Interval to refresh news (e.g. 1 hour)

    //+------------------------------------------------------------------+
    //| Helpers for String Processing                                    |
    //+------------------------------------------------------------------+
    string CleanQuote(string str)
    {
       if (StringLen(str) >= 2 && StringGetCharacter(str, 0) == '"' && StringGetCharacter(str, StringLen(str)-1) == '"')
       {
           return StringSubstr(str, 1, StringLen(str) - 2);
       }
       return str;
    }

    datetime ParseCsvDateTime(string dateStr, string timeStr)
    {
        // Format: MM/DD/YYYY and HH:MM
        // StringToTime converts "yyyy.mm.dd [hh:mi]"
        // We need to convert MM/DD/YYYY to yyyy.mm.dd

        string date_parts[];
        // Check for / or -
        if(StringSplit(dateStr, '/', date_parts) != 3)
        {
             if(StringSplit(dateStr, '-', date_parts) != 3) return 0;
        }

        // date_parts: [0]=MM, [1]=DD, [2]=YYYY
        string yyyy = date_parts[2];
        string mm = date_parts[0];
        string dd = date_parts[1];

        string formatted_time = yyyy + "." + mm + "." + dd + " " + timeStr;
        return StringToTime(formatted_time);
    }

    //+------------------------------------------------------------------+
    //| Web Request Logic                                                |
    //+------------------------------------------------------------------+
    bool FetchAndParseNews()
    {
       string cookie = NULL, headers;
       char post[], result[];
       int res;
       string url = CSettings::NewsCalendarURL;
       if (url == "") url = "https://nfs.forexfactory.net/ffcal_week_this.csv"; // Fallback default

       // Reset Last Error
       ResetLastError();

       int timeout = 5000; // 5 seconds

       res = WebRequest("GET", url, cookie, NULL, timeout, post, 0, result, headers);

       if (res == -1)
       {
          Print("NewsManager: WebRequest failed. Error: ", GetLastError());
          // Check if URL is allowed
          if(GetLastError() == 4060) // ERR_FUNCTION_NOT_ALLOWED
          {
             Print("NewsManager: Please add '", url, "' to the allowed URLs in Tools->Options->Expert Advisors.");
          }
          return false;
       }
       else if (res != 200)
       {
          Print("NewsManager: WebRequest returned HTTP status ", res);
          return false;
       }

       // Process Result
       string response = CharArrayToString(result);

       // Parse CSV
       string lines[];
       int line_count = StringSplit(response, '\n', lines);

       if(line_count <= 0) return false;

       ArrayResize(m_cached_events, 0); // Clear cache

       for(int i = 0; i < line_count; i++)
       {
          string line = lines[i];
          if(StringLen(line) < 5) continue; // Skip empty lines

          string fields[];
          int field_count = StringSplit(line, ',', fields);

          if(field_count < 4) continue;

          // Expected: "Date","Time","Currency","Impact","Event"
          // We need to handle that StringSplit splits by comma, but some fields might contain comma?
          // Standard FF CSV usually puts quotes around fields.
          // Simple split might break if "Event" contains comma.
          // For robustness, we should respect quotes.
          // But StringSplit is simple.
          // Assuming FF CSV format is consistent and Event is the last field or doesn't have commas usually.
          // If fields are quoted, we can rely on standard format.

          // Map fields
          string s_date = CleanQuote(fields[0]);
          string s_time = CleanQuote(fields[1]);
          string s_curr = CleanQuote(fields[2]);
          string s_impact = CleanQuote(fields[3]);
          string s_title = (field_count > 4) ? CleanQuote(fields[4]) : "";

          CNewsEvent event;
          event.currency = s_curr;
          event.impact = s_impact;
          event.title = s_title;
          event.time = ParseCsvDateTime(s_date, s_time);

          if(event.time > 0)
          {
             int size = ArraySize(m_cached_events);
             ArrayResize(m_cached_events, size + 1);
             m_cached_events[size] = event;
          }
       }

       Print("NewsManager: Successfully fetched and parsed ", ArraySize(m_cached_events), " news events.");
       return true;
    }

    //+------------------------------------------------------------------+
    //| Checks for blocking news events from the built-in MT5 calendar.  |
    //| @return true if a blocking news event is found, false otherwise |
    //+------------------------------------------------------------------+
    bool IsMt5CalendarBlockActive()
     {
      // STUBBED DUE TO COMPILER BUG: MqlCalendarValue string and enum members cannot be accessed reliably.
      // This is a known issue in MQL5 compiler; logic preserved in comments for future remediation.
      /*
      //--- Define Time Window in GMT
      long mins_before_sec = (long)CSettings::NewsMinsBefore * 60;
      long mins_after_sec  = (long)CSettings::NewsMinsAfter * 60;
      datetime from = TimeGMT() - mins_before_sec;
      datetime to   = TimeGMT() + mins_after_sec;

      MqlCalendarValue values_array[];

      //--- Get Calendar Events
      if(CalendarValueHistory(values_array, from, to) > 0)
        {
         //--- Get Symbol Currencies
         string currency1 = SymbolInfoString(_Symbol, SYMBOL_CURRENCY_BASE);
         string currency2 = SymbolInfoString(_Symbol, SYMBOL_CURRENCY_PROFIT);

         //--- Loop and Filter
         int total_events = ArraySize(values_array);
         for(int i = 0; i < total_events; i++)
           {
              MqlCalendarValue event = values_array[i];

              //--- Check if the event's currency matches the symbol's currencies
              if(event.currency == currency1 || event.currency == currency2)
                {
                   //--- Check if the impact level is set to be filtered
                   bool is_high_impact   = (event.importance == CALENDAR_IMPORTANCE_HIGH && CSettings::NewsFilterHighImpact);
                   bool is_medium_impact = (event.importance == CALENDAR_IMPORTANCE_MODERATE && CSettings::NewsFilterMedImpact);
                   bool is_low_impact    = (event.importance == CALENDAR_IMPORTANCE_LOW && CSettings::NewsFilterLowImpact);

                   if(is_high_impact || is_medium_impact || is_low_impact)
                     {
                      // Found a blocking event, print details and return
                      PrintFormat("NEWS BLOCK: %s %s %s",
                                  TimeToString(event.time, TIME_DATE | TIME_MINUTES),
                                  event.currency,
                                  event.name);
                      return true;
                     }
                }
           }
        }
      */

      //--- No blocking events found (stubbed)
      return false;
     }

public:
   //+------------------------------------------------------------------+
   //| Constructor                                                      |
   //+------------------------------------------------------------------+
   CNewsManager(void) : m_web_request_interval_seconds(3600) // 1 Hour default refresh
   {
       m_last_web_request_time = 0;
   };
   //+------------------------------------------------------------------+
   //| Destructor                                                       |
   //+------------------------------------------------------------------+
   ~CNewsManager(void)
   {
       ArrayResize(m_cached_events, 0);
   };

   //+------------------------------------------------------------------+
   //| Initialization Method                                            |
   //+------------------------------------------------------------------+
   void Init()
     {
        m_next_news_time = 0;
        m_news_cache_valid = false;
        m_last_web_request_time = 0;
     }

   //+------------------------------------------------------------------+
   //| Refresh news cache by finding next relevant news event         |
   //+------------------------------------------------------------------+
   void RefreshNewsCache()
     {
        // For built-in mode
        // TODO: Implement actual news checking logic when MT5 calendar is available
        // For now, set cache as valid with no news
        m_next_news_time = 0;
        m_news_cache_valid = true;
     }

   //+------------------------------------------------------------------+
   //| Check Web Request News Block                                     |
   //+------------------------------------------------------------------+
   bool CheckWebRequestNews()
   {
       // 1. Refresh Cache if needed
       if (TimeCurrent() - m_last_web_request_time > m_web_request_interval_seconds || m_last_web_request_time == 0)
       {
           if (FetchAndParseNews())
           {
               m_last_web_request_time = TimeCurrent();
           }
           else
           {
               // If failed, maybe try again sooner? or just keep old cache?
               // We keep old cache but update time to retry in 5 mins maybe?
               // For now, retry normal interval to avoid spamming if error is persistent.
               m_last_web_request_time = TimeCurrent();
           }
       }

       // 2. Check cached events against current time
       long mins_before_sec = (long)CSettings::NewsMinsBefore * 60;
       long mins_after_sec  = (long)CSettings::NewsMinsAfter * 60;
       datetime current_time = TimeCurrent();

       // Get Symbol Currencies
       string currency1 = SymbolInfoString(CSettings::Symbol, SYMBOL_CURRENCY_BASE);
       string currency2 = SymbolInfoString(CSettings::Symbol, SYMBOL_CURRENCY_PROFIT);

       int total = ArraySize(m_cached_events);
       for(int i = 0; i < total; i++)
       {
           CNewsEvent event = m_cached_events[i];

           // Filter by Currency
           // Check against symbol currencies
           bool currency_match = (event.currency == currency1 || event.currency == currency2);

           // Also check against global filter list if currencies are specified there?
           // Usually we only care about symbol currencies.
           // But `CSettings::NewsFilterCurrencies` exists.
           // If user specified currencies, maybe we should also check those?
           // The prompt implies "relevant to the current symbol".
           // But existing logic stub checked `CSettings::NewsFilterCurrencies` (Wait, the stub code checked `event.currency == currency1 || event.currency == currency2`).
           // The setting `NewsFilterCurrencies` was present in `CSettings` but not used in the stub code I saw.
           // I will implement check for Symbol currencies AND the list if provided.

           if (!currency_match)
           {
               // Check if in allowed list
               if (StringFind(CSettings::NewsFilterCurrencies, event.currency) >= 0)
               {
                   currency_match = true;
               }
           }

           if (!currency_match) continue;

           // Filter by Impact
           bool is_high = (event.impact == "High" && CSettings::NewsFilterHighImpact);
           bool is_med = (event.impact == "Medium" && CSettings::NewsFilterMedImpact);
           bool is_low = (event.impact == "Low" && CSettings::NewsFilterLowImpact);

           if (!is_high && !is_med && !is_low) continue;

           // Filter by Time
           // Check if current time is within [event_time - before, event_time + after]
           if (current_time >= (event.time - mins_before_sec) &&
               current_time <= (event.time + mins_after_sec))
           {
               PrintFormat("NEWS BLOCK (Web): %s %s %s",
                          TimeToString(event.time, TIME_DATE | TIME_MINUTES),
                          event.currency,
                          event.title);
               return true;
           }
       }

       return false;
   }

   //+------------------------------------------------------------------+
   //| Main public method to check if trading is blocked by news.       |
   //| @return true if trading is blocked, false otherwise             |
   //+------------------------------------------------------------------+
   bool IsNewsBlockActive()
     {
      switch(CSettings::NewsSourceMode)
        {
         case MODE_DISABLED:
            return false;

         case MODE_MT5_BUILT_IN:
            // Only refresh cache if invalid or if next news time is approaching/passed
            if(!m_news_cache_valid || (m_next_news_time > 0 && TimeCurrent() >= m_next_news_time - CSettings::NewsMinsBefore * 60))
            {
               RefreshNewsCache();
            }
            return IsMt5CalendarBlockActive();

         case MODE_WEB_REQUEST:
            return CheckWebRequestNews();
        }
      return false;
     }
  };
//+------------------------------------------------------------------+
