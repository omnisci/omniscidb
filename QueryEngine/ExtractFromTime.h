/*
 * Copyright 2017 MapD Technologies, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef QUERYENGINE_EXTRACTFROMTIME_H
#define QUERYENGINE_EXTRACTFROMTIME_H

#include <cstdint>
#include <ctime>
#include "Shared/funcannotations.h"

static constexpr int64_t kNanoSecsPerSec = 1000000000;
static constexpr int64_t kMicroSecsPerSec = 1000000;
static constexpr int64_t kMilliSecsPerSec = 1000;
static constexpr int64_t kSecsPerMin = 60;
static constexpr int64_t kMinsPerHour = 60;
static constexpr int64_t kHoursPerDay = 24;
static constexpr int64_t kSecsPerHour = 3600;
static constexpr int64_t kSecsPerDay = 86400;
static constexpr int64_t kSecsPerQuarterDay = 21600;
static constexpr int32_t kDaysPerWeek = 7;
static constexpr int32_t kMonsPerYear = 12;
static constexpr int64_t kSecsPerHalfDay = 43200;
static constexpr int64_t kMinsPerMonth = 43200;  // Month of 30 days

static constexpr int32_t kYearBase = 1900;

/* move epoch from 01.01.1970 to 01.03.2000 - this is the first day of new
 * 400-year long cycle, right after additional day of leap year. This adjustment
 * is required only for date calculation, so instead of modifying time value
 * (which would require 64-bit operations to work correctly) it's enough to
 * adjust the calculated number of days since epoch. */
static constexpr int32_t kEpochAdjustedDays = 11017;
/* year to which the adjustment was made */
static constexpr int32_t kEpochAdjustedYears = 2000;
/* 1st March of 2000 is Wednesday */
static constexpr int32_t kEpochAdjustedWDay = 3;
/* there are 97 leap years in 400-year periods. ((400 - 97) * 365 + 97 * 366) */
static constexpr int64_t kDaysPer400Years = 146097;
/* there are 24 leap years in 100-year periods. ((100 - 24) * 365 + 24 * 366) */
static constexpr int64_t kDaysPer100Years = 36524;
/* there is one leap year every 4 years */
static constexpr int32_t kDaysPer4Years = 3 * 365 + 366;
/* number of days in a non-leap year */
static constexpr int32_t kDaysPerYear = 365;
/* number of days in January */
static constexpr int32_t kDaysInJanuary = 31;
/* number of days in non-leap February */
static constexpr int32_t kDaysInFebruary = 28;

static constexpr uint32_t kSecondsPerNonLeapYear = 31536000;
static constexpr uint32_t kSecondsPer4YearCycle = 126230400;
static constexpr uint32_t kUSecsPerDay = 86400;
static constexpr uint32_t kEpochOffsetYear1900 = 2208988800;
static constexpr uint32_t kSecsJanToMar1900 = 5097600;

// Number of days from March 1 to Jan 1.
constexpr unsigned MARJAN = 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30 + 31;
constexpr unsigned JANMAR = 31 + 28;  // leap day handled separately

enum ExtractField {
  kYEAR,
  kQUARTER,
  kMONTH,
  kDAY,
  kHOUR,
  kMINUTE,
  kSECOND,
  kMILLISECOND,
  kMICROSECOND,
  kNANOSECOND,
  kDOW,
  kISODOW,
  kDOY,
  kEPOCH,
  kQUARTERDAY,
  kWEEK,
  kWEEK_SUNDAY,
  kWEEK_SATURDAY,
  kDATEEPOCH,
  kLDOM,
  kISLEAP,
  kISEOM,
  kISSOM,
  kISEOQ,
  kISSOQ,
  kISEOY,
  kISSOY
};

DEVICE int64_t ExtractFromTime(ExtractField field, const int64_t timeval);

// Return floor(dividend / divisor).
// Assumes 0 < divisor.
DEVICE inline int64_t floor_div(int64_t const dividend, int64_t const divisor) {
  return (dividend < 0 ? dividend - (divisor - 1) : dividend) / divisor;
}

// Return remainer r of dividend / divisor, where 0 <= r < divisor.
// Assumes 0 < divisor.
DEVICE inline int64_t unsigned_mod(int64_t const dividend, int64_t const divisor) {
  int64_t mod = dividend % divisor;
  if (mod < 0) {
    mod += divisor;
  }
  return mod;
}

#endif  // QUERYENGINE_EXTRACTFROMTIME_H
