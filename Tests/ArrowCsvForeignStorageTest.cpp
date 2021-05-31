/*
 * Copyright 2020 OmniSci, Inc.
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

#include "TestHelpers.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <limits>
#include <string>

#include "Archive/PosixFileArchive.h"
#include "Catalog/Catalog.h"
#include "DataMgr/ForeignStorage/ArrowForeignStorage.h"
#include "Geospatial/Types.h"
#include "ImportExport/Importer.h"
#include "Parser/parser.h"
#include "QueryEngine/ResultSet.h"
#include "QueryRunner/QueryRunner.h"
#include "Shared/scope.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

using namespace std;
using namespace TestHelpers;
using QR = QueryRunner::QueryRunner;

extern bool g_use_date_in_days_default_encoding;

namespace {

inline void run_ddl_statement(const string& input_str) {
  QR::get()->runDDLStatement(input_str);
}

TargetValue run_simple_agg(const string& query_str) {
  auto rows = QR::get()->runSQL(query_str, ExecutorDeviceType::CPU, false);
  auto crt_row = rows->getNextRow(true, true);
  CHECK_EQ(size_t(1), crt_row.size()) << query_str;
  return crt_row[0];
}

template <typename T>
void check_query(const std::string& query, const std::vector<T>& expects) {
  auto rows = QR::get()->runSQL(query, ExecutorDeviceType::CPU, false);
  CHECK_EQ(expects.size(), rows->rowCount());
  for (auto exp : expects) {
    auto crt_row = rows->getNextRow(true, true);
    CHECK_EQ(size_t(1), crt_row.size());
    CHECK_EQ(exp, v<T>(crt_row[0]));
  }
}

#define FP_CHECK_EQ(val1, val2, type)               \
  if (typeid(type) == typeid(double)) {             \
    EXPECT_DOUBLE_EQ(v<type>(val1), v<type>(val1)); \
  } else if (typeid(T) == typeid(float)) {          \
    EXPECT_FLOAT_EQ(v<type>(val1), v<type>(val1));  \
  } else {                                          \
    CHECK_EQ(v<type>(val1), v<type>(val1));         \
  }

template <typename T>
void check_table(const std::string& query, const std::vector<std::vector<T>>& expects) {
  auto rows = QR::get()->runSQL(query, ExecutorDeviceType::CPU, false);
  CHECK_EQ(expects.size(), rows->rowCount());
  CHECK_EQ(expects[0].size(), rows->colCount());
  for (auto exp_row : expects) {
    auto crt_row = rows->getNextRow(true, true);
    for (size_t val_idx = 0; val_idx < exp_row.size(); ++val_idx) {
      FP_CHECK_EQ(exp_row[val_idx], crt_row[val_idx], T);
    }
  }
}

template <typename T>
void check_tables(const std::string& query, const std::string& query_expects) {
  auto rows = QR::get()->runSQL(query, ExecutorDeviceType::CPU, false);
  auto rows_expects = QR::get()->runSQL(query_expects, ExecutorDeviceType::CPU, false);
  CHECK_EQ(rows_expects->rowCount(), rows->rowCount());
  CHECK_EQ(rows_expects->colCount(), rows->colCount());
  for (size_t row_idx = 0; row_idx < rows_expects->rowCount(); ++row_idx) {
    auto row = rows->getNextRow(true, true);
    auto row_expect = rows_expects->getNextRow(true, true);
    for (size_t col_idx = 0; col_idx < rows_expects->colCount(); ++col_idx) {
      FP_CHECK_EQ(row_expect[col_idx], row[col_idx], T);
    }
  }
}

const char* trips_table_ddl = R"(
CREATE TEMPORARY TABLE trips (
trip_id BIGINT,
vendor_id TEXT ENCODING NONE,
pickup_datetime TIMESTAMP,
dropoff_datetime TIMESTAMP,
store_and_fwd_flag TEXT ENCODING DICT,
rate_code_id BIGINT,
pickup_longitude DOUBLE,
pickup_latitude DOUBLE,
dropoff_longitude DOUBLE,
dropoff_latitude DOUBLE,
passenger_count BIGINT,
trip_distance DOUBLE,
fare_amount DOUBLE,
extra DOUBLE,
mta_tax DOUBLE,
tip_amount DOUBLE,
tolls_amount DOUBLE,
ehail_fee DOUBLE,
improvement_surcharge DOUBLE,
total_amount DOUBLE,
payment_type TEXT ENCODING DICT,
trip_type BIGINT,
pickup TEXT ENCODING DICT,
dropoff TEXT ENCODING NONE,
cab_type TEXT ENCODING DICT,
precipitation DOUBLE,
snow_depth BIGINT,
snowfall DOUBLE,
max_temperature BIGINT,
min_temperature BIGINT,
average_wind_speed DOUBLE,
pickup_nyct2010_gid BIGINT,
pickup_ctlabel DOUBLE,
pickup_borocode BIGINT,
pickup_boroname TEXT ENCODING NONE,
pickup_ct2010 BIGINT,
pickup_boroct2010 BIGINT,
pickup_cdeligibil TEXT ENCODING DICT,
pickup_ntacode TEXT ENCODING DICT,
pickup_ntaname TEXT ENCODING DICT,
pickup_puma BIGINT,
dropoff_nyct2010_gid BIGINT,
dropoff_ctlabel DOUBLE,
dropoff_borocode BIGINT,
dropoff_boroname TEXT ENCODING NONE,
dropoff_ct2010 BIGINT,
dropoff_boroct2010 BIGINT,
dropoff_cdeligibil TEXT ENCODING NONE,
dropoff_ntacode TEXT ENCODING NONE,
dropoff_ntaname TEXT ENCODING NONE,
dropoff_puma BIGINT) WITH (storage_type='CSV:../../Tests/Import/datafiles/trips_with_headers_top1000.csv', fragment_size=100);
)";

class NycTaxiTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists trips;"););
    ASSERT_NO_THROW(run_ddl_statement(trips_table_ddl));
  }

  void TearDown() override {
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists trips;"););
  }
};

TEST_F(NycTaxiTest, RunSimpleQuery) {
  // TODO: expect +1 rows when move to arrow 0.15 as current arrow doesn't support
  // headerless csv
  ASSERT_EQ(999,
            v<int64_t>(run_simple_agg(
                "SELECT count(vendor_id) FROM trips where vendor_id < '5'")));
}

TEST_F(NycTaxiTest, GroupByColumnWithNulls) {
  // TODO: expect +1 rows when move to arrow 0.15 as current arrow doesn't support
  // headerless csv
  ASSERT_EQ(
      619,
      v<int64_t>(run_simple_agg(
          " select count(*) from (select pickup, count(*) from trips group by pickup)")));
}

TEST_F(NycTaxiTest, CheckGroupBy) {
  check_query<NullableString>(
      "select pickup_ntaname from trips where pickup_ntaname IS NOT NULL group by "
      "pickup_ntaname order by pickup_ntaname limit 5;",
      {"Astoria",
       "Bedford Park-Fordham North",
       "Belmont",
       "Briarwood-Jamaica Hills",
       "Central Harlem North-Polo Grounds"});

  check_query<double>(
      "select tip_amount from trips group by tip_amount order by tip_amount limit 5;",
      {0, 0.01, 0.02, 0.03, 0.05});

  check_query<NullableString>(
      "select store_and_fwd_flag from trips group by store_and_fwd_flag order by "
      "store_and_fwd_flag limit 5;",
      {"N", "Y"});
}

TEST_F(NycTaxiTest, RunSelects) {
  check_query<int64_t>(
      "select rate_code_id from trips group by rate_code_id order by rate_code_id limit "
      "5;",
      {1, 2, 3, 4, 5});
}

TEST_F(NycTaxiTest, RunSelectsEncodingNoneNotNull) {
  check_query<NullableString>(
      "select dropoff_ntaname from trips where dropoff_ntaname is not NULL order by "
      "dropoff_ntaname limit 50;",
      {"Airport",
       "Airport",
       "Airport",
       "Airport",
       "Airport",
       "Airport",
       "Airport",
       "Airport",
       "Airport",
       "Airport",
       "Allerton-Pelham Gardens",
       "Astoria",
       "Astoria",
       "Astoria",
       "Astoria",
       "Astoria",
       "Astoria",
       "Astoria",
       "Astoria",
       "Astoria",
       "Astoria",
       "Astoria",
       "Astoria",
       "Astoria",
       "Astoria",
       "Astoria",
       "Battery Park City-Lower Manhattan",
       "Battery Park City-Lower Manhattan",
       "Battery Park City-Lower Manhattan",
       "Battery Park City-Lower Manhattan",
       "Battery Park City-Lower Manhattan",
       "Bedford Park-Fordham North",
       "Bedford Park-Fordham North",
       "Bellerose",
       "Belmont",
       "Belmont",
       "Belmont",
       "Belmont",
       "Belmont",
       "Borough Park",
       "Briarwood-Jamaica Hills",
       "Brooklyn Heights-Cobble Hill",
       "Central Harlem North-Polo Grounds",
       "Central Harlem North-Polo Grounds",
       "Central Harlem North-Polo Grounds",
       "Central Harlem North-Polo Grounds",
       "Central Harlem North-Polo Grounds",
       "Central Harlem North-Polo Grounds",
       "Central Harlem North-Polo Grounds",
       "Central Harlem North-Polo Grounds"});
}

TEST_F(NycTaxiTest, RunSelectsEncodingNoneWhereGreater) {
  check_query<NullableString>(
      "select dropoff_ntaname from trips where dropoff_ntaname > "
      "'Queensbridge-Ravenswood-Long Island City' order by dropoff_ntaname limit 10;",
      {"Richmond Hill",
       "Richmond Hill",
       "Richmond Hill",
       "Richmond Hill",
       "Richmond Hill",
       "Richmond Hill",
       "Rosedale",
       "SoHo-TriBeCa-Civic Center-Little Italy",
       "SoHo-TriBeCa-Civic Center-Little Italy",
       "Soundview-Bruckner"});
}

TEST_F(NycTaxiTest, RunSelectsEncodingDictWhereGreater) {
  check_query<NullableString>(
      "select pickup_ntaname from trips where pickup_ntaname is not NULL and "
      "pickup_ntaname > 'Queensbridge-Ravenswood-Long Island City' order by "
      "pickup_ntaname limit 3;",
      {"Rego Park", "Richmond Hill", "Richmond Hill"});
}

TEST(Unsupported, Syntax) {
  run_ddl_statement("DROP TABLE IF EXISTS fsi_unsupported;");
  EXPECT_ANY_THROW(
      run_ddl_statement("CREATE TABLE fsi_unsupported (x INT, y DOUBLE) WITH "
                        "(storage_type='CSV:../../Tests/Import/datafiles/"
                        "trips_with_headers_top1000.csv');"));
}

const std::string decimal_table_format =
    "(ID_code TEXT ENCODING NONE,target SMALLINT,var_0 "
    "DECIMAL(8, 4),var_1 DECIMAL(8, 4),var_2 DECIMAL(8, 4),var_3 DECIMAL(8, 4),var_4 "
    "DECIMAL(8, 4),var_5 DECIMAL(8, 4),var_6 DECIMAL(8, 4),var_7 DECIMAL(8, 4),var_8 "
    "DECIMAL(8, 4),var_9 DECIMAL(8, 4),var_10 DECIMAL(8, 4),var_11 DECIMAL(8, "
    "4),var_12 DECIMAL(8, 4),var_13 DECIMAL(8, 4),var_14 DECIMAL(8, 4),var_15 "
    "DECIMAL(8, 4),var_16 DECIMAL(8, 4),var_17 DECIMAL(8, 4),var_18 DECIMAL(8, "
    "4),var_19 DECIMAL(8, 4),var_20 DECIMAL(8, 4),var_21 DECIMAL(8, 4),var_22 "
    "DECIMAL(8, 4),var_23 DECIMAL(8, 4),var_24 DECIMAL(8, 4),var_25 DECIMAL(8, "
    "4),var_26 DECIMAL(8, 4),var_27 DECIMAL(8, 4),var_28 DECIMAL(8, 4),var_29 "
    "DECIMAL(8, 4),var_30 DECIMAL(8, 4),var_31 DECIMAL(8, 4),var_32 DECIMAL(8, "
    "4),var_33 DECIMAL(8, 4),var_34 DECIMAL(8, 4),var_35 DECIMAL(8, 4),var_36 "
    "DECIMAL(8, 4),var_37 DECIMAL(8, 4),var_38 DECIMAL(8, 4),var_39 DECIMAL(8, "
    "4),var_40 DECIMAL(8, 4),var_41 DECIMAL(8, 4),var_42 DECIMAL(8, 4),var_43 "
    "DECIMAL(8, 4),var_44 DECIMAL(8, 4),var_45 DECIMAL(8, 4),var_46 DECIMAL(8, "
    "4),var_47 DECIMAL(8, 4),var_48 DECIMAL(8, 4),var_49 DECIMAL(8, 4),var_50 "
    "DECIMAL(8, 4),var_51 DECIMAL(8, 4),var_52 DECIMAL(8, 4),var_53 DECIMAL(8, "
    "4),var_54 DECIMAL(8, 4),var_55 DECIMAL(8, 4),var_56 DECIMAL(8, 4),var_57 "
    "DECIMAL(8, 4),var_58 DECIMAL(8, 4),var_59 DECIMAL(8, 4),var_60 DECIMAL(8, "
    "4),var_61 DECIMAL(8, 4),var_62 DECIMAL(8, 4),var_63 DECIMAL(8, 4),var_64 "
    "DECIMAL(8, 4),var_65 DECIMAL(8, 4),var_66 DECIMAL(8, 4),var_67 DECIMAL(8, "
    "4),var_68 DECIMAL(8, 4),var_69 DECIMAL(8, 4),var_70 DECIMAL(8, 4),var_71 "
    "DECIMAL(8, 4),var_72 DECIMAL(8, 4),var_73 DECIMAL(8, 4),var_74 DECIMAL(8, "
    "4),var_75 DECIMAL(8, 4),var_76 DECIMAL(8, 4),var_77 DECIMAL(8, 4),var_78 "
    "DECIMAL(8, 4),var_79 DECIMAL(8, 4),var_80 DECIMAL(8, 4),var_81 DECIMAL(8, "
    "4),var_82 DECIMAL(8, 4),var_83 DECIMAL(8, 4),var_84 DECIMAL(8, 4),var_85 "
    "DECIMAL(8, 4),var_86 DECIMAL(8, 4),var_87 DECIMAL(8, 4),var_88 DECIMAL(8, "
    "4),var_89 DECIMAL(8, 4),var_90 DECIMAL(8, 4),var_91 DECIMAL(8, 4),var_92 "
    "DECIMAL(8, 4),var_93 DECIMAL(8, 4),var_94 DECIMAL(8, 4),var_95 DECIMAL(8, "
    "4),var_96 DECIMAL(8, 4),var_97 DECIMAL(8, 4),var_98 DECIMAL(8, 4),var_99 "
    "DECIMAL(8, 4),var_100 DECIMAL(8, 4),var_101 DECIMAL(8, 4),var_102 DECIMAL(8, "
    "4),var_103 DECIMAL(8, 4),var_104 DECIMAL(8, 4),var_105 DECIMAL(8, 4),var_106 "
    "DECIMAL(8, 4),var_107 DECIMAL(8, 4),var_108 DECIMAL(8, 4),var_109 DECIMAL(8, "
    "4),var_110 DECIMAL(8, 4),var_111 DECIMAL(8, 4),var_112 DECIMAL(8, 4),var_113 "
    "DECIMAL(8, 4),var_114 DECIMAL(8, 4),var_115 DECIMAL(8, 4),var_116 DECIMAL(8, "
    "4),var_117 DECIMAL(8, 4),var_118 DECIMAL(8, 4),var_119 DECIMAL(8, 4),var_120 "
    "DECIMAL(8, 4),var_121 DECIMAL(8, 4),var_122 DECIMAL(8, 4),var_123 DECIMAL(8, "
    "4),var_124 DECIMAL(8, 4),var_125 DECIMAL(8, 4),var_126 DECIMAL(8, 4),var_127 "
    "DECIMAL(8, 4),var_128 DECIMAL(8, 4),var_129 DECIMAL(8, 4),var_130 DECIMAL(8, "
    "4),var_131 DECIMAL(8, 4),var_132 DECIMAL(8, 4),var_133 DECIMAL(8, 4),var_134 "
    "DECIMAL(8, 4),var_135 DECIMAL(8, 4),var_136 DECIMAL(8, 4),var_137 DECIMAL(8, "
    "4),var_138 DECIMAL(8, 4),var_139 DECIMAL(8, 4),var_140 DECIMAL(8, 4),var_141 "
    "DECIMAL(8, 4),var_142 DECIMAL(8, 4),var_143 DECIMAL(8, 4),var_144 DECIMAL(8, "
    "4),var_145 DECIMAL(8, 4),var_146 DECIMAL(8, 4),var_147 DECIMAL(8, 4),var_148 "
    "DECIMAL(8, 4),var_149 DECIMAL(8, 4),var_150 DECIMAL(8, 4),var_151 DECIMAL(8, "
    "4),var_152 DECIMAL(8, 4),var_153 DECIMAL(8, 4),var_154 DECIMAL(8, 4),var_155 "
    "DECIMAL(8, 4),var_156 DECIMAL(8, 4),var_157 DECIMAL(8, 4),var_158 DECIMAL(8, "
    "4),var_159 DECIMAL(8, 4),var_160 DECIMAL(8, 4),var_161 DECIMAL(8, 4),var_162 "
    "DECIMAL(8, 4),var_163 DECIMAL(8, 4),var_164 DECIMAL(8, 4),var_165 DECIMAL(8, "
    "4),var_166 DECIMAL(8, 4),var_167 DECIMAL(8, 4),var_168 DECIMAL(8, 4),var_169 "
    "DECIMAL(8, 4),var_170 DECIMAL(8, 4),var_171 DECIMAL(8, 4),var_172 DECIMAL(8, "
    "4),var_173 DECIMAL(8, 4),var_174 DECIMAL(8, 4),var_175 DECIMAL(8, 4),var_176 "
    "DECIMAL(8, 4),var_177 DECIMAL(8, 4),var_178 DECIMAL(8, 4),var_179 DECIMAL(8, "
    "4),var_180 DECIMAL(8, 4),var_181 DECIMAL(8, 4),var_182 DECIMAL(8, 4),var_183 "
    "DECIMAL(8, 4),var_184 DECIMAL(8, 4),var_185 DECIMAL(8, 4),var_186 DECIMAL(8, "
    "4),var_187 DECIMAL(8, 4),var_188 DECIMAL(8, 4),var_189 DECIMAL(8, 4),var_190 "
    "DECIMAL(8, 4),var_191 DECIMAL(8, 4),var_192 DECIMAL(8, 4),var_193 DECIMAL(8, "
    "4),var_194 DECIMAL(8, 4),var_195 DECIMAL(8, 4),var_196 DECIMAL(8, 4),var_197 "
    "DECIMAL(8, 4),var_198 DECIMAL(8, 4),var_199 DECIMAL(8, 4))";

class DecimalTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists decimal_table;"));
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists decimal_dataframe;"));
    ASSERT_NO_THROW(
        run_ddl_statement("CREATE TABLE decimal_table" + decimal_table_format + ";"));
    ASSERT_NO_THROW(run_ddl_statement("TRUNCATE TABLE decimal_table;"));
    ASSERT_NO_THROW(run_ddl_statement(
        "COPY decimal_table FROM '../../Tests/Import/datafiles/santander_top1000.csv';"));

    ASSERT_NO_THROW(run_ddl_statement(
        "CREATE DATAFRAME decimal_dataframe" + decimal_table_format +
        " FROM 'CSV:../../Tests/Import/datafiles/santander_top1000.csv'"));
  }

  void TearDown() override {
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists decimal_table;"));
    ASSERT_NO_THROW(run_ddl_statement("drop table if exists decimal_dataframe;"));
  }
};

TEST_F(DecimalTest, DifferentSizesOfDecimal) {
  run_ddl_statement(
      "CREATE DATAFRAME fsi_decimal (decimal2 DECIMAL(4,1), decimal4 NUMERIC(9,2), "
      "decimal8 DECIMAL(18,5)) from "
      "'CSV:../../Tests/Import/datafiles/decimal_data.csv' WITH(fragment_size=1);");
  check_table<double>(
      "SELECT decimal2, decimal4, decimal8 FROM fsi_decimal order by decimal2",
      {{4, 0, 1.1},
       {213.4, 2389341.23, 4857364039384.75638},
       {999.9, 9384612.78, 2947583746581.92748}});
}

TEST_F(DecimalTest, GoupByDecimal) {
  check_tables<double>(
      "SELECT var_101 FROM decimal_dataframe WHERE var_190>=0 GROUP BY var_101 ORDER BY "
      "var_101;",
      "SELECT var_101 FROM decimal_table WHERE var_190>=0 GROUP BY var_101 ORDER BY "
      "var_101;");
  check_tables<double>(
      "SELECT var_103, SUM(var_172) AS VAR_SUM FROM decimal_dataframe GROUP BY var_103 "
      "ORDER BY var_103;",
      "SELECT var_103, SUM(var_172) AS VAR_SUM FROM decimal_table GROUP BY var_103 ORDER "
      "BY var_103;");

  check_tables<double>(
      "SELECT var_1, MAX(var_92) AS VAR_MAX FROM decimal_dataframe GROUP BY var_1 ORDER "
      "BY var_1;",
      "SELECT var_1, MAX(var_92) AS VAR_MAX FROM decimal_table GROUP BY var_1 ORDER BY "
      "var_1;");
}

TEST_F(DecimalTest, FragmentsTableDecimal) {
  ASSERT_NO_THROW(run_ddl_statement("CREATE DATAFRAME decimal_dataframe_frag100" +
                                    decimal_table_format +
                                    " FROM "
                                    "'CSV:../../Tests/Import/datafiles/"
                                    "santander_top1000.csv' WITH (fragment_size=100);"));
  ASSERT_NO_THROW(run_ddl_statement("CREATE DATAFRAME decimal_dataframe_frag333" +
                                    decimal_table_format +
                                    " FROM "
                                    "'CSV:../../Tests/Import/datafiles/"
                                    "santander_top1000.csv' WITH (fragment_size=333);"));
  std::string col_names = "";
  const size_t n_decimal_cols = 200;

  for (size_t i = 0; i < n_decimal_cols - 1; ++i) {
    col_names += "var_" + std::to_string(i) + ", ";
  }
  col_names += "var_" + std::to_string(n_decimal_cols - 1);

  check_tables<double>("SELECT " + col_names + " FROM decimal_dataframe ORDER BY var_1;",
                       "SELECT " + col_names + " FROM decimal_table ORDER BY var_1;");
  check_tables<double>(
      "SELECT " + col_names + " FROM decimal_dataframe_frag100 ORDER BY var_1;",
      "SELECT " + col_names + " FROM decimal_table ORDER BY var_1;");
  check_tables<double>(
      "SELECT " + col_names + " FROM decimal_dataframe_frag333 ORDER BY var_1;",
      "SELECT " + col_names + " FROM decimal_table ORDER BY var_1;");
}

TEST(DataframeOptionsTest, SkipRowsTest) {
  ASSERT_NO_THROW(
      run_ddl_statement("CREATE DATAFRAME opt_dataframe (int4 INTEGER, int8 BIGINT) FROM "
                        "'CSV:../../Tests/Import/datafiles/dataframe_options.csv' with "
                        "(DELIMITER='|', SKIP_ROWS=2);"));
  ASSERT_EQ(4, v<int64_t>(run_simple_agg("SELECT sum(int4) FROM opt_dataframe;")));
}

TEST(DataframeOptionsTest, HeaderlessTest) {
  ASSERT_NO_THROW(run_ddl_statement(
      "CREATE DATAFRAME opt_dataframe2 (int4 INTEGER, int8 BIGINT) FROM "
      "'CSV:../../Tests/Import/datafiles/dataframe_options.csv' with (DELIMITER='|', "
      "SKIP_ROWS=3, HEADER='false');"));
  ASSERT_EQ(8, v<int64_t>(run_simple_agg("SELECT sum(int8) FROM opt_dataframe2;")));
}

TEST(NullValuesTest, NullDifferentTypes) {
  run_ddl_statement(
      "CREATE DATAFRAME fsi_nulls (int4 INTEGER, int8 BIGINT, fp4 FLOAT, fp8 DOUBLE, ts "
      "TIMESTAMP, b8 BOOLEAN) from "
      "'CSV:../../Tests/Import/datafiles/null_values_numeric.csv';");
  check_query<int64_t>("SELECT int4 FROM fsi_nulls WHERE fp8 IS NULL;", {12});

  CHECK_EQ(65,
           v<int64_t>(run_simple_agg("SELECT int8 FROM fsi_nulls WHERE int4 IS NULL;")));

  EXPECT_FLOAT_EQ(
      34.2, v<float>(run_simple_agg("SELECT fp4 FROM fsi_nulls WHERE int8 IS NULL;")));

  check_query<double>("SELECT fp8 FROM fsi_nulls WHERE fp4 IS NULL;", {76.2});

  check_query<int64_t>("SELECT int4 FROM fsi_nulls WHERE ts IS NULL;", {12});

  check_query<int64_t>("SELECT int4 FROM fsi_nulls WHERE b8 IS NULL;", {12});
}

TEST(NullValuesTest, NullFullColumn) {
  run_ddl_statement(
      "CREATE DATAFRAME fsi_nulls_full (int4 INTEGER, int8 BIGINT, fp4 FLOAT, fp8 "
      "DOUBLE, b8 BOOLEAN) "
      "from 'CSV:../../Tests/Import/datafiles/null_values_full_column.csv';");
  CHECK_EQ(0,
           v<int64_t>(run_simple_agg(
               "SELECT COUNT(int4) FROM fsi_nulls_full WHERE int4 IS NOT NULL;")));

  CHECK_EQ(0,
           v<int64_t>(run_simple_agg(
               "SELECT COUNT(int8) FROM fsi_nulls_full WHERE int8 IS NOT NULL;")));

  CHECK_EQ(0,
           v<int64_t>(run_simple_agg(
               "SELECT COUNT(fp4) FROM fsi_nulls_full WHERE fp4 IS NOT NULL;")));

  CHECK_EQ(0,
           v<int64_t>(run_simple_agg(
               "SELECT COUNT(fp8) FROM fsi_nulls_full WHERE fp8 IS NOT NULL;")));

  CHECK_EQ(0,
           v<int64_t>(run_simple_agg(
               "SELECT COUNT(fp8) FROM fsi_nulls_full WHERE b8 IS NOT NULL;")));
}

TEST(NullValuesTest, NullFragmentedColumn) {
  run_ddl_statement(
      "CREATE DATAFRAME fsi_nulls_frag (int4 INTEGER, int8 BIGINT, fp4 FLOAT, fp8 "
      "DOUBLE) "
      "from 'CSV:../../Tests/Import/datafiles/null_values_fragments.csv' WITH "
      "(fragment_size=50);");
  CHECK_EQ(45, v<int64_t>(run_simple_agg("SELECT SUM(int4) FROM fsi_nulls_frag;")));

  CHECK_EQ(45, v<int64_t>(run_simple_agg("SELECT SUM(int8) FROM fsi_nulls_frag;")));

  EXPECT_FLOAT_EQ(45.f, v<float>(run_simple_agg("SELECT SUM(fp4) FROM fsi_nulls_frag;")));

  EXPECT_DOUBLE_EQ(45.,
                   v<double>(run_simple_agg("SELECT SUM(fp8) FROM fsi_nulls_frag;")));
}

TEST(NullValuesTest, NullTextColumn) {
  run_ddl_statement(
      "CREATE DATAFRAME fsi_nulls_text (col0 TEXT ENCODING DICT, col1 INTEGER, col2 "
      "CHAR(4), col3 TEXT ENCODING NONE) from "
      "'CSV:../../Tests/Import/datafiles/null_values_text.csv';");
  CHECK_EQ(4,
           v<int64_t>(run_simple_agg(
               "SELECT COUNT(col1) FROM fsi_nulls_text WHERE col0 IS NULL;")));

  CHECK_EQ(1,
           v<int64_t>(run_simple_agg(
               "SELECT COUNT(col2) FROM fsi_nulls_text WHERE col2 IS NOT NULL;")));
  CHECK_EQ(1,
           v<int64_t>(run_simple_agg(
               "SELECT COUNT(col3) FROM fsi_nulls_text WHERE col3 IS NOT NULL;")));
}

}  // namespace

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);

  namespace po = boost::program_options;

  po::options_description desc("Options");

  // these two are here to allow passing correctly google testing parameters
  desc.add_options()("gtest_list_tests", "list all tests");
  desc.add_options()("gtest_filter", "filters tests, use --help for details");

  desc.add_options()("test-help",
                     "Print all ArrowCsvForeighStorageTest specific options (for gtest "
                     "options use `--help`).");

  logger::LogOptions log_options(argv[0]);
  log_options.max_files_ = 0;  // stderr only by default
  desc.add(log_options.get_options());

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

  if (vm.count("test-help")) {
    std::cout << "Usage: ArrowCsvForeighStorageTest" << std::endl << std::endl;
    std::cout << desc << std::endl;
    return 0;
  }

  logger::init(log_options);

  QR::init(BASE_PATH);
  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  QR::reset();
  return err;
}
