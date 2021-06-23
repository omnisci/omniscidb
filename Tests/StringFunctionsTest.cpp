/*
 * Copyright 2019 OmniSci, Inc.
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

/**
 * @file StringFunctionsTest.cpp
 * @brief Test suite for string functions
 */

#include <QueryEngine/ResultSet.h>
#include <gtest/gtest.h>
#include <boost/format.hpp>
#include <boost/locale/generator.hpp>

#include "Catalog/Catalog.h"

#include "../QueryRunner/QueryRunner.h"
#include "TestHelpers.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

extern bool g_enable_experimental_string_functions;

namespace {
inline auto sql(const std::string& sql_stmt) {
  return QueryRunner::QueryRunner::get()->runSQL(
      sql_stmt, ExecutorDeviceType::CPU, true, true);
}

inline auto multi_sql(const std::string& sql_stmts) {
  return QueryRunner::QueryRunner::get()
      ->runMultipleStatements(sql_stmts, ExecutorDeviceType::CPU)
      .back();
}

class AssertValueEqualsVisitor : public boost::static_visitor<> {
 public:
  AssertValueEqualsVisitor(const size_t& row, const size_t& column)
      : row(row), column(column) {}

  template <typename T, typename U>
  void operator()(const T& expected, const U& actual) const {
    FAIL() << boost::format(
                  "Values are of different types. Expected result set value: %s is of "
                  "type: %s while actual result set value: %s is of type: %s. At row: "
                  "%d, column: %d") %
                  expected % typeid(expected).name() % actual % typeid(actual).name() %
                  row % column;
  }

  template <typename T>
  void operator()(const T& expected, const T& actual) const {
    EXPECT_EQ(expected, actual) << boost::format("At row: %d, column: %d") % row % column;
  }

 private:
  size_t row;
  size_t column;
};

template <>
void AssertValueEqualsVisitor::operator()<NullableString>(
    const NullableString& expected,
    const NullableString& actual) const {
  boost::apply_visitor(AssertValueEqualsVisitor(row, column), expected, actual);
}

void assert_value_equals(ScalarTargetValue& expected,
                         ScalarTargetValue& actual,
                         const size_t& row,
                         const size_t& column) {
  boost::apply_visitor(AssertValueEqualsVisitor(row, column), expected, actual);
}

void compare_result_set(
    const std::vector<std::vector<ScalarTargetValue>>& expected_result_set,
    const std::shared_ptr<ResultSet>& actual_result_set) {
  auto row_count = actual_result_set->rowCount(false);
  ASSERT_EQ(expected_result_set.size(), row_count)
      << "Returned result set does not have the expected number of rows";

  if (row_count == 0) {
    return;
  }

  auto expected_column_count = expected_result_set[0].size();
  auto column_count = actual_result_set->colCount();
  ASSERT_EQ(expected_column_count, column_count)
      << "Returned result set does not have the expected number of columns";
  ;

  for (size_t r = 0; r < row_count; ++r) {
    auto row = actual_result_set->getNextRow(true, true);
    for (size_t c = 0; c < column_count; c++) {
      auto column_value = boost::get<ScalarTargetValue>(row[c]);
      auto expected_column_value = expected_result_set[r][c];
      assert_value_equals(expected_column_value, column_value, r, c);
    }
  }
}
}  // namespace

// begin LOWER function tests

/**
 * @brief Class used for setting up and tearing down tables and records that are required
 * by the LOWER function test cases
 */
class LowerFunctionTest : public testing::Test {
 public:
  void SetUp() override {
    ASSERT_NO_THROW(multi_sql(R"(
        drop table if exists lower_function_test_people;
        create table lower_function_test_people(first_name text, last_name text encoding none, age integer, country_code text);
        insert into lower_function_test_people values('JOHN', 'SMITH', 25, 'us');
        insert into lower_function_test_people values('John', 'Banks', 30, 'Us');
        insert into lower_function_test_people values('JOHN', 'Wilson', 20, 'cA');
        insert into lower_function_test_people values('Sue', 'Smith', 25, 'CA');
        drop table if exists lower_function_test_countries;
        create table lower_function_test_countries(code text, name text, capital text encoding none);
        insert into lower_function_test_countries values('US', 'United States', 'Washington');
        insert into lower_function_test_countries values('ca', 'Canada', 'Ottawa');
        insert into lower_function_test_countries values('Gb', 'United Kingdom', 'London');
        insert into lower_function_test_countries values('dE', 'Germany', 'Berlin');
      )"););
  }

  void TearDown() override {
    ASSERT_NO_THROW(multi_sql(R"(
        drop table lower_function_test_people;
        drop table lower_function_test_countries;
      )"););
  }
};

TEST_F(LowerFunctionTest, LowercaseProjection) {
  auto result_set = sql("select lower(first_name) from lower_function_test_people;");
  std::vector<std::vector<ScalarTargetValue>> expected_result_set{
      {"john"}, {"john"}, {"john"}, {"sue"}};
  compare_result_set(expected_result_set, result_set);
}

TEST_F(LowerFunctionTest, LowercaseFilter) {
  auto result_set =
      sql("select first_name, last_name from lower_function_test_people "
          "where lower(country_code) = 'us';");
  std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"JOHN", "SMITH"},
                                                                  {"John", "Banks"}};
  compare_result_set(expected_result_set, result_set);
}

TEST_F(LowerFunctionTest, MultipleLowercaseFilters) {
  auto result_set =
      sql("select first_name, last_name from lower_function_test_people "
          "where lower(country_code) = 'us' or lower(first_name) = 'sue';");
  std::vector<std::vector<ScalarTargetValue>> expected_result_set{
      {"JOHN", "SMITH"}, {"John", "Banks"}, {"Sue", "Smith"}};
  compare_result_set(expected_result_set, result_set);
}

TEST_F(LowerFunctionTest, MixedFilters) {
  auto result_set =
      sql("select first_name, last_name from lower_function_test_people "
          "where lower(country_code) = 'ca' and age > 20;");
  std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"Sue", "Smith"}};
  compare_result_set(expected_result_set, result_set);
}

TEST_F(LowerFunctionTest, LowercaseGroupBy) {
  auto result_set =
      sql("select lower(first_name), count(*) from lower_function_test_people "
          "group by lower(first_name) order by lower(first_name);");
  std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"john", int64_t(3)},
                                                                  {"sue", int64_t(1)}};
  compare_result_set(expected_result_set, result_set);
}

TEST_F(LowerFunctionTest, LowercaseJoin) {
  auto result_set =
      sql("select first_name, name as country_name "
          "from lower_function_test_people "
          "join lower_function_test_countries on lower(country_code) = lower(code);");
  std::vector<std::vector<ScalarTargetValue>> expected_result_set{
      {"JOHN", "United States"},
      {"John", "United States"},
      {"JOHN", "Canada"},
      {"Sue", "Canada"}};
  compare_result_set(expected_result_set, result_set);
}

TEST_F(LowerFunctionTest, SelectLowercaseLiteral) {
  auto result_set =
      sql("select first_name, lower('SMiTH') from lower_function_test_people;");
  std::vector<std::vector<ScalarTargetValue>> expected_result_set{
      {"JOHN", "smith"}, {"John", "smith"}, {"JOHN", "smith"}, {"Sue", "smith"}};
  compare_result_set(expected_result_set, result_set);
}

TEST_F(LowerFunctionTest, UpdateLowercase_EncodedColumnOnly) {
  auto result_set = multi_sql(R"(
       update lower_function_test_people set country_code = lower(country_code);
       select country_code from lower_function_test_people;
     )");
  std::vector<std::vector<ScalarTargetValue>> expected_result_set{
      {"us"}, {"us"}, {"ca"}, {"ca"}};
  compare_result_set(expected_result_set, result_set);
}

/**
 * UPDATE statements with at least one non-encoded column follow a different code path
 * from those with only encoded columns (see StorageIOFacility::yieldUpdateCallback for
 * more details).InsertIntoSelectLowercase
 */
TEST_F(LowerFunctionTest, UpdateLowercase_EncodedAndNonEncodedColumns) {
  auto result_set = multi_sql(R"(
       update lower_function_test_people set last_name = last_name, country_code = lower(country_code);
       select last_name, country_code from lower_function_test_people;
     )");
  std::vector<std::vector<ScalarTargetValue>> expected_result_set{
      {"SMITH", "us"}, {"Banks", "us"}, {"Wilson", "ca"}, {"Smith", "ca"}};
  compare_result_set(expected_result_set, result_set);
}

// TODO: Re-enable after clear definition around handling non-ASCII characters
TEST_F(LowerFunctionTest, DISABLED_LowercaseNonAscii) {
  auto result_set = multi_sql(R"(
       insert into lower_function_test_people values('Ħ', 'Ħ', 25, 'GB');
       select lower(first_name), last_name from lower_function_test_people where country_code = 'GB';
     )");
  std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"ħ", "Ħ"}};
  compare_result_set(expected_result_set, result_set);
}

TEST_F(LowerFunctionTest, LowercaseNonEncodedTextColumn) {
  try {
    sql("select lower(last_name) from lower_function_test_people;");
    FAIL() << "An exception should have been thrown for this test case";
  } catch (const std::exception& e) {
    ASSERT_STREQ("LOWER expects a dictionary encoded text column or a literal.",
                 e.what());
  }
}

TEST_F(LowerFunctionTest, LowercaseNonTextColumn) {
  try {
    sql("select lower(age) from lower_function_test_people;");
    FAIL() << "An exception should have been thrown for this test case";
  } catch (const std::exception& e) {
    ASSERT_STREQ("LOWER expects a dictionary encoded text column or a literal.",
                 e.what());
  }
}

TEST_F(LowerFunctionTest, LowercaseGpuMode) {
#ifndef HAVE_CUDA
  LOG(ERROR)
      << "This test case only applies to uses case where CUDA is enabled. Skipping test.";
  return;
#else
  auto result_set = QueryRunner::QueryRunner::get()->runSQL(
      "select first_name, last_name from lower_function_test_people "
      "where lower(country_code) = 'us';",
      ExecutorDeviceType::GPU,
      true,
      true);
  std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"JOHN", "SMITH"},
                                                                  {"John", "Banks"}};
  compare_result_set(expected_result_set, result_set);
#endif
}

TEST_F(LowerFunctionTest, LowercaseNullColumn) {
  auto result_set = multi_sql(R"(
       insert into lower_function_test_people values(null, 'Empty', 25, 'US');
       select lower(first_name), last_name from lower_function_test_people where last_name = 'Empty';
     )");
  std::vector<std::vector<ScalarTargetValue>> expected_result_set{{"", "Empty"}};
  compare_result_set(expected_result_set, result_set);
}

TEST_F(LowerFunctionTest, SelectLowercase_ExperimentalStringFunctionsDisabled) {
  g_enable_experimental_string_functions = false;

  try {
    sql("select lower(first_name) from lower_function_test_people;");
    FAIL() << "An exception should have been thrown for this test case";
  } catch (const std::exception& e) {
    ASSERT_STREQ("Function LOWER(TEXT) not supported.", e.what());
    g_enable_experimental_string_functions = true;
  }
}

// end LOWER function tests

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  QueryRunner::QueryRunner::init(BASE_PATH);
  g_enable_experimental_string_functions = true;

  // Use system locale setting by default (as done in the server).
  boost::locale::generator generator;
  std::locale::global(generator.generate(""));

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }

  g_enable_experimental_string_functions = false;
  QueryRunner::QueryRunner::reset();
  return err;
}
