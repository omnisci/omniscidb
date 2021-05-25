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

#include <boost/filesystem.hpp>
#include <fstream>

#include "../QueryEngine/Execute.h"
#include "../QueryEngine/InputMetadata.h"
#include "../QueryRunner/QueryRunner.h"

#ifndef BASE_PATH
#define BASE_PATH "./tmp"
#endif

extern bool g_is_test_env;
extern bool g_enable_watchdog;
extern size_t g_big_group_threshold;
extern size_t g_watchdog_baseline_max_groups;

using QR = QueryRunner::QueryRunner;
using namespace TestHelpers;

inline void run_ddl_statement(const std::string& input_str) {
  QR::get()->runDDLStatement(input_str);
}

bool skip_tests(const ExecutorDeviceType device_type) {
#ifdef HAVE_CUDA
  return device_type == ExecutorDeviceType::GPU && !(QR::get()->gpusPresent());
#else
  return device_type == ExecutorDeviceType::GPU;
#endif
}

#define SKIP_NO_GPU()                                        \
  if (skip_tests(dt)) {                                      \
    CHECK(dt == ExecutorDeviceType::GPU);                    \
    LOG(WARNING) << "GPU not available, skipping GPU tests"; \
    continue;                                                \
  }
class HighCardinalityStringEnv : public ::testing::Test {
 protected:
  void SetUp() override {
    run_ddl_statement("DROP TABLE IF EXISTS high_cardinality_str;");
    run_ddl_statement(
        "CREATE TABLE high_cardinality_str (x INT, str TEXT ENCODING DICT (32));");
    QR::get()->runSQL("INSERT INTO high_cardinality_str VALUES (1, 'hi');",
                      ExecutorDeviceType::CPU);
    QR::get()->runSQL("INSERT INTO high_cardinality_str VALUES (2, 'bye');",
                      ExecutorDeviceType::CPU);
  }

  void TearDown() override {
    run_ddl_statement("DROP TABLE IF EXISTS high_cardinality_str;");
  }
};

TEST_F(HighCardinalityStringEnv, PerfectHashNoFallback) {
  // make our own executor with a custom col ranges cache
  auto executor =
      Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID, "", "", SystemParameters());
  auto cat = QR::get()->getCatalog().get();
  CHECK(cat);
  executor->setCatalog(cat);

  auto td = cat->getMetadataForTable("high_cardinality_str");
  CHECK(td);
  auto cd = cat->getMetadataForColumn(td->tableId, "str");
  CHECK(cd);
  auto filter_cd = cat->getMetadataForColumn(td->tableId, "x");
  CHECK(filter_cd);

  PhysicalInput group_phys_input{cd->columnId, td->tableId};
  PhysicalInput filter_phys_input{filter_cd->columnId, td->tableId};

  std::unordered_set<PhysicalInput> phys_inputs{group_phys_input, filter_phys_input};
  std::unordered_set<int> phys_table_ids;
  phys_table_ids.insert(group_phys_input.table_id);
  executor->setupCaching(phys_inputs, phys_table_ids);

  auto input_descs = std::vector<InputDescriptor>{InputDescriptor(td->tableId, 0)};
  std::list<std::shared_ptr<const InputColDescriptor>> input_col_descs;
  input_col_descs.push_back(
      std::make_shared<InputColDescriptor>(cd->columnId, td->tableId, 0));
  input_col_descs.push_back(
      std::make_shared<InputColDescriptor>(filter_cd->columnId, td->tableId, 0));

  std::vector<InputTableInfo> table_infos = get_table_infos(input_descs, executor.get());

  auto count_expr = makeExpr<Analyzer::AggExpr>(
      SQLTypeInfo(kBIGINT, false), kCOUNT, nullptr, false, nullptr);
  auto group_expr =
      makeExpr<Analyzer::ColumnVar>(cd->columnType, td->tableId, cd->columnId, 0);
  auto filter_col_expr = makeExpr<Analyzer::ColumnVar>(
      filter_cd->columnType, td->tableId, filter_cd->columnId, 0);
  Datum d{int64_t(1)};
  auto filter_val_expr = makeExpr<Analyzer::Constant>(SQLTypeInfo(kINT, false), false, d);
  auto simple_filter_expr = makeExpr<Analyzer::BinOper>(SQLTypeInfo(kBOOLEAN, false),
                                                        false,
                                                        SQLOps::kEQ,
                                                        SQLQualifier::kONE,
                                                        filter_col_expr,
                                                        filter_val_expr);
  RelAlgExecutionUnit ra_exe_unit{input_descs,
                                  input_col_descs,
                                  {simple_filter_expr},
                                  {},
                                  {},
                                  {group_expr},
                                  {count_expr.get()},
                                  nullptr,
                                  SortInfo{},
                                  0};

  ColumnCacheMap column_cache;
  size_t max_groups_buffer_entry_guess = 1;

  auto result =
      executor->executeWorkUnit(max_groups_buffer_entry_guess,
                                /*is_agg=*/true,
                                table_infos,
                                ra_exe_unit,
                                CompilationOptions::defaults(ExecutorDeviceType::CPU),
                                ExecutionOptions::defaults(),
                                *cat,
                                nullptr,
                                /*has_cardinality_estimation=*/false,
                                column_cache);
  EXPECT_TRUE(result);
  EXPECT_EQ(result->rowCount(), size_t(1));
  auto row = result->getNextRow(false, false);
  EXPECT_EQ(row.size(), size_t(1));
  EXPECT_EQ(v<int64_t>(row[0]), 1);
}

std::unordered_set<PhysicalInput> setup_str_col_caching(PhysicalInput& group_phys_input,
                                                        const int64_t min,
                                                        const int64_t max,
                                                        PhysicalInput& filter_phys_input,
                                                        Executor* executor) {
  std::unordered_set<PhysicalInput> phys_inputs{group_phys_input, filter_phys_input};
  std::unordered_set<int> phys_table_ids;
  phys_table_ids.insert(group_phys_input.table_id);
  executor->setupCaching(phys_inputs, phys_table_ids);
  auto filter_col_range = executor->getColRange(filter_phys_input);
  // reset the col range to trigger the optimization
  AggregatedColRange col_range_cache;
  col_range_cache.setColRange(group_phys_input,
                              ExpressionRange::makeIntRange(min, max, 0, false));
  col_range_cache.setColRange(filter_phys_input, filter_col_range);
  executor->setColRangeCache(col_range_cache);
  return phys_inputs;
}

TEST_F(HighCardinalityStringEnv, BaselineFallbackTest) {
  // make our own executor with a custom col ranges cache
  auto executor =
      Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID, "", "", SystemParameters());
  auto cat = QR::get()->getCatalog().get();
  CHECK(cat);
  executor->setCatalog(cat);

  auto td = cat->getMetadataForTable("high_cardinality_str");
  CHECK(td);
  auto cd = cat->getMetadataForColumn(td->tableId, "str");
  CHECK(cd);
  auto filter_cd = cat->getMetadataForColumn(td->tableId, "x");
  CHECK(filter_cd);

  PhysicalInput group_phys_input{cd->columnId, td->tableId};
  PhysicalInput filter_phys_input{filter_cd->columnId, td->tableId};

  // 134217728 is 1 additional value over the max buffer size
  auto phys_inputs = setup_str_col_caching(
      group_phys_input, /*min=*/0, /*max=*/134217728, filter_phys_input, executor.get());

  auto input_descs = std::vector<InputDescriptor>{InputDescriptor(td->tableId, 0)};
  std::list<std::shared_ptr<const InputColDescriptor>> input_col_descs;
  input_col_descs.push_back(
      std::make_shared<InputColDescriptor>(cd->columnId, td->tableId, 0));
  input_col_descs.push_back(
      std::make_shared<InputColDescriptor>(filter_cd->columnId, td->tableId, 0));

  std::vector<InputTableInfo> table_infos = get_table_infos(input_descs, executor.get());

  auto count_expr = makeExpr<Analyzer::AggExpr>(
      SQLTypeInfo(kBIGINT, false), kCOUNT, nullptr, false, nullptr);
  auto group_expr =
      makeExpr<Analyzer::ColumnVar>(cd->columnType, td->tableId, cd->columnId, 0);
  auto filter_col_expr = makeExpr<Analyzer::ColumnVar>(
      filter_cd->columnType, td->tableId, filter_cd->columnId, 0);
  Datum d{int64_t(1)};
  auto filter_val_expr = makeExpr<Analyzer::Constant>(SQLTypeInfo(kINT, false), false, d);
  auto simple_filter_expr = makeExpr<Analyzer::BinOper>(SQLTypeInfo(kBOOLEAN, false),
                                                        false,
                                                        SQLOps::kEQ,
                                                        SQLQualifier::kONE,
                                                        filter_col_expr,
                                                        filter_val_expr);
  RelAlgExecutionUnit ra_exe_unit{input_descs,
                                  input_col_descs,
                                  {simple_filter_expr},
                                  {},
                                  {},
                                  {group_expr},
                                  {count_expr.get()},
                                  nullptr,
                                  SortInfo{},
                                  0};

  ColumnCacheMap column_cache;
  size_t max_groups_buffer_entry_guess = 1;
  // expect throw w/out cardinality estimation
  EXPECT_THROW(
      executor->executeWorkUnit(max_groups_buffer_entry_guess,
                                /*is_agg=*/true,
                                table_infos,
                                ra_exe_unit,
                                CompilationOptions::defaults(ExecutorDeviceType::CPU),
                                ExecutionOptions::defaults(),
                                *cat,
                                nullptr,
                                /*has_cardinality_estimation=*/false,
                                column_cache),
      CardinalityEstimationRequired);

  auto result =
      executor->executeWorkUnit(max_groups_buffer_entry_guess,
                                /*is_agg=*/true,
                                table_infos,
                                ra_exe_unit,
                                CompilationOptions::defaults(ExecutorDeviceType::CPU),
                                ExecutionOptions::defaults(),
                                *cat,
                                nullptr,
                                /*has_cardinality_estimation=*/true,
                                column_cache);
  EXPECT_TRUE(result);
  EXPECT_EQ(result->rowCount(), size_t(1));
  auto row = result->getNextRow(false, false);
  EXPECT_EQ(row.size(), size_t(1));
  EXPECT_EQ(v<int64_t>(row[0]), 1);
}

TEST_F(HighCardinalityStringEnv, BaselineNoFilters) {
  // make our own executor with a custom col ranges cache
  auto executor =
      Executor::getExecutor(Executor::UNITARY_EXECUTOR_ID, "", "", SystemParameters());
  auto cat = QR::get()->getCatalog().get();
  CHECK(cat);
  executor->setCatalog(cat);

  auto td = cat->getMetadataForTable("high_cardinality_str");
  CHECK(td);
  auto cd = cat->getMetadataForColumn(td->tableId, "str");
  CHECK(cd);
  auto filter_cd = cat->getMetadataForColumn(td->tableId, "x");
  CHECK(filter_cd);

  PhysicalInput group_phys_input{cd->columnId, td->tableId};
  PhysicalInput filter_phys_input{filter_cd->columnId, td->tableId};

  // 134217728 is 1 additional value over the max buffer size
  auto phys_inputs = setup_str_col_caching(
      group_phys_input, /*min=*/0, /*max=*/134217728, filter_phys_input, executor.get());

  auto input_descs = std::vector<InputDescriptor>{InputDescriptor(td->tableId, 0)};
  std::list<std::shared_ptr<const InputColDescriptor>> input_col_descs;
  input_col_descs.push_back(
      std::make_shared<InputColDescriptor>(cd->columnId, td->tableId, 0));
  input_col_descs.push_back(
      std::make_shared<InputColDescriptor>(filter_cd->columnId, td->tableId, 0));

  std::vector<InputTableInfo> table_infos = get_table_infos(input_descs, executor.get());

  auto count_expr = makeExpr<Analyzer::AggExpr>(
      SQLTypeInfo(kBIGINT, false), kCOUNT, nullptr, false, nullptr);
  auto group_expr =
      makeExpr<Analyzer::ColumnVar>(cd->columnType, td->tableId, cd->columnId, 0);

  RelAlgExecutionUnit ra_exe_unit{input_descs,
                                  input_col_descs,
                                  {},
                                  {},
                                  {},
                                  {group_expr},
                                  {count_expr.get()},
                                  nullptr,
                                  SortInfo{},
                                  0};

  ColumnCacheMap column_cache;
  size_t max_groups_buffer_entry_guess = 1;
  // no filters, so expect no throw w/out cardinality estimation
  auto result =
      executor->executeWorkUnit(max_groups_buffer_entry_guess,
                                /*is_agg=*/true,
                                table_infos,
                                ra_exe_unit,
                                CompilationOptions::defaults(ExecutorDeviceType::CPU),
                                ExecutionOptions::defaults(),
                                *cat,
                                nullptr,
                                /*has_cardinality_estimation=*/false,
                                column_cache);
  EXPECT_TRUE(result);
  EXPECT_EQ(result->rowCount(), size_t(2));
  {
    auto row = result->getNextRow(false, false);
    EXPECT_EQ(row.size(), size_t(1));
    EXPECT_EQ(v<int64_t>(row[0]), 1);
  }
  {
    auto row = result->getNextRow(false, false);
    EXPECT_EQ(row.size(), size_t(1));
    EXPECT_EQ(v<int64_t>(row[0]), 1);
  }
}

class LowCardinalityThresholdTest : public ::testing::Test {
 protected:
  void SetUp() override {
    run_ddl_statement("DROP TABLE IF EXISTS low_cardinality;");
    run_ddl_statement("CREATE TABLE low_cardinality (fl text,ar text, dep text);");

    // write some data to a file
    boost::filesystem::path temp_path =
        boost::filesystem::temp_directory_path() / boost::filesystem::unique_path();
    const std::string filename_with_ext = temp_path.native() + ".csv";
    std::fstream f(filename_with_ext, f.binary | f.out | f.trunc);
    CHECK(f.is_open());
    for (size_t i = 0; i < g_big_group_threshold; i++) {
      f << i << ", " << i + 1 << ", " << i + 2 << std::endl;
    }
    f.close();

    run_ddl_statement("COPY low_cardinality FROM '" + filename_with_ext +
                      "' WITH (header='false');");
  }

  void TearDown() override { run_ddl_statement("DROP TABLE IF EXISTS low_cardinality;"); }
};

TEST_F(LowCardinalityThresholdTest, GroupBy) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    auto result = QR::get()->runSQL(
        R"(select fl,ar,dep from low_cardinality group by fl,ar,dep;)", dt);
    EXPECT_EQ(result->rowCount(), g_big_group_threshold);
  }
}

class BigCardinalityThresholdTest : public ::testing::Test {
 protected:
  void SetUp() override {
    g_enable_watchdog = true;
    initial_g_watchdog_baseline_max_groups = g_watchdog_baseline_max_groups;
    g_watchdog_baseline_max_groups = g_big_group_threshold + 1;

    run_ddl_statement("DROP TABLE IF EXISTS big_cardinality;");
    run_ddl_statement("CREATE TABLE big_cardinality (fl text,ar text, dep text);");

    // write some data to a file
    boost::filesystem::path temp_path =
        boost::filesystem::temp_directory_path() / boost::filesystem::unique_path();
    const std::string filename_with_ext = temp_path.native() + ".csv";
    std::fstream f(filename_with_ext, f.binary | f.out | f.trunc);
    CHECK(f.is_open());
    // add enough groups to trigger the watchdog exception if we use a poor estimate
    for (size_t i = 0; i < g_watchdog_baseline_max_groups; i++) {
      f << i << ", " << i + 1 << ", " << i + 2 << std::endl;
    }
    f.close();

    run_ddl_statement("COPY big_cardinality FROM '" + filename_with_ext +
                      "' WITH (header='false');");
  }

  void TearDown() override {
    g_enable_watchdog = false;
    g_watchdog_baseline_max_groups = initial_g_watchdog_baseline_max_groups;
    run_ddl_statement("DROP TABLE IF EXISTS big_cardinality;");
  }

  size_t initial_g_watchdog_baseline_max_groups{0};
};

TEST_F(BigCardinalityThresholdTest, EmptyFilters) {
  for (auto dt : {ExecutorDeviceType::CPU, ExecutorDeviceType::GPU}) {
    SKIP_NO_GPU();

    auto result = QR::get()->runSQL(
        R"(SELECT fl,ar,dep FROM big_cardinality WHERE fl = 'a' GROUP BY fl,ar,dep;)",
        dt);
    EXPECT_EQ(result->rowCount(), size_t(0));
  }
}

int main(int argc, char** argv) {
  g_is_test_env = true;

  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  namespace po = boost::program_options;

  po::options_description desc("Options");

  logger::LogOptions log_options(argv[0]);
  log_options.max_files_ = 0;  // stderr only by default
  desc.add(log_options.get_options());

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).run(), vm);
  po::notify(vm);

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
