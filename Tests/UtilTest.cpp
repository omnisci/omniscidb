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

#include "Shared/Intervals.h"
#include "TestHelpers.h"
#include "Utils/Regexp.h"
#include "Utils/StringLike.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <future>

// for (auto const interval : makeIntervals(0, M, n_workers)) {...}
// iterates over interval={begin,end} pairs which satisfy:
// Requirement_1) Mutually exclusive and complete coverage over [0,M).
// Requirement_2) Difference between max(end-begin) and min(end-begin) is either 0 or 1.
TEST(Shared, Intervals) {
  constexpr int M = 50;
  std::array<std::atomic<int>, M> array;
  for (int n_workers = 1; n_workers <= 2 * M; ++n_workers) {
    std::for_each(array.begin(), array.end(), [](auto& v) { v = 0; });
    std::vector<utils::future<void>> threads;
    int max_interval_size = std::numeric_limits<int>::min();
    int min_interval_size = std::numeric_limits<int>::max();
    for (auto const interval : makeIntervals(0, M, n_workers)) {
      EXPECT_LT(interval.begin, interval.end);
      max_interval_size = std::max(max_interval_size, interval.end - interval.begin);
      min_interval_size = std::min(min_interval_size, interval.end - interval.begin);
      threads.push_back(utils::async(
          [&array](Interval<int> interval) {
            for (int i = interval.begin; i < interval.end; ++i) {
              ++array.at(i);
            }
          },
          interval));
    }
    for (auto& thread : threads) {
      thread.wait();
    }
    // Test Requirement_1
    for (auto const& v : array) {
      EXPECT_EQ(v, 1);
    }
    // Test Requirement_2
    EXPECT_LE(0, max_interval_size - min_interval_size);
    EXPECT_LE(max_interval_size - min_interval_size, 1);
    // If M < n_workers, then there is no need to spawn unneeded workers.
    EXPECT_LE(threads.size(), static_cast<size_t>(std::numeric_limits<int>::max()));
    EXPECT_LE(static_cast<int>(threads.size()), std::min(M, n_workers));
  }
}

TEST(Shared, IntervalsBounds) {
  using Integer = int8_t;
  using Unsigned = std::make_unsigned<Integer>::type;
  constexpr int M = 255;  // Number of elements in [-128,127)
  std::array<std::atomic<int>, M> array;
  // Test with 0-value, and values that go outside the range of Integer.
  for (unsigned n_workers = 0; n_workers <= 260; ++n_workers) {
    std::for_each(array.begin(), array.end(), [](auto& v) { v = 0; });
    std::vector<std::future<void>> threads;
    Integer const begin = std::numeric_limits<Integer>::min();
    Integer const end = std::numeric_limits<Integer>::max();
    int const offset = -static_cast<int>(begin);
    Unsigned max_interval_size = std::numeric_limits<Unsigned>::min();
    Unsigned min_interval_size = std::numeric_limits<Unsigned>::max();
    for (auto const interval : makeIntervals(begin, end, n_workers)) {
      EXPECT_LT(interval.begin, interval.end);
      max_interval_size = std::max(max_interval_size,
                                   static_cast<Unsigned>(interval.end - interval.begin));
      min_interval_size = std::min(min_interval_size,
                                   static_cast<Unsigned>(interval.end - interval.begin));
      threads.push_back(std::async(
          std::launch::async,
          [&array](Interval<Integer> interval) {
            for (Integer i = interval.begin; i < interval.end; ++i) {
              ++array.at(offset + i);
            }
          },
          interval));
    }
    for (auto& thread : threads) {
      thread.wait();
    }
    // Test Requirement_1
    for (auto const& v : array) {
      EXPECT_EQ(v, int(0 != n_workers));
    }
    // Test Requirement_2
    if (n_workers) {
      EXPECT_LE(0, max_interval_size - min_interval_size);
      EXPECT_LE(max_interval_size - min_interval_size, 1);
    } else {  // unchanged
      EXPECT_EQ(max_interval_size, std::numeric_limits<Unsigned>::min());
      EXPECT_EQ(min_interval_size, std::numeric_limits<Unsigned>::max());
    }
    // If M < n_workers, then there is no need to spawn unneeded workers.
    EXPECT_LE(threads.size(), static_cast<size_t>(std::numeric_limits<int>::max()));
    EXPECT_LE(static_cast<int>(threads.size()), std::min(M, static_cast<int>(n_workers)));
  }
}

TEST(Shared, IntervalsInvalidBounds) {
  using Integer = int;
  using Unsigned = std::make_unsigned<Integer>::type;
  bool loop_body_executed = false;
  // Expect loop body is never executed since begin == end.
  for (auto const interval : makeIntervals(10, 10, 1)) {
    (void)interval;
    loop_body_executed = true;
  }
  EXPECT_FALSE(loop_body_executed);
  // Expect loop body is never executed since begin > end.
  for (auto const interval : makeIntervals(10, 9, 1)) {
    (void)interval;
    loop_body_executed = true;
  }
  EXPECT_FALSE(loop_body_executed);
  // Expect loop body is never executed since n_workers == 0.
  for (auto const interval : makeIntervals(9, 10, 0)) {
    (void)interval;
    loop_body_executed = true;
  }
  EXPECT_FALSE(loop_body_executed);
  // Sanity check that loop body is entered with valid values.
  for (auto const interval : makeIntervals(9, 10, 1)) {
    (void)interval;
    loop_body_executed = true;
  }
  EXPECT_TRUE(loop_body_executed);
}

TEST(Utils, StringLike) {
  ASSERT_TRUE(string_like("abc", 3, "abc", 3, '\\'));
  ASSERT_FALSE(string_like("abc", 3, "ABC", 3, '\\'));
  ASSERT_TRUE(string_ilike("Xyzabc", 6, "xyz%", 4, '\\'));
  ASSERT_TRUE(string_like("abcxyzefg", 9, "%xyz%", 5, '\\'));
  ASSERT_TRUE(string_like("abcxyzefgXYZhij", 15, "%xyz%XYZ%", 9, '\\'));
  ASSERT_TRUE(string_like("abcxOzefgXpZhij", 15, "%x_z%X_Z%", 9, '\\'));
  ASSERT_TRUE(string_like("abc100%efg", 10, "%100!%___", 9, '!'));
  ASSERT_TRUE(string_like("[ hello", 7, "%\\[%", 4, '\\'));
  ASSERT_TRUE(string_like("hello [", 7, "%\\[%", 4, '\\'));
}

TEST(Utils, Regexp) {
  ASSERT_TRUE(regexp_like("abc", 3, "abc", 3, '\\'));
  ASSERT_FALSE(regexp_like("abc", 3, "ABC", 3, '\\'));
  ASSERT_TRUE(regexp_like("Xyzabc", 6, "[xX]yz.*", 8, '\\'));
  ASSERT_TRUE(regexp_like("abcxyzefg", 9, ".*xyz.*", 7, '\\'));
  ASSERT_TRUE(regexp_like("abcxyzefgXYZhij", 15, ".*xyz.*XYZ.*", 12, '\\'));
  ASSERT_TRUE(regexp_like("abcxOzefgXpZhij", 15, ".+x.z.*X.Z.*", 12, '\\'));
  // Custom escape characters are not yet supported.
  ASSERT_FALSE(regexp_like("abc100%efg", 10, ".+100!%...", 10, '!'));
  ASSERT_TRUE(regexp_like("[ hello", 7, ".*\\[.*", 6, '\\'));
  ASSERT_TRUE(regexp_like("hello [", 7, ".*\\[.*", 6, '\\'));
}

int main(int argc, char* argv[]) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);

  int err{0};
  try {
    err = RUN_ALL_TESTS();
  } catch (const std::exception& e) {
    LOG(ERROR) << e.what();
  }
  return err;
}
