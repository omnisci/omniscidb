#include <gtest/gtest.h>

#include <level_zero/ze_api.h>
#include "TestHelpers.h"

TEST(L0BuildTest, Init) {
  auto res = zeInit(0);
  ASSERT_EQ(res, ZE_RESULT_SUCCESS);
}

int main(int argc, char** argv) {
  TestHelpers::init_logger_stderr_only(argc, argv);
  testing::InitGoogleTest(&argc, argv);
  int err = RUN_ALL_TESTS();
  return err;
}
