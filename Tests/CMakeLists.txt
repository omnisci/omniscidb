set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(TEST_BASE_PATH "./tmp")
add_definitions("-DBASE_PATH=\"${TEST_BASE_PATH}\"")

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/Udf/udf_sample.cpp DESTINATION ${CMAKE_BINARY_DIR}/Tests/Udf)

add_executable(CodeGeneratorTest CodeGeneratorTest.cpp)
add_executable(ExecuteTest ExecuteTest.cpp ClusterTester.cpp)
add_executable(GeospatialTest GeospatialTest.cpp)
add_executable(RunQueryLoop RunQueryLoop.cpp)
add_executable(StringDictionaryTest StringDictionaryTest.cpp)
add_executable(StringTransformTest StringTransformTest.cpp)
add_executable(StringFunctionsTest StringFunctionsTest.cpp)
add_executable(ProfileTest ProfileTest.cpp)
add_executable(ForeignServerDdlTest ForeignServerDdlTest.cpp)
add_executable(ShowCommandsDdlTest ShowCommandsDdlTest.cpp)
add_executable(CatalogMigrationTest CatalogMigrationTest.cpp)
add_executable(CreateAndDropTableDdlTest CreateAndDropTableDdlTest.cpp)
add_executable(ForeignTableDmlTest ForeignTableDmlTest.cpp)
add_executable(DashboardAndCustomExpressionTest DashboardAndCustomExpressionTest.cpp)
add_executable(FileMgrTest FileMgrTest.cpp)
add_executable(FilePathWhitelistTest FilePathWhitelistTest.cpp)
add_executable(EncoderTest EncoderTest.cpp)
add_executable(ForeignStorageCacheTest ForeignStorageCacheTest.cpp)
add_executable(PersistentStorageTest PersistentStorageTest.cpp)
add_executable(ShardedTableEpochConsistencyTest ShardedTableEpochConsistencyTest.cpp)
add_executable(DiskCacheQueryTest DiskCacheQueryTest.cpp)
add_executable(CachingFileMgrTest CachingFileMgrTest.cpp)
add_executable(JSONTest JSONTest.cpp)

if(ENABLE_L0)
  add_executable(SpirvBuildTest SpirvBuildTest.cpp)
  add_executable(SpirvExecuteTest SpirvExecuteTest.cpp)
endif()

if(ENABLE_CUDA)
  message(DEBUG "Tests CUDA_COMPILATION_ARCH: ${CUDA_COMPILATION_ARCH}")
  set(MAPD_DEFINITIONS -DHAVE_CUDA)
  if (SUPPRESS_NULL_LOGGER_DEPRECATION_WARNINGS)
    list(APPEND MAPD_DEFINITIONS -DSUPPRESS_NULL_LOGGER_DEPRECATION_WARNINGS)
  endif()
  add_custom_command(
    DEPENDS ProfileUtils.cu ProfileTest.h
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/libProfileUtils.so
    COMMAND nvcc
    ARGS
        ${MAPD_HOST_COMPILER_FLAG}
        -Xcompiler -fPIC
        -O3
        -D_FORCE_INLINES
        ${MAPD_DEFINITIONS}
        ${CUDA_COMPILATION_ARCH}
        -std=c++14
        -shared
        ${CMAKE_CURRENT_SOURCE_DIR}/ProfileUtils.cu
        -o ${CMAKE_CURRENT_BINARY_DIR}/libProfileUtils.so
    )
  add_custom_target(ProfileUtils DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libProfileUtils.so)
  add_dependencies(ProfileTest ProfileUtils)
  set(PROF_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/libProfileUtils.so)
endif()

add_executable(DumpRestoreTest DumpRestoreTest.cpp)
add_executable(ResultSetTest ResultSetTest.cpp ResultSetTestUtils.cpp)
add_executable(FromTableReorderingTest FromTableReorderingTest.cpp)
add_executable(ResultSetBaselineRadixSortTest ResultSetBaselineRadixSortTest.cpp ResultSetTestUtils.cpp)
add_executable(UtilTest UtilTest.cpp)
add_executable(StorageTest StorageTest.cpp PopulateTableRandom.cpp ScanTable.cpp)
add_executable(StoragePerfTest StoragePerfTest.cpp PopulateTableRandom.cpp ScanTable.cpp)
add_executable(ImportExportTest ImportExportTest.cpp)
add_executable(ArrowCsvForeignStorageTest ArrowCsvForeignStorageTest.cpp)
add_executable(AlterTableDdlTest AlterTableDdlTest.cpp)
add_executable(UpdelStorageTest UpdelStorageTest.cpp)
add_executable(ComputeMetadataTest ComputeMetadataTest.cpp)
add_executable(BumpAllocatorTest BumpAllocatorTest.cpp)
add_executable(SpecialCharsTest SpecialCharsTest.cpp)
add_executable(TableFunctionsTest TableFunctionsTest.cpp)
add_executable(ArrayTest ArrayTest.cpp)
add_executable(GroupByTest GroupByTest.cpp)
add_executable(ParallelExecutorsTest ParallelExecutorsTest.cpp)
add_executable(MigrationMgrTest MigrationMgrTest.cpp)
add_executable(TopKTest TopKTest.cpp)
add_executable(TokenCompletionHintsTest TokenCompletionHintsTest.cpp)
add_executable(OmniSQLCommandTest OmniSQLCommandTest.cpp)
add_executable(OmniSQLUtilitiesTest OmniSQLUtilitiesTest.cpp)
add_executable(DBObjectPrivilegesTest DBObjectPrivilegesTest.cpp)
add_executable(GeoTypesTest Geospatial/GeoTypesTest.cpp)
add_executable(OverlapsJoinTest OverlapsJoinTest.cpp)
add_executable(CtasUpdateTest CtasUpdateTest.cpp)
add_executable(CorrelatedSubqueryTest CorrelatedSubqueryTest.cpp)
add_executable(CtasIntegrationTest CtasIntegrationTest.cpp)
add_executable(DateTimeUtilsTest Shared/DateTimeUtilsTest.cpp)
add_executable(UpdateMetadataTest UpdateMetadataTest.cpp)
add_executable(CalciteOptimizeTest CalciteOptimizeTest.cpp)
add_executable(JoinHashTableTest JoinHashTableTest.cpp)
add_executable(CachedHashTableTest CachedHashTableTest.cpp)
add_executable(RuntimeInterruptTest RuntimeInterruptTest.cpp)
add_executable(ColumnarResultsTest ColumnarResultsTest.cpp ResultSetTestUtils.cpp)
add_executable(CommandLineTest CommandLineTest.cpp)
add_executable(SQLHintTest SQLHintTest.cpp)
add_executable(LoadTableTest LoadTableTest.cpp)
add_executable(QuantileCpuTest Quantile/QuantileCpuTest.cpp)

if(NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
  add_executable(UdfTest UdfTest.cpp)
endif()
if(ENABLE_CUDA)
  add_executable(GpuSharedMemoryTest GpuSharedMemoryTest.cpp ResultSetTestUtils.cpp)
elseif(ENABLE_DBE)
  add_executable(EmbeddedDatabaseTest EmbeddedDatabaseTest.cpp)
endif()
if(("${MAPD_EDITION_LOWER}" STREQUAL "ee") AND ENABLE_AWS_S3)
  add_executable(UserMappingDdlTest ee/UserMappingDdlTest.cpp)
  add_executable(PkiEncryptorTest ee/PkiEncryptorTest.cpp)
endif()

# Tests + Microbenchmarks
add_executable(TableUpdateDeleteBenchmark TableUpdateDeleteBenchmark.cpp)

set(EXECUTE_TEST_LIBS gtest mapd_thrift QueryRunner ${MAPD_LIBRARIES} ${CMAKE_DL_LIBS} ${CUDA_LIBRARIES} ${Boost_LIBRARIES} ${ZLIB_LIBRARIES} ${PROFILER_LIBS})
set(THRIFT_HANDLER_TEST_LIBRARIES thrift_handler ${EXECUTE_TEST_LIBS})

# Replace Licensing library with TestLicensing library
if("${MAPD_EDITION_LOWER}" STREQUAL "ee")
  list(FIND THRIFT_HANDLER_TEST_LIBRARIES Licensing LICENSING_INDEX)
  list(INSERT THRIFT_HANDLER_TEST_LIBRARIES ${LICENSING_INDEX} TestLicensing)
  list(REMOVE_ITEM THRIFT_HANDLER_TEST_LIBRARIES Licensing)
endif()

list(APPEND EXECUTE_TEST_LIBS L0Mgr ${LevelZero_LIBRARY})
if(ENABLE_L0)
  add_executable(L0BuildTest L0BuildTest.cpp)
  target_include_directories(L0BuildTest PRIVATE ${LevelZero_INCLUDE_DIRS})
  target_link_libraries(L0BuildTest ${LevelZero_LIBRARY} ${EXECUTE_TEST_LIBS})
  target_link_libraries(SpirvBuildTest LLVMSPIRVLib ${THRIFT_HANDLER_TEST_LIBRARIES})  # quick fix for thrift link issue
  target_link_libraries(SpirvExecuteTest LLVMSPIRVLib ${THRIFT_HANDLER_TEST_LIBRARIES})
endif()

target_link_libraries(ProfileTest ${EXECUTE_TEST_LIBS})
target_link_libraries(ResultSetTest ${EXECUTE_TEST_LIBS})
target_link_libraries(ColumnarResultsTest ${EXECUTE_TEST_LIBS})
target_link_libraries(FromTableReorderingTest ${EXECUTE_TEST_LIBS})
target_link_libraries(ResultSetBaselineRadixSortTest ${EXECUTE_TEST_LIBS})
target_link_libraries(UtilTest Utils gtest Logger Shared ${Boost_LIBRARIES})
target_link_libraries(StringTransformTest Logger Shared gtest ${Boost_LIBRARIES})
target_link_libraries(StringFunctionsTest ${EXECUTE_TEST_LIBS})
target_link_libraries(TokenCompletionHintsTest token_completion_hints gtest mapd_thrift Logger Shared ${Boost_LIBRARIES})
target_link_libraries(DumpRestoreTest ${EXECUTE_TEST_LIBS})
target_link_libraries(CodeGeneratorTest ${EXECUTE_TEST_LIBS})
target_link_libraries(ExecuteTest ${EXECUTE_TEST_LIBS})
target_link_libraries(GeospatialTest ${EXECUTE_TEST_LIBS})
target_link_libraries(RunQueryLoop ${EXECUTE_TEST_LIBS} bcrypt)
target_link_libraries(ImportExportTest gtest ${EXECUTE_TEST_LIBS})
target_link_libraries(ArrowCsvForeignStorageTest gtest ${EXECUTE_TEST_LIBS})
target_link_libraries(AlterTableDdlTest ${THRIFT_HANDLER_TEST_LIBRARIES})
target_link_libraries(UpdelStorageTest gtest ${EXECUTE_TEST_LIBS})
target_link_libraries(StorageTest gtest ${EXECUTE_TEST_LIBS})
target_link_libraries(ComputeMetadataTest ${THRIFT_HANDLER_TEST_LIBRARIES})
target_link_libraries(TableFunctionsTest ${EXECUTE_TEST_LIBS})
target_link_libraries(ArrayTest ${EXECUTE_TEST_LIBS})
target_link_libraries(ParallelExecutorsTest ${EXECUTE_TEST_LIBS})
target_link_libraries(GroupByTest ${EXECUTE_TEST_LIBS})
target_link_libraries(MigrationMgrTest ${EXECUTE_TEST_LIBS})
target_link_libraries(BumpAllocatorTest ${EXECUTE_TEST_LIBS})
target_link_libraries(SpecialCharsTest ${EXECUTE_TEST_LIBS})
target_link_libraries(UpdateMetadataTest ${EXECUTE_TEST_LIBS})
target_link_libraries(StoragePerfTest gtest ${EXECUTE_TEST_LIBS})
target_link_libraries(TopKTest ${EXECUTE_TEST_LIBS})
target_link_libraries(OmniSQLCommandTest gtest ${Boost_LIBRARIES} mapd_thrift)
if(ENABLE_KRB5)
	target_link_libraries(OmniSQLCommandTest krb5_gss Shared)
endif()
target_link_libraries(OmniSQLUtilitiesTest gtest ${Boost_LIBRARIES})

if(ENABLE_FOLLY) 
  target_link_libraries(StringDictionaryTest gtest mapd_thrift StringDictionary Logger Utils ${CMAKE_DL_LIBS} ${Folly_LIBRARIES} ${Boost_LIBRARIES} ${ZLIB_LIBRARIES} ${PROFILER_LIBS})
else()
  target_link_libraries(StringDictionaryTest gtest mapd_thrift StringDictionary Logger Utils ${CMAKE_DL_LIBS} ${Boost_LIBRARIES} ${ZLIB_LIBRARIES} ${PROFILER_LIBS})
endif()

target_link_libraries(GeoTypesTest gtest Logger Shared Geospatial ${CMAKE_DL_LIBS})
target_link_libraries(OverlapsJoinTest ${EXECUTE_TEST_LIBS})

target_link_libraries(CorrelatedSubqueryTest ${EXECUTE_TEST_LIBS})
target_link_libraries(CtasIntegrationTest gtest Logger Shared mapd_thrift ThriftClient ${LLVM_LINKER_FLAGS})
target_link_libraries(DateTimeUtilsTest gtest Logger Shared ${LLVM_LINKER_FLAGS})
target_link_libraries(CalciteOptimizeTest ${EXECUTE_TEST_LIBS})
target_link_libraries(JoinHashTableTest ${EXECUTE_TEST_LIBS})
target_link_libraries(CachedHashTableTest ${EXECUTE_TEST_LIBS})
target_link_libraries(RuntimeInterruptTest ${EXECUTE_TEST_LIBS})
target_link_libraries(UtilTest OSDependent)
target_link_libraries(EncoderTest gtest ${Arrow_LIBRARIES} Catalog ImportExport Geospatial Parser DataMgr Logger)
target_link_libraries(CommandLineTest gtest Logger Shared ${Boost_LIBRARIES})
#Requires thrift_handler for DBHandler test fixture
target_link_libraries(CtasUpdateTest ${THRIFT_HANDLER_TEST_LIBRARIES})
target_link_libraries(DBObjectPrivilegesTest ${THRIFT_HANDLER_TEST_LIBRARIES})
target_link_libraries(ForeignServerDdlTest ${THRIFT_HANDLER_TEST_LIBRARIES})
target_link_libraries(CatalogMigrationTest ${THRIFT_HANDLER_TEST_LIBRARIES})
target_link_libraries(CreateAndDropTableDdlTest ${THRIFT_HANDLER_TEST_LIBRARIES})
target_link_libraries(ShowCommandsDdlTest ${THRIFT_HANDLER_TEST_LIBRARIES})
target_link_libraries(ForeignTableDmlTest ${THRIFT_HANDLER_TEST_LIBRARIES})
target_link_libraries(DashboardAndCustomExpressionTest ${THRIFT_HANDLER_TEST_LIBRARIES})
target_link_libraries(FileMgrTest ${THRIFT_HANDLER_TEST_LIBRARIES})
target_link_libraries(FilePathWhitelistTest ${THRIFT_HANDLER_TEST_LIBRARIES})
target_link_libraries(SQLHintTest ${EXECUTE_TEST_LIBS})
target_link_libraries(QuantileCpuTest gtest ${MAPD_LIBRARIES})
target_link_libraries(ForeignStorageCacheTest gtest ${MAPD_LIBRARIES})
target_link_libraries(PersistentStorageTest gtest ${MAPD_LIBRARIES})
target_link_libraries(ShardedTableEpochConsistencyTest ${THRIFT_HANDLER_TEST_LIBRARIES})
target_link_libraries(DiskCacheQueryTest ${THRIFT_HANDLER_TEST_LIBRARIES})
target_link_libraries(CachingFileMgrTest gtest DataMgr ${Boost_LIBRARIES})
target_link_libraries(LoadTableTest ${THRIFT_HANDLER_TEST_LIBRARIES})
target_link_libraries(JSONTest gtest Logger Shared)

if(NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
  target_link_libraries(UdfTest gtest ${EXECUTE_TEST_LIBS})
endif()

target_link_libraries(TableUpdateDeleteBenchmark benchmark ${EXECUTE_TEST_LIBS})
if(ENABLE_CUDA)
  target_link_libraries(GpuSharedMemoryTest ${EXECUTE_TEST_LIBS})
elseif(ENABLE_DBE)
  target_link_libraries(EmbeddedDatabaseTest gtest DBEngine)
endif()

if(("${MAPD_EDITION_LOWER}" STREQUAL "ee") AND ENABLE_AWS_S3)
  target_link_libraries(UserMappingDdlTest ${THRIFT_HANDLER_TEST_LIBRARIES})
  target_link_libraries(PkiEncryptorTest gtest Logger Shared)
  target_link_libraries(S3SelectClientTest ${EXECUTE_TEST_LIBS})
endif()

set(TEST_ARGS "--gtest_output=xml:../")
add_test(DumpRestoreTest DumpRestoreTest ${TEST_ARGS})
add_test(UpdelStorageTest UpdelStorageTest ${TEST_ARGS})
add_test(ImportExportTest ImportExportTest ${TEST_ARGS})
add_test(ArrowCsvForeignStorageTest ArrowCsvForeignStorageTest ${TEST_ARGS})
add_test(AlterTableDdlTest AlterTableDdlTest ${TEST_ARGS})
add_test(UtilTest UtilTest ${TEST_ARGS})
add_test(ExecuteTest ExecuteTest ${TEST_ARGS})
add_test(NAME ExecuteTestTemporaryTables COMMAND ExecuteTest ${TEST_ARGS} "--use-temporary-tables")
add_test(GeospatialTest GeospatialTest ${TEST_ARGS})
add_test(CodeGeneratorTest CodeGeneratorTest ${TEST_ARGS})
add_test(ResultSetTest ResultSetTest ${TEST_ARGS})
add_test(ColumnarResultsTest ColumnarResultsTest ${TEST_ARGS})
add_test(FromTableReorderingTest FromTableReorderingTest ${TEST_ARGS})
add_test(JoinHashTableTest JoinHashTableTest ${TEST_ARGS})
add_test(CachedHashTableTest CachedHashTableTest ${TEST_ARGS})
add_test(ResultSetBaselineRadixSortTest ResultSetBaselineRadixSortTest ${TEST_ARGS})
add_test(RunQueryLoop RunQueryLoop ${TEST_ARGS})
add_test(StringDictionaryTest StringDictionaryTest ${TEST_ARGS})
add_test(NAME StringDictionaryHashTest COMMAND StringDictionaryTest ${TEST_ARGS} "--enable-string-dict-hash-cache")
add_test(StringTransformTest StringTransformTest ${TEST_ARGS})
add_test(StringFunctionsTest StringFunctionsTest ${TEST_ARGS})
add_test(StorageTest StorageTest ${TEST_ARGS})
add_test(ComputeMetadataTest ComputeMetadataTest ${TEST_ARGS})
add_test(BumpAllocatorTest BumpAllocatorTest ${TEST_ARGS})
add_test(SpecialCharsTest SpecialCharsTest ${TEST_ARGS})
add_test(TableFunctionsTest TableFunctionsTest ${TEST_ARGS})
add_test(ArrayTest ArrayTest ${TEST_ARGS})
add_test(ParallelExecutorsTest ParallelExecutorsTest ${TEST_ARGS})
add_test(GroupByTest GroupByTest ${TEST_ARGS})
add_test(MigrationMgrTest MigrationMgrTest ${TEST_ARGS})
add_test(StoragePerfTest StoragePerfTest ${TEST_ARGS})
add_test(TopKTest TopKTest ${TEST_ARGS})
add_test(TokenCompletionHintsTest TokenCompletionHintsTest ${TEST_ARGS})
add_test(OmniSQLCommandTest OmniSQLCommandTest ${TEST_ARGS})
add_test(OmniSQLUtilitiesTest OmniSQLUtilitiesTest ${TEST_ARGS})
add_test(DBObjectPrivilegesTest DBObjectPrivilegesTest ${TEST_ARGS})
add_test(GeoTypesTest GeoTypesTest ${TEST_ARGS})
add_test(OverlapsJoinTest OverlapsJoinTest ${TEST_ARGS})
add_test(CtasUpdateTest CtasUpdateTest ${TEST_ARGS})
add_test(CorrelatedSubqueryTest CorrelatedSubqueryTest ${TEST_ARGS})
add_test(DateTimeUtilsTest DateTimeUtilsTest ${TEST_ARGS})
add_test(UpdateMetadataTest UpdateMetadataTest ${TEST_ARGS})
add_test(CalciteOptimizeTest CalciteOptimizeTest ${TEST_ARGS})
add_test(JoinHashTableTest JoinHashTableTest ${TEST_ARGS})
add_tesT(RuntimeInterruptTest RuntimeInterruptTest ${TEST_ARGS})
add_test(CommandLineTest CommandLineTest ${TEST_ARGS})
add_test(ForeignServerDdlTest ForeignServerDdlTest ${TEST_ARGS})
add_test(ShowCommandsDdlTest ShowCommandsDdlTest ${TEST_ARGS})
add_test(CatalogMigrationTest CatalogMigrationTest ${TEST_ARGS})
add_test(CreateAndDropTableDdlTest CreateAndDropTableDdlTest ${TEST_ARGS})
add_test(ForeignTableDmlTest ForeignTableDmlTest ${TEST_ARGS})
add_test(DashboardAndCustomExpressionTest DashboardAndCustomExpressionTest ${TEST_ARGS})
add_test(FileMgrTest FileMgrTest ${TEST_ARGS})
add_test(FilePathWhitelistTest FilePathWhitelistTest ${TEST_ARGS})
add_test(EncoderTest EncoderTest ${TEST_ARGS})
add_test(SQLHintTest SQLHintTest ${TEST_ARGS})
add_test(ForeignStorageCacheTest ForeignStorageCacheTest ${TEST_ARGS})
add_test(PersistentStorageTest PersistentStorageTest ${TEST_ARGS})
add_test(ShardedTableEpochConsistencyTest ShardedTableEpochConsistencyTest ${TEST_ARGS})
add_test(DiskCacheQueryTest DiskCacheQueryTest ${TEST_ARGS})
add_test(CachingFileMgrTest CachingFileMgrTest ${TEST_ARGS})
add_test(LoadTableTest LoadTableTest ${TEST_ARGS})
add_test(JSONTest JSONTest ${TEST_ARGS})

if(ENABLE_CUDA)
  add_test(GpuSharedMemoryTest GpuSharedMemoryTest ${TEST_ARGS})
elseif(ENABLE_DBE)
  add_test(EmbeddedDatabaseTest EmbeddedDatabaseTest ${TEST_ARGS})
endif()

if(ENABLE_L0)
  add_test(L0BuildTest L0BuildTest ${TEST_ARGS})
  add_test(SpirvBuildTest SpirvBuildTest ${TEST_ARGS})
  add_test(SpirvExecuteTest SpirvExecuteTest ${TEST_ARGS})
endif()

if(NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
  add_test(UdfTest UdfTest ${TEST_ARGS})
endif()

if(("${MAPD_EDITION_LOWER}" STREQUAL "ee") AND ENABLE_AWS_S3)
  add_test(UserMappingDdlTest UserMappingDdlTest ${TEST_ARGS})
  add_test(PkiEncryptorTest PkiEncryptorTest ${TEST_ARGS})
  add_test(S3SelectClientTest S3SelectClientTest ${TEST_ARGS})
endif()

find_package(fmt REQUIRED)
add_executable(ArrowIpcIntegrationTest ArrowIpcIntegrationTest.cpp)
target_link_libraries(ArrowIpcIntegrationTest gtest fmt::fmt ThriftClient mapd_thrift Logger Shared ${CUDA_LIBRARIES} ${Arrow_LIBRARIES})
add_test(ArrowIpcIntegrationTest ArrowIpcIntegrationTest ${TEST_ARGS})

#parse s3 credentials
file(READ aws/s3client.conf S3CLIENT_CONF)
if("${S3CLIENT_CONF}" MATCHES "AWS_ACCESS_KEY_ID=([^\n]+)")
  set(AWS_ACCESS_KEY_ID "${CMAKE_MATCH_1}")
endif()
if("${S3CLIENT_CONF}" MATCHES "AWS_SECRET_ACCESS_KEY=([^\n]+)")
  set(AWS_SECRET_ACCESS_KEY "${CMAKE_MATCH_1}")
endif()
if("${S3CLIENT_CONF}" MATCHES "AWS_REGION=([^\n]+)")
  set(AWS_REGION "${CMAKE_MATCH_1}")
endif()

set(TEST_PROGRAMS 
  DumpRestoreTest
  ExecuteTest
  GeospatialTest
  CodeGeneratorTest
  ResultSetTest
  ColumnarResultsTest
  FromTableReorderingTest
  ResultSetBaselineRadixSortTest
  StorageTest
  ImportExportTest
  ArrowCsvForeignStorageTest
  AlterTableDdlTest
  UpdelStorageTest
  ComputeMetadataTest
  BumpAllocatorTest
  SpecialCharsTest
  TableFunctionsTest
  ArrayTest
  GroupByTest
  ParallelExecutorsTest
  MigrationMgrTest
  TopKTest
  TokenCompletionHintsTest
  OmniSQLCommandTest
  OmniSQLUtilitiesTest
  DBObjectPrivilegesTest
  GeoTypesTest
  OverlapsJoinTest
  CtasUpdateTest
  CorrelatedSubqueryTest
  DateTimeUtilsTest
  UpdateMetadataTest
  CalciteOptimizeTest
  JoinHashTableTest
  CachedHashTableTest
  RuntimeInterruptTest
  StringFunctionsTest
  StringDictionaryTest
  CommandLineTest
  ForeignServerDdlTest
  ShowCommandsDdlTest
  CatalogMigrationTest
  CreateAndDropTableDdlTest
  ForeignTableDmlTest
  DashboardAndCustomExpressionTest
  FileMgrTest
  FilePathWhitelistTest
  EncoderTest
  SQLHintTest
  ForeignStorageCacheTest
  PersistentStorageTest
  ShardedTableEpochConsistencyTest
  DiskCacheQueryTest
  CachingFileMgrTest
  LoadTableTest
  JSONTest
)

if(ENABLE_CUDA)
  list(APPEND TEST_PROGRAMS GpuSharedMemoryTest)
elseif(ENABLE_DBE)
  list(APPEND TEST_PROGRAMS EmbeddedDatabaseTest)
endif()

if(NOT ${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
  list(APPEND TEST_PROGRAMS UdfTest)
endif()

if(("${MAPD_EDITION_LOWER}" STREQUAL "ee") AND ENABLE_AWS_S3)
  list(APPEND TEST_PROGRAMS UserMappingDdlTest)
  list(APPEND TEST_PROGRAMS PkiEncryptorTest)
endif()

set(SANITY_TESTS ${TEST_PROGRAMS})
list(APPEND SANITY_TESTS ExecuteTestTemporaryTables)
list(APPEND SANITY_TESTS StringDictionaryHashTest)

set_tests_properties(${SANITY_TESTS} PROPERTIES LABELS "sanity")

add_custom_target(sanity_tests
    COMMAND mkdir -p ${TEST_BASE_PATH}
    COMMAND initdb -f ${TEST_BASE_PATH}
    COMMAND env AWS_REGION=${AWS_REGION}
                AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
                AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
                ${CMAKE_CTEST_COMMAND} --verbose --label-regex sanity
    DEPENDS ${TEST_PROGRAMS}
    USES_TERMINAL)

add_custom_target(all_tests
    COMMAND mkdir -p ${TEST_BASE_PATH}
    COMMAND initdb -f ${TEST_BASE_PATH}
    COMMAND env AWS_REGION=${AWS_REGION}
                AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
                AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
                ${CMAKE_CTEST_COMMAND} --verbose
    DEPENDS ${TEST_PROGRAMS} ProfileTest UtilTest RunQueryLoop StringDictionaryTest StringTransformTest StoragePerfTest
    USES_TERMINAL)

add_custom_target(storage_perf_tests
    COMMAND mkdir -p ${TEST_BASE_PATH}
    COMMAND initdb -f ${TEST_BASE_PATH}
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose --tests-regex "\"(StoragePerfTest)\""
    DEPENDS StoragePerfTest)

add_custom_target(topk_tests
    COMMAND mkdir -p ${TEST_BASE_PATH}
    COMMAND initdb -f ${TEST_BASE_PATH}
    COMMAND ${CMAKE_CTEST_COMMAND} --verbose --tests-regex "\"(TopKTest)\""
    DEPENDS TopKTest)

if(ENABLE_DBE)
  set(DBE_TEST_PROGRAMS EmbeddedDatabaseTest)
  set_tests_properties(${DBE_TEST_PROGRAMS} PROPERTIES LABELS "embedded")
  add_custom_target(dbe_all_tests
    COMMAND mkdir -p ${TEST_BASE_PATH}
    COMMAND env AWS_REGION=${AWS_REGION}
                AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
                AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
                ${CMAKE_CTEST_COMMAND} --verbose --label-regex embedded
    DEPENDS DBEngine ${DBE_TEST_PROGRAMS}
    USES_TERMINAL)
  add_dependencies(dbe_all_tests dbe4py-test)
  add_dependencies(all_tests dbe4py-test)
endif()
