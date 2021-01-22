find_package(Boost COMPONENTS program_options filesystem regex system REQUIRED QUIET)

include_directories(${CMAKE_SOURCE_DIR}/ThirdParty/linenoise)

add_executable(omnisql omnisql.cpp)

target_link_libraries(omnisql mapd_thrift Logger Shared ThriftClient linenoise ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} ${PROFILER_LIBS} ${TBB_LIBS} ${Thrift_LIBRARIES})
if(ENABLE_KRB5)
  target_link_libraries(omnisql krb5_gss)
endif()

install(TARGETS omnisql DESTINATION bin COMPONENT "exe")
