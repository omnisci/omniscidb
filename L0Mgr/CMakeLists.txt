if(ENABLE_L0)
  set(L0_SOURCES L0Mgr.cpp L0Exception.cpp)
else()
  set(L0_SOURCES L0MgrNoL0.cpp L0ExceptionNoL0.cpp)
endif()

add_library(L0Mgr ${L0_SOURCES})
target_link_libraries(L0Mgr ${LevelZero_LIBRARY})