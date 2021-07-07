add_library(DBEngine SHARED DBETypes.cpp DBEngine.cpp DBEngine.h DBETypes.h)

target_link_libraries(DBEngine mapd_thrift thrift_handler Catalog QueryState Parser QueryEngine DataMgr ${Arrow_LIBRARIES} ${Boost_LIBRARIES} ${CMAKE_DL_LIBS} ${TBB_LIBS} Geospatial)
install(TARGETS DBEngine DESTINATION "lib" COMPONENT "DBE")
install(FILES DBEngine.h DBETypes.h DESTINATION "Embedded" COMPONENT "include")
install(FILES Python/DBEngine.pxd DESTINATION "Embedded" COMPONENT "include")

add_executable(EmbeddedDbTest EmbeddedDbTest.cpp)
target_link_libraries(EmbeddedDbTest DBEngine)

add_executable(EmbeddedDbFSITest EmbeddedDbFSITest.cpp)
target_link_libraries(EmbeddedDbFSITest DBEngine)


set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/setup.py")
# Currently dbe.cpp is generated near dbe.pyx and not in build dir. This could be revised later
set(OUTPUT      "${CMAKE_CURRENT_SOURCE_DIR}/dbe.cpp")

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/setup.in.py" "${CMAKE_CURRENT_BINARY_DIR}/setup.py.in" @ONLY)
# cannot be done in one step, splitting in configure_file and file commands
file(GENERATE OUTPUT "${SETUP_PY}" INPUT "${CMAKE_CURRENT_BINARY_DIR}/setup.py.in")

# Adding Python targets here for sake of better development experience, to give an example
find_package(PythonInterp)
if(PYTHON_EXECUTABLE)
    set(pydeps ${SETUP_PY} ${CMAKE_CURRENT_SOURCE_DIR}/DBEngine.h ${CMAKE_CURRENT_SOURCE_DIR}/Python/dbe.pyx ${CMAKE_CURRENT_SOURCE_DIR}/Python/DBEngine.pxd)

    add_custom_target(dbe4py
        COMMAND cd ${CMAKE_CURRENT_BINARY_DIR} && ${PYTHON_EXECUTABLE} ${SETUP_PY} build_ext -g -f -I ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS DBEngine ${pydeps} mapd_java_components
    )
    add_custom_target(dbe4py-test
        COMMAND ${PYTHON_EXECUTABLE} -m pytest -v ${CMAKE_CURRENT_SOURCE_DIR}/test/test_exceptions.py
        COMMAND ${PYTHON_EXECUTABLE} -m pytest -v ${CMAKE_CURRENT_SOURCE_DIR}/test/test_readcsv.py
        COMMAND ${PYTHON_EXECUTABLE} -m pytest -v ${CMAKE_CURRENT_SOURCE_DIR}/test/test_fsi.py
        COMMAND ${PYTHON_EXECUTABLE} -m pytest -v ${CMAKE_CURRENT_SOURCE_DIR}/test/test_ra.py
        DEPENDS dbe4py
    )
    add_custom_target(dbe4py-install
        COMMAND cd ${CMAKE_CURRENT_BINARY_DIR} && ${PYTHON_EXECUTABLE} ${SETUP_PY} build_ext -g -f -I ${CMAKE_CURRENT_SOURCE_DIR} install
        DEPENDS DBEngine ${pydeps}
    )
    add_custom_target(dbe4py-devel 
        cd ${CMAKE_CURRENT_BINARY_DIR} && pip install -v -e .
        DEPENDS dbe4py)

endif()
