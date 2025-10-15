set(SCC_VERSION "v3.5.0")
set(SCC_DOWNLOADS_URL "https://github.com/boyter/scc/releases/download")
set(SCC_INSTALL_DIR "${CMAKE_BINARY_DIR}/bin/scc")

file(MAKE_DIRECTORY "${SCC_INSTALL_DIR}/tmp")

if(UNIX)
    set(URL "${SCC_DOWNLOADS_URL}/${SCC_VERSION}/scc_Linux_x86_64.tar.gz")
    set(ARCHIVE_PATH "${SCC_INSTALL_DIR}/tmp/scc_Linux_x86_64.tar.gz")
    set(SCC_EXE "${SCC_INSTALL_DIR}/scc")
elseif(WIN32)
    set(URL "${SCC_DOWNLOADS_URL}/${SCC_VERSION}/scc_Windows_x86_64.zip")
    set(ARCHIVE_PATH "${SCC_INSTALL_DIR}/tmp/scc_Windows_x86_64.zip")
    set(SCC_EXE "${SCC_INSTALL_DIR}/scc.exe")
endif()

if(NOT EXISTS ${SCC_EXE})
    if(NOT EXISTS ${ARCHIVE_PATH})
        file(DOWNLOAD ${URL} ${ARCHIVE_PATH})
    endif()
    file(ARCHIVE_EXTRACT
        INPUT ${ARCHIVE_PATH}
        DESTINATION "${SCC_INSTALL_DIR}/tmp"
    )
    file(RENAME "${SCC_INSTALL_DIR}/tmp/scc" ${SCC_EXE})
    file(REMOVE_RECURSE "${SCC_INSTALL_DIR}/tmp")
endif()

execute_process(
    COMMAND ${SCC_EXE}
        "--cocomo-project-type" "custom,2.4,1.05,2.5,0.38"
        ${CMAKE_CURRENT_SOURCE_DIR}
    WORKING_DIRECTORY ${SCC_INSTALL_DIR}
    RESULT_VARIABLE scc_result
    OUTPUT_VARIABLE scc_output
    ERROR_VARIABLE scc_error
)

if(NOT scc_result EQUAL 0)
    message(WARNING ${scc_error})
else()
    message(STATUS "COCOMO statistics:\n${scc_output}")
endif()
