add_executable(prog
    main.cpp
    loader.cpp
    progress.cpp
)

target_include_directories(prog PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")