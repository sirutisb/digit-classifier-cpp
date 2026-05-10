find_package(OpenMP REQUIRED)

# add_compile_options(-g)

add_executable(prog
    main.cpp
    loader.cpp
    progress.cpp
    network.cpp
    model.cpp
)

target_include_directories(prog PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")


target_link_libraries(prog PRIVATE OpenMP::OpenMP_CXX)
