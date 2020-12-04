project(isodisreg)
cmake_minimum_required(VERSION 3.10)


add_subdirectory(lib/pybind11)
pybind11_add_module(_isodisreg src/isodisreg/isocdf_seq.cpp)
