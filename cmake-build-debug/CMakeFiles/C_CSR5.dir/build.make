# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.17.3/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.17.3/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/lianhaocheng/CLionProjects/Benchmark_SpMV_using_CSR5_avx2_C

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/lianhaocheng/CLionProjects/Benchmark_SpMV_using_CSR5_avx2_C/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/C_CSR5.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/C_CSR5.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/C_CSR5.dir/flags.make

CMakeFiles/C_CSR5.dir/main.c.o: CMakeFiles/C_CSR5.dir/flags.make
CMakeFiles/C_CSR5.dir/main.c.o: ../main.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/lianhaocheng/CLionProjects/Benchmark_SpMV_using_CSR5_avx2_C/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/C_CSR5.dir/main.c.o"
	gcc-9 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/C_CSR5.dir/main.c.o   -c /Users/lianhaocheng/CLionProjects/Benchmark_SpMV_using_CSR5_avx2_C/main.c

CMakeFiles/C_CSR5.dir/main.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/C_CSR5.dir/main.c.i"
	gcc-9 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /Users/lianhaocheng/CLionProjects/Benchmark_SpMV_using_CSR5_avx2_C/main.c > CMakeFiles/C_CSR5.dir/main.c.i

CMakeFiles/C_CSR5.dir/main.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/C_CSR5.dir/main.c.s"
	gcc-9 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /Users/lianhaocheng/CLionProjects/Benchmark_SpMV_using_CSR5_avx2_C/main.c -o CMakeFiles/C_CSR5.dir/main.c.s

# Object files for target C_CSR5
C_CSR5_OBJECTS = \
"CMakeFiles/C_CSR5.dir/main.c.o"

# External object files for target C_CSR5
C_CSR5_EXTERNAL_OBJECTS =

C_CSR5: CMakeFiles/C_CSR5.dir/main.c.o
C_CSR5: CMakeFiles/C_CSR5.dir/build.make
C_CSR5: CMakeFiles/C_CSR5.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/lianhaocheng/CLionProjects/Benchmark_SpMV_using_CSR5_avx2_C/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking C executable C_CSR5"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/C_CSR5.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/C_CSR5.dir/build: C_CSR5

.PHONY : CMakeFiles/C_CSR5.dir/build

CMakeFiles/C_CSR5.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/C_CSR5.dir/cmake_clean.cmake
.PHONY : CMakeFiles/C_CSR5.dir/clean

CMakeFiles/C_CSR5.dir/depend:
	cd /Users/lianhaocheng/CLionProjects/Benchmark_SpMV_using_CSR5_avx2_C/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/lianhaocheng/CLionProjects/Benchmark_SpMV_using_CSR5_avx2_C /Users/lianhaocheng/CLionProjects/Benchmark_SpMV_using_CSR5_avx2_C /Users/lianhaocheng/CLionProjects/Benchmark_SpMV_using_CSR5_avx2_C/cmake-build-debug /Users/lianhaocheng/CLionProjects/Benchmark_SpMV_using_CSR5_avx2_C/cmake-build-debug /Users/lianhaocheng/CLionProjects/Benchmark_SpMV_using_CSR5_avx2_C/cmake-build-debug/CMakeFiles/C_CSR5.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/C_CSR5.dir/depend

