# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


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
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/wangjiahui/Desktop/Concordia/ComputerVision/CVA2_M

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/wangjiahui/Desktop/Concordia/ComputerVision/CVA2_M/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/CVA2_M.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/CVA2_M.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/CVA2_M.dir/flags.make

CMakeFiles/CVA2_M.dir/main.cpp.o: CMakeFiles/CVA2_M.dir/flags.make
CMakeFiles/CVA2_M.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/wangjiahui/Desktop/Concordia/ComputerVision/CVA2_M/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/CVA2_M.dir/main.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CVA2_M.dir/main.cpp.o -c /Users/wangjiahui/Desktop/Concordia/ComputerVision/CVA2_M/main.cpp

CMakeFiles/CVA2_M.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CVA2_M.dir/main.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/wangjiahui/Desktop/Concordia/ComputerVision/CVA2_M/main.cpp > CMakeFiles/CVA2_M.dir/main.cpp.i

CMakeFiles/CVA2_M.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CVA2_M.dir/main.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/wangjiahui/Desktop/Concordia/ComputerVision/CVA2_M/main.cpp -o CMakeFiles/CVA2_M.dir/main.cpp.s

# Object files for target CVA2_M
CVA2_M_OBJECTS = \
"CMakeFiles/CVA2_M.dir/main.cpp.o"

# External object files for target CVA2_M
CVA2_M_EXTERNAL_OBJECTS =

CVA2_M: CMakeFiles/CVA2_M.dir/main.cpp.o
CVA2_M: CMakeFiles/CVA2_M.dir/build.make
CVA2_M: CMakeFiles/CVA2_M.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/wangjiahui/Desktop/Concordia/ComputerVision/CVA2_M/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable CVA2_M"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/CVA2_M.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/CVA2_M.dir/build: CVA2_M

.PHONY : CMakeFiles/CVA2_M.dir/build

CMakeFiles/CVA2_M.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/CVA2_M.dir/cmake_clean.cmake
.PHONY : CMakeFiles/CVA2_M.dir/clean

CMakeFiles/CVA2_M.dir/depend:
	cd /Users/wangjiahui/Desktop/Concordia/ComputerVision/CVA2_M/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/wangjiahui/Desktop/Concordia/ComputerVision/CVA2_M /Users/wangjiahui/Desktop/Concordia/ComputerVision/CVA2_M /Users/wangjiahui/Desktop/Concordia/ComputerVision/CVA2_M/cmake-build-debug /Users/wangjiahui/Desktop/Concordia/ComputerVision/CVA2_M/cmake-build-debug /Users/wangjiahui/Desktop/Concordia/ComputerVision/CVA2_M/cmake-build-debug/CMakeFiles/CVA2_M.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/CVA2_M.dir/depend

