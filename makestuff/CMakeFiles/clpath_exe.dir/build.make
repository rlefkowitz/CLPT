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
CMAKE_COMMAND = /Applications/CMake.app/Contents/bin/cmake

# The command to remove a file.
RM = /Applications/CMake.app/Contents/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/ryanlefkowitz/Programming/OpenCL/CLPT

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/ryanlefkowitz/Programming/OpenCL/CLPT/makestuff

# Include any dependencies generated for this target.
include CMakeFiles/clpath_exe.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/clpath_exe.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/clpath_exe.dir/flags.make

CMakeFiles/clpath_exe.dir/src/main.cpp.o: CMakeFiles/clpath_exe.dir/flags.make
CMakeFiles/clpath_exe.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/ryanlefkowitz/Programming/OpenCL/CLPT/makestuff/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/clpath_exe.dir/src/main.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/clpath_exe.dir/src/main.cpp.o -c /Users/ryanlefkowitz/Programming/OpenCL/CLPT/src/main.cpp

CMakeFiles/clpath_exe.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/clpath_exe.dir/src/main.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/ryanlefkowitz/Programming/OpenCL/CLPT/src/main.cpp > CMakeFiles/clpath_exe.dir/src/main.cpp.i

CMakeFiles/clpath_exe.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/clpath_exe.dir/src/main.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/ryanlefkowitz/Programming/OpenCL/CLPT/src/main.cpp -o CMakeFiles/clpath_exe.dir/src/main.cpp.s

# Object files for target clpath_exe
clpath_exe_OBJECTS = \
"CMakeFiles/clpath_exe.dir/src/main.cpp.o"

# External object files for target clpath_exe
clpath_exe_EXTERNAL_OBJECTS =

clpath: CMakeFiles/clpath_exe.dir/src/main.cpp.o
clpath: CMakeFiles/clpath_exe.dir/build.make
clpath: libclpath.a
clpath: CMakeFiles/clpath_exe.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/ryanlefkowitz/Programming/OpenCL/CLPT/makestuff/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable clpath"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/clpath_exe.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/clpath_exe.dir/build: clpath

.PHONY : CMakeFiles/clpath_exe.dir/build

CMakeFiles/clpath_exe.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/clpath_exe.dir/cmake_clean.cmake
.PHONY : CMakeFiles/clpath_exe.dir/clean

CMakeFiles/clpath_exe.dir/depend:
	cd /Users/ryanlefkowitz/Programming/OpenCL/CLPT/makestuff && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/ryanlefkowitz/Programming/OpenCL/CLPT /Users/ryanlefkowitz/Programming/OpenCL/CLPT /Users/ryanlefkowitz/Programming/OpenCL/CLPT/makestuff /Users/ryanlefkowitz/Programming/OpenCL/CLPT/makestuff /Users/ryanlefkowitz/Programming/OpenCL/CLPT/makestuff/CMakeFiles/clpath_exe.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/clpath_exe.dir/depend
