# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jwei/mydetec1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jwei/mydetec1/build

# Include any dependencies generated for this target.
include CMakeFiles/network.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/network.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/network.dir/flags.make

CMakeFiles/network.dir/network.cpp.o: CMakeFiles/network.dir/flags.make
CMakeFiles/network.dir/network.cpp.o: ../network.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jwei/mydetec1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/network.dir/network.cpp.o"
	/usr/bin/g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/network.dir/network.cpp.o -c /home/jwei/mydetec1/network.cpp

CMakeFiles/network.dir/network.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/network.dir/network.cpp.i"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jwei/mydetec1/network.cpp > CMakeFiles/network.dir/network.cpp.i

CMakeFiles/network.dir/network.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/network.dir/network.cpp.s"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jwei/mydetec1/network.cpp -o CMakeFiles/network.dir/network.cpp.s

CMakeFiles/network.dir/network.cpp.o.requires:

.PHONY : CMakeFiles/network.dir/network.cpp.o.requires

CMakeFiles/network.dir/network.cpp.o.provides: CMakeFiles/network.dir/network.cpp.o.requires
	$(MAKE) -f CMakeFiles/network.dir/build.make CMakeFiles/network.dir/network.cpp.o.provides.build
.PHONY : CMakeFiles/network.dir/network.cpp.o.provides

CMakeFiles/network.dir/network.cpp.o.provides.build: CMakeFiles/network.dir/network.cpp.o


# Object files for target network
network_OBJECTS = \
"CMakeFiles/network.dir/network.cpp.o"

# External object files for target network
network_EXTERNAL_OBJECTS =

libnetwork.so: CMakeFiles/network.dir/network.cpp.o
libnetwork.so: CMakeFiles/network.dir/build.make
libnetwork.so: /usr/lib/libblas.so.3
libnetwork.so: CMakeFiles/network.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jwei/mydetec1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libnetwork.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/network.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/network.dir/build: libnetwork.so

.PHONY : CMakeFiles/network.dir/build

CMakeFiles/network.dir/requires: CMakeFiles/network.dir/network.cpp.o.requires

.PHONY : CMakeFiles/network.dir/requires

CMakeFiles/network.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/network.dir/cmake_clean.cmake
.PHONY : CMakeFiles/network.dir/clean

CMakeFiles/network.dir/depend:
	cd /home/jwei/mydetec1/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jwei/mydetec1 /home/jwei/mydetec1 /home/jwei/mydetec1/build /home/jwei/mydetec1/build /home/jwei/mydetec1/build/CMakeFiles/network.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/network.dir/depend

