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
include CMakeFiles/libnetwork.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/libnetwork.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/libnetwork.dir/flags.make

CMakeFiles/libnetwork.dir/network.cpp.o: CMakeFiles/libnetwork.dir/flags.make
CMakeFiles/libnetwork.dir/network.cpp.o: ../network.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jwei/mydetec1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/libnetwork.dir/network.cpp.o"
	/usr/bin/g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/libnetwork.dir/network.cpp.o -c /home/jwei/mydetec1/network.cpp

CMakeFiles/libnetwork.dir/network.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/libnetwork.dir/network.cpp.i"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jwei/mydetec1/network.cpp > CMakeFiles/libnetwork.dir/network.cpp.i

CMakeFiles/libnetwork.dir/network.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/libnetwork.dir/network.cpp.s"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jwei/mydetec1/network.cpp -o CMakeFiles/libnetwork.dir/network.cpp.s

CMakeFiles/libnetwork.dir/network.cpp.o.requires:

.PHONY : CMakeFiles/libnetwork.dir/network.cpp.o.requires

CMakeFiles/libnetwork.dir/network.cpp.o.provides: CMakeFiles/libnetwork.dir/network.cpp.o.requires
	$(MAKE) -f CMakeFiles/libnetwork.dir/build.make CMakeFiles/libnetwork.dir/network.cpp.o.provides.build
.PHONY : CMakeFiles/libnetwork.dir/network.cpp.o.provides

CMakeFiles/libnetwork.dir/network.cpp.o.provides.build: CMakeFiles/libnetwork.dir/network.cpp.o


# Object files for target libnetwork
libnetwork_OBJECTS = \
"CMakeFiles/libnetwork.dir/network.cpp.o"

# External object files for target libnetwork
libnetwork_EXTERNAL_OBJECTS =

liblibnetwork.so: CMakeFiles/libnetwork.dir/network.cpp.o
liblibnetwork.so: CMakeFiles/libnetwork.dir/build.make
liblibnetwork.so: CMakeFiles/libnetwork.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jwei/mydetec1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library liblibnetwork.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/libnetwork.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/libnetwork.dir/build: liblibnetwork.so

.PHONY : CMakeFiles/libnetwork.dir/build

CMakeFiles/libnetwork.dir/requires: CMakeFiles/libnetwork.dir/network.cpp.o.requires

.PHONY : CMakeFiles/libnetwork.dir/requires

CMakeFiles/libnetwork.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/libnetwork.dir/cmake_clean.cmake
.PHONY : CMakeFiles/libnetwork.dir/clean

CMakeFiles/libnetwork.dir/depend:
	cd /home/jwei/mydetec1/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jwei/mydetec1 /home/jwei/mydetec1 /home/jwei/mydetec1/build /home/jwei/mydetec1/build /home/jwei/mydetec1/build/CMakeFiles/libnetwork.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/libnetwork.dir/depend

