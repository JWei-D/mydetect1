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
include CMakeFiles/libpbox.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/libpbox.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/libpbox.dir/flags.make

CMakeFiles/libpbox.dir/pBox.cpp.o: CMakeFiles/libpbox.dir/flags.make
CMakeFiles/libpbox.dir/pBox.cpp.o: ../pBox.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/jwei/mydetec1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/libpbox.dir/pBox.cpp.o"
	/usr/bin/g++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/libpbox.dir/pBox.cpp.o -c /home/jwei/mydetec1/pBox.cpp

CMakeFiles/libpbox.dir/pBox.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/libpbox.dir/pBox.cpp.i"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/jwei/mydetec1/pBox.cpp > CMakeFiles/libpbox.dir/pBox.cpp.i

CMakeFiles/libpbox.dir/pBox.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/libpbox.dir/pBox.cpp.s"
	/usr/bin/g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/jwei/mydetec1/pBox.cpp -o CMakeFiles/libpbox.dir/pBox.cpp.s

CMakeFiles/libpbox.dir/pBox.cpp.o.requires:

.PHONY : CMakeFiles/libpbox.dir/pBox.cpp.o.requires

CMakeFiles/libpbox.dir/pBox.cpp.o.provides: CMakeFiles/libpbox.dir/pBox.cpp.o.requires
	$(MAKE) -f CMakeFiles/libpbox.dir/build.make CMakeFiles/libpbox.dir/pBox.cpp.o.provides.build
.PHONY : CMakeFiles/libpbox.dir/pBox.cpp.o.provides

CMakeFiles/libpbox.dir/pBox.cpp.o.provides.build: CMakeFiles/libpbox.dir/pBox.cpp.o


# Object files for target libpbox
libpbox_OBJECTS = \
"CMakeFiles/libpbox.dir/pBox.cpp.o"

# External object files for target libpbox
libpbox_EXTERNAL_OBJECTS =

liblibpbox.so: CMakeFiles/libpbox.dir/pBox.cpp.o
liblibpbox.so: CMakeFiles/libpbox.dir/build.make
liblibpbox.so: CMakeFiles/libpbox.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/jwei/mydetec1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library liblibpbox.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/libpbox.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/libpbox.dir/build: liblibpbox.so

.PHONY : CMakeFiles/libpbox.dir/build

CMakeFiles/libpbox.dir/requires: CMakeFiles/libpbox.dir/pBox.cpp.o.requires

.PHONY : CMakeFiles/libpbox.dir/requires

CMakeFiles/libpbox.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/libpbox.dir/cmake_clean.cmake
.PHONY : CMakeFiles/libpbox.dir/clean

CMakeFiles/libpbox.dir/depend:
	cd /home/jwei/mydetec1/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jwei/mydetec1 /home/jwei/mydetec1 /home/jwei/mydetec1/build /home/jwei/mydetec1/build /home/jwei/mydetec1/build/CMakeFiles/libpbox.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/libpbox.dir/depend

