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
CMAKE_SOURCE_DIR = /home/pfalez/SNNv2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/pfalez/SNNv2

# Utility rule file for libsvm_automoc.

# Include the progress variables for this target.
include dep/libsvm/CMakeFiles/libsvm_automoc.dir/progress.make

dep/libsvm/CMakeFiles/libsvm_automoc:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/pfalez/SNNv2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Automatic moc for target libsvm"
	cd /home/pfalez/SNNv2/dep/libsvm && /usr/bin/cmake -E cmake_autogen /home/pfalez/SNNv2/dep/libsvm/CMakeFiles/libsvm_automoc.dir/ ""

libsvm_automoc: dep/libsvm/CMakeFiles/libsvm_automoc
libsvm_automoc: dep/libsvm/CMakeFiles/libsvm_automoc.dir/build.make

.PHONY : libsvm_automoc

# Rule to build all files generated by this target.
dep/libsvm/CMakeFiles/libsvm_automoc.dir/build: libsvm_automoc

.PHONY : dep/libsvm/CMakeFiles/libsvm_automoc.dir/build

dep/libsvm/CMakeFiles/libsvm_automoc.dir/clean:
	cd /home/pfalez/SNNv2/dep/libsvm && $(CMAKE_COMMAND) -P CMakeFiles/libsvm_automoc.dir/cmake_clean.cmake
.PHONY : dep/libsvm/CMakeFiles/libsvm_automoc.dir/clean

dep/libsvm/CMakeFiles/libsvm_automoc.dir/depend:
	cd /home/pfalez/SNNv2 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/pfalez/SNNv2 /home/pfalez/SNNv2/dep/libsvm /home/pfalez/SNNv2 /home/pfalez/SNNv2/dep/libsvm /home/pfalez/SNNv2/dep/libsvm/CMakeFiles/libsvm_automoc.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : dep/libsvm/CMakeFiles/libsvm_automoc.dir/depend
