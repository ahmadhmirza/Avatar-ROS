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
CMAKE_SOURCE_DIR = /home/ahmad/catkinJava_ws/src

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ahmad/catkinJava_ws/build

# Utility rule file for gradle-clean-avatar_ros.

# Include the progress variables for this target.
include avatar_ros/CMakeFiles/gradle-clean-avatar_ros.dir/progress.make

avatar_ros/CMakeFiles/gradle-clean-avatar_ros:
	cd /home/ahmad/catkinJava_ws/src/avatar_ros && /home/ahmad/catkinJava_ws/build/catkin_generated/env_cached.sh /home/ahmad/catkinJava_ws/src/avatar_ros/gradlew clean

gradle-clean-avatar_ros: avatar_ros/CMakeFiles/gradle-clean-avatar_ros
gradle-clean-avatar_ros: avatar_ros/CMakeFiles/gradle-clean-avatar_ros.dir/build.make

.PHONY : gradle-clean-avatar_ros

# Rule to build all files generated by this target.
avatar_ros/CMakeFiles/gradle-clean-avatar_ros.dir/build: gradle-clean-avatar_ros

.PHONY : avatar_ros/CMakeFiles/gradle-clean-avatar_ros.dir/build

avatar_ros/CMakeFiles/gradle-clean-avatar_ros.dir/clean:
	cd /home/ahmad/catkinJava_ws/build/avatar_ros && $(CMAKE_COMMAND) -P CMakeFiles/gradle-clean-avatar_ros.dir/cmake_clean.cmake
.PHONY : avatar_ros/CMakeFiles/gradle-clean-avatar_ros.dir/clean

avatar_ros/CMakeFiles/gradle-clean-avatar_ros.dir/depend:
	cd /home/ahmad/catkinJava_ws/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ahmad/catkinJava_ws/src /home/ahmad/catkinJava_ws/src/avatar_ros /home/ahmad/catkinJava_ws/build /home/ahmad/catkinJava_ws/build/avatar_ros /home/ahmad/catkinJava_ws/build/avatar_ros/CMakeFiles/gradle-clean-avatar_ros.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : avatar_ros/CMakeFiles/gradle-clean-avatar_ros.dir/depend
