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

# Utility rule file for gradle-android_foo.

# Include the progress variables for this target.
include android_foo/CMakeFiles/gradle-android_foo.dir/progress.make

android_foo/CMakeFiles/gradle-android_foo:
	cd /home/ahmad/catkinJava_ws/src/android_foo && ROS_MAVEN_REPOSITORY=https://github.com/rosjava/rosjava_mvn_repo/raw/master /home/ahmad/catkinJava_ws/build/catkin_generated/env_cached.sh /home/ahmad/catkinJava_ws/src/android_foo/gradlew assembleRelease uploadArchives

gradle-android_foo: android_foo/CMakeFiles/gradle-android_foo
gradle-android_foo: android_foo/CMakeFiles/gradle-android_foo.dir/build.make

.PHONY : gradle-android_foo

# Rule to build all files generated by this target.
android_foo/CMakeFiles/gradle-android_foo.dir/build: gradle-android_foo

.PHONY : android_foo/CMakeFiles/gradle-android_foo.dir/build

android_foo/CMakeFiles/gradle-android_foo.dir/clean:
	cd /home/ahmad/catkinJava_ws/build/android_foo && $(CMAKE_COMMAND) -P CMakeFiles/gradle-android_foo.dir/cmake_clean.cmake
.PHONY : android_foo/CMakeFiles/gradle-android_foo.dir/clean

android_foo/CMakeFiles/gradle-android_foo.dir/depend:
	cd /home/ahmad/catkinJava_ws/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ahmad/catkinJava_ws/src /home/ahmad/catkinJava_ws/src/android_foo /home/ahmad/catkinJava_ws/build /home/ahmad/catkinJava_ws/build/android_foo /home/ahmad/catkinJava_ws/build/android_foo/CMakeFiles/gradle-android_foo.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : android_foo/CMakeFiles/gradle-android_foo.dir/depend

