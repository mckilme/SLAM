# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_SOURCE_DIR = /home/mckilme/slambook2/MYNT-EYE-D-SDK-master/samples

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/mckilme/slambook2/MYNT-EYE-D-SDK-master/samples/_build

# Include any dependencies generated for this target.
include CMakeFiles/get_stereo_image.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/get_stereo_image.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/get_stereo_image.dir/flags.make

CMakeFiles/get_stereo_image.dir/src/get_stereo_image.cc.o: CMakeFiles/get_stereo_image.dir/flags.make
CMakeFiles/get_stereo_image.dir/src/get_stereo_image.cc.o: ../src/get_stereo_image.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mckilme/slambook2/MYNT-EYE-D-SDK-master/samples/_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/get_stereo_image.dir/src/get_stereo_image.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/get_stereo_image.dir/src/get_stereo_image.cc.o -c /home/mckilme/slambook2/MYNT-EYE-D-SDK-master/samples/src/get_stereo_image.cc

CMakeFiles/get_stereo_image.dir/src/get_stereo_image.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/get_stereo_image.dir/src/get_stereo_image.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mckilme/slambook2/MYNT-EYE-D-SDK-master/samples/src/get_stereo_image.cc > CMakeFiles/get_stereo_image.dir/src/get_stereo_image.cc.i

CMakeFiles/get_stereo_image.dir/src/get_stereo_image.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/get_stereo_image.dir/src/get_stereo_image.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mckilme/slambook2/MYNT-EYE-D-SDK-master/samples/src/get_stereo_image.cc -o CMakeFiles/get_stereo_image.dir/src/get_stereo_image.cc.s

CMakeFiles/get_stereo_image.dir/src/get_stereo_image.cc.o.requires:

.PHONY : CMakeFiles/get_stereo_image.dir/src/get_stereo_image.cc.o.requires

CMakeFiles/get_stereo_image.dir/src/get_stereo_image.cc.o.provides: CMakeFiles/get_stereo_image.dir/src/get_stereo_image.cc.o.requires
	$(MAKE) -f CMakeFiles/get_stereo_image.dir/build.make CMakeFiles/get_stereo_image.dir/src/get_stereo_image.cc.o.provides.build
.PHONY : CMakeFiles/get_stereo_image.dir/src/get_stereo_image.cc.o.provides

CMakeFiles/get_stereo_image.dir/src/get_stereo_image.cc.o.provides.build: CMakeFiles/get_stereo_image.dir/src/get_stereo_image.cc.o


CMakeFiles/get_stereo_image.dir/src/util/cam_utils.cc.o: CMakeFiles/get_stereo_image.dir/flags.make
CMakeFiles/get_stereo_image.dir/src/util/cam_utils.cc.o: ../src/util/cam_utils.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mckilme/slambook2/MYNT-EYE-D-SDK-master/samples/_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/get_stereo_image.dir/src/util/cam_utils.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/get_stereo_image.dir/src/util/cam_utils.cc.o -c /home/mckilme/slambook2/MYNT-EYE-D-SDK-master/samples/src/util/cam_utils.cc

CMakeFiles/get_stereo_image.dir/src/util/cam_utils.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/get_stereo_image.dir/src/util/cam_utils.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mckilme/slambook2/MYNT-EYE-D-SDK-master/samples/src/util/cam_utils.cc > CMakeFiles/get_stereo_image.dir/src/util/cam_utils.cc.i

CMakeFiles/get_stereo_image.dir/src/util/cam_utils.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/get_stereo_image.dir/src/util/cam_utils.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mckilme/slambook2/MYNT-EYE-D-SDK-master/samples/src/util/cam_utils.cc -o CMakeFiles/get_stereo_image.dir/src/util/cam_utils.cc.s

CMakeFiles/get_stereo_image.dir/src/util/cam_utils.cc.o.requires:

.PHONY : CMakeFiles/get_stereo_image.dir/src/util/cam_utils.cc.o.requires

CMakeFiles/get_stereo_image.dir/src/util/cam_utils.cc.o.provides: CMakeFiles/get_stereo_image.dir/src/util/cam_utils.cc.o.requires
	$(MAKE) -f CMakeFiles/get_stereo_image.dir/build.make CMakeFiles/get_stereo_image.dir/src/util/cam_utils.cc.o.provides.build
.PHONY : CMakeFiles/get_stereo_image.dir/src/util/cam_utils.cc.o.provides

CMakeFiles/get_stereo_image.dir/src/util/cam_utils.cc.o.provides.build: CMakeFiles/get_stereo_image.dir/src/util/cam_utils.cc.o


CMakeFiles/get_stereo_image.dir/src/util/cv_painter.cc.o: CMakeFiles/get_stereo_image.dir/flags.make
CMakeFiles/get_stereo_image.dir/src/util/cv_painter.cc.o: ../src/util/cv_painter.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/mckilme/slambook2/MYNT-EYE-D-SDK-master/samples/_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/get_stereo_image.dir/src/util/cv_painter.cc.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/get_stereo_image.dir/src/util/cv_painter.cc.o -c /home/mckilme/slambook2/MYNT-EYE-D-SDK-master/samples/src/util/cv_painter.cc

CMakeFiles/get_stereo_image.dir/src/util/cv_painter.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/get_stereo_image.dir/src/util/cv_painter.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/mckilme/slambook2/MYNT-EYE-D-SDK-master/samples/src/util/cv_painter.cc > CMakeFiles/get_stereo_image.dir/src/util/cv_painter.cc.i

CMakeFiles/get_stereo_image.dir/src/util/cv_painter.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/get_stereo_image.dir/src/util/cv_painter.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/mckilme/slambook2/MYNT-EYE-D-SDK-master/samples/src/util/cv_painter.cc -o CMakeFiles/get_stereo_image.dir/src/util/cv_painter.cc.s

CMakeFiles/get_stereo_image.dir/src/util/cv_painter.cc.o.requires:

.PHONY : CMakeFiles/get_stereo_image.dir/src/util/cv_painter.cc.o.requires

CMakeFiles/get_stereo_image.dir/src/util/cv_painter.cc.o.provides: CMakeFiles/get_stereo_image.dir/src/util/cv_painter.cc.o.requires
	$(MAKE) -f CMakeFiles/get_stereo_image.dir/build.make CMakeFiles/get_stereo_image.dir/src/util/cv_painter.cc.o.provides.build
.PHONY : CMakeFiles/get_stereo_image.dir/src/util/cv_painter.cc.o.provides

CMakeFiles/get_stereo_image.dir/src/util/cv_painter.cc.o.provides.build: CMakeFiles/get_stereo_image.dir/src/util/cv_painter.cc.o


# Object files for target get_stereo_image
get_stereo_image_OBJECTS = \
"CMakeFiles/get_stereo_image.dir/src/get_stereo_image.cc.o" \
"CMakeFiles/get_stereo_image.dir/src/util/cam_utils.cc.o" \
"CMakeFiles/get_stereo_image.dir/src/util/cv_painter.cc.o"

# External object files for target get_stereo_image
get_stereo_image_EXTERNAL_OBJECTS =

../_output/bin/get_stereo_image: CMakeFiles/get_stereo_image.dir/src/get_stereo_image.cc.o
../_output/bin/get_stereo_image: CMakeFiles/get_stereo_image.dir/src/util/cam_utils.cc.o
../_output/bin/get_stereo_image: CMakeFiles/get_stereo_image.dir/src/util/cv_painter.cc.o
../_output/bin/get_stereo_image: CMakeFiles/get_stereo_image.dir/build.make
../_output/bin/get_stereo_image: /usr/local/lib/libmynteye_depth.so.1.8.0
../_output/bin/get_stereo_image: /usr/local/lib/libopencv_dnn.so.3.4.3
../_output/bin/get_stereo_image: /usr/local/lib/libopencv_ml.so.3.4.3
../_output/bin/get_stereo_image: /usr/local/lib/libopencv_objdetect.so.3.4.3
../_output/bin/get_stereo_image: /usr/local/lib/libopencv_shape.so.3.4.3
../_output/bin/get_stereo_image: /usr/local/lib/libopencv_stitching.so.3.4.3
../_output/bin/get_stereo_image: /usr/local/lib/libopencv_superres.so.3.4.3
../_output/bin/get_stereo_image: /usr/local/lib/libopencv_videostab.so.3.4.3
../_output/bin/get_stereo_image: /usr/local/lib/libopencv_viz.so.3.4.3
../_output/bin/get_stereo_image: /usr/local/lib/libopencv_calib3d.so.3.4.3
../_output/bin/get_stereo_image: /usr/local/lib/libopencv_features2d.so.3.4.3
../_output/bin/get_stereo_image: /usr/local/lib/libopencv_flann.so.3.4.3
../_output/bin/get_stereo_image: /usr/local/lib/libopencv_highgui.so.3.4.3
../_output/bin/get_stereo_image: /usr/local/lib/libopencv_photo.so.3.4.3
../_output/bin/get_stereo_image: /usr/local/lib/libopencv_video.so.3.4.3
../_output/bin/get_stereo_image: /usr/local/lib/libopencv_videoio.so.3.4.3
../_output/bin/get_stereo_image: /usr/local/lib/libopencv_imgcodecs.so.3.4.3
../_output/bin/get_stereo_image: /usr/local/lib/libopencv_imgproc.so.3.4.3
../_output/bin/get_stereo_image: /usr/local/lib/libopencv_core.so.3.4.3
../_output/bin/get_stereo_image: /home/mckilme/slambook2/MYNT-EYE-D-SDK-master/3rdparty/eSPDI/linux/x64/libeSPDI.so
../_output/bin/get_stereo_image: /usr/lib/x86_64-linux-gnu/libjpeg.so
../_output/bin/get_stereo_image: CMakeFiles/get_stereo_image.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/mckilme/slambook2/MYNT-EYE-D-SDK-master/samples/_build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable ../_output/bin/get_stereo_image"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/get_stereo_image.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/get_stereo_image.dir/build: ../_output/bin/get_stereo_image

.PHONY : CMakeFiles/get_stereo_image.dir/build

CMakeFiles/get_stereo_image.dir/requires: CMakeFiles/get_stereo_image.dir/src/get_stereo_image.cc.o.requires
CMakeFiles/get_stereo_image.dir/requires: CMakeFiles/get_stereo_image.dir/src/util/cam_utils.cc.o.requires
CMakeFiles/get_stereo_image.dir/requires: CMakeFiles/get_stereo_image.dir/src/util/cv_painter.cc.o.requires

.PHONY : CMakeFiles/get_stereo_image.dir/requires

CMakeFiles/get_stereo_image.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/get_stereo_image.dir/cmake_clean.cmake
.PHONY : CMakeFiles/get_stereo_image.dir/clean

CMakeFiles/get_stereo_image.dir/depend:
	cd /home/mckilme/slambook2/MYNT-EYE-D-SDK-master/samples/_build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/mckilme/slambook2/MYNT-EYE-D-SDK-master/samples /home/mckilme/slambook2/MYNT-EYE-D-SDK-master/samples /home/mckilme/slambook2/MYNT-EYE-D-SDK-master/samples/_build /home/mckilme/slambook2/MYNT-EYE-D-SDK-master/samples/_build /home/mckilme/slambook2/MYNT-EYE-D-SDK-master/samples/_build/CMakeFiles/get_stereo_image.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/get_stereo_image.dir/depend

