#ifndef _PROCESS_MOTION_GRID_V5_H
#define _PROCESS_MOTION_GRID_V5_H

#include <filesystem>
#include <iostream>
#include "Process.h"
#include "NumpyReader.h"
/**
 * 	MotionGridV5 - GO CHECK PYTHON CODE IN /src/process/OpticalFlowProcess IN ORDER TO GENERATE MG
 */
namespace process
{

	/**
	 * @brief MotionGridV5 takes two frame differences from each three consiqutive frames and seperates them into +ve and -va frames (where the -ve is taken as absolute value). Then these values are put into a motion grids.
	 * It takes the vertical and horizontal frame numbers, in addition to the value of the scaler.
	 * This scaler is used to amplify the weak movement that results fro them optical flow.
	 * @param expName The name of the expirement, this name will log the drawn frames.
	 * @param draw A flag that draws the pre-processed information in the build file.
	 * @param frames_width % to scale the output grid.
	 * @param frames_height % to scale the output grid.
	 * @param frames_total Number of frames that are used to create a grid.
	 * @param mg_vertical_frames Number of vertical frames.
	 * @param mg_horizontal_frames Number of horizontal frames.
	 *
	 */
	class MotionGridV5 : public UniquePassProcess
	{
	public:
		MotionGridV5();
		MotionGridV5(std::string expName, size_t draw = 0, size_t frames_width = 0, size_t frames_height = 0, size_t frames_total = 24, size_t mg_vertical_frames = 4, size_t mg_horizontal_frames = 8, size_t scalar = 50);

		virtual Shape compute_shape(const Shape &shape);
		virtual void process_train(const std::string &label, Tensor<float> &sample);
		virtual void process_test(const std::string &label, Tensor<float> &sample);

	private:
		void _process(const std::string &label, Tensor<float> &in) const;

		size_t _draw;
		std::string _expName;
		// The number of frames in the motion grid.
		size_t _mg_vertical_frames;
		size_t _mg_horizontal_frames;
		size_t _frames_total;
		size_t _frames_width;
		size_t _frames_height;

		// a value used to amplify the insegnificant movement in the optical flow.
		size_t _scaler;

		size_t _width;
		size_t _width_1;
		size_t _height;
		size_t _height_1;
		size_t _depth;
		size_t _conv_depth;
	};

}
#endif
