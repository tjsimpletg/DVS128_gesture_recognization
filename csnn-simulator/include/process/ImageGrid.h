#ifndef _PROCESS_IMAGE_GRID_H
#define _PROCESS_IMAGE_GRID_H

#include <filesystem>
#include <iostream>
#include "Process.h"
#include "NumpyReader.h"
/**
 * 	ImageGrid - GO CHECK PYTHON CODE IN /src/process/OpticalFlowProcess IN ORDER TO GENERATE MG
 */
namespace process
{

	/**
	 * @brief ImageGrid combines the optical flow information from a frame sequence into a motion grids.
	 * It takes the vertical and horizontal frame numbers, in addition to the value of the scaler.
	 * scalar * (std::abs(x) - x) / 2
	 * This scaler is used to amplify the weak movement that results fro them optical flow.
	 * @param expName The name of the expirement, this name will log the drawn frames.
	 * @param draw A flag that draws the pre-processed information in the build file.
	 * @param frames_width Width of the output grid.
	 * @param frames_height Height of the output grid.
	 * @param frames_total Number of frames that are used to create a grid.
	 * @param ig_vertical_frames Number of vertical frames.
	 * @param ig_horizontal_frames Number of horizontal frames.
	 *
	 */
	class ImageGrid : public UniquePassProcess
	{
	public:
		ImageGrid();
		ImageGrid(std::string expName, size_t draw = 0, size_t frames_width = 0, size_t frames_height = 0, size_t frames_total = 10, size_t ig_vertical_frames = 2, size_t ig_horizontal_frames = 4, size_t scalar = 50);

		virtual Shape compute_shape(const Shape &shape);
		virtual void process_train(const std::string &label, Tensor<float> &sample);
		virtual void process_test(const std::string &label, Tensor<float> &sample);

	private:
		void _process(const std::string &label, Tensor<float> &in) const;

		size_t _draw;
		std::string _expName;
		// The number of frames in the motion grid.
		size_t _ig_vertical_frames;
		size_t _ig_horizontal_frames;
		size_t _frames_total;
		size_t _frames_width;
		size_t _frames_height;

		// a value used to amplify the insegnificant movement in the optical flow.
		size_t _scaler;

		size_t _width;
		size_t _height;
		size_t _depth;
		size_t _conv_depth;
	};

}
#endif
