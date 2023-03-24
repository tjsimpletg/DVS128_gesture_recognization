#ifndef _PROCESS_SPIKE_MOTION_GRID_H
#define _PROCESS_SPIKE_MOTION_GRID_H

#include <filesystem>
#include <iostream>
#include "Process.h"
#include "Spike.h"
#include "NumpyReader.h"
/**
 * 	SpikeMotionGrid - GO CHECK PYTHON CODE IN /src/process/OpticalFlowProcess IN ORDER TO GENERATE MG
*/
namespace process
{

	namespace _priv
	{
		class SpikeMotionGridHelper
		{

		public:
			SpikeMotionGridHelper() = delete;
		};
	}

/**
 * @brief SpikeMotionGrid combines the optical flow information from a frame sequence into a motion grids.
 * It takes the vertical and horizontal frame numbers, in addition to the value of the scaler.
 * This scaler is used to amplify the weak movement that results fro them optical flow.
 * @param expName The name of the expirement, this name will log the drawn frames.
 * @param draw A flag that draws the pre-processed information in the build file.
 * @param frames_width Width of the output grid.
 * @param frames_height Height of the output grid.
 * @param frames_total Number of frames that are used to create a grid.
 * @param mg_vertical_frames Number of vertical frames.
 * @param mg_horizontal_frames Number of horizontal frames.
 * 
 */
	class SpikeMotionGrid : public UniquePassProcess
	{	
	public:
		SpikeMotionGrid();
		SpikeMotionGrid(size_t threshold);

		virtual Shape compute_shape(const Shape &shape);
		virtual void process_train(const std::string &label, Tensor<float> &sample);
		virtual void process_test(const std::string &label, Tensor<float> &sample);

	private:
		void _process(const std::string &label, Tensor<float> &in) const;

		size_t _width;
		size_t _height;
		size_t _depth;
		size_t _conv_depth;
		size_t _threshold;
	};

}
#endif
