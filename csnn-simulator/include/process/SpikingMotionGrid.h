#ifndef _PROCESS_SPIKING_MOTION_GRID_H
#define _PROCESS_SPIKING_MOTION_GRID_H

#include <filesystem>
#include <iostream>
#include "Process.h"
#include "Spike.h"
#include "NumpyReader.h"
/**
 * 	SpikingMotionGrid - GO CHECK PYTHON CODE IN /src/process/OpticalFlowProcess IN ORDER TO GENERATE MG
 */
namespace process
{

	namespace _priv
	{
		class SpikingMotionGridHelper
		{

		public:
			SpikingMotionGridHelper() = delete;
		};
	}

	/**
	 * @brief SpikingMotionGrid applies background subtraction on the level of spiking timesptamps.
	 */
	class SpikingMotionGrid : public UniquePassProcess
	{
	public:
		SpikingMotionGrid();
		SpikingMotionGrid(std::string expName, size_t threshold = 0, size_t mg_vertical_frames = 8, size_t mg_horizontal_frames = 16, size_t scalar = 50);

		virtual Shape compute_shape(const Shape &shape);
		virtual void process_train(const std::string &label, Tensor<float> &sample);
		virtual void process_test(const std::string &label, Tensor<float> &sample);

	private:
		void _process(const std::string &label, Tensor<float> &in) const;
		// The number of frames in the motion grid.
		size_t _mg_vertical_frames;
		size_t _mg_horizontal_frames;

		std::string _expName;
		// a value used to amplify the insegnificant movement in the optical flow.
		size_t _scaler;

		size_t _width;
		size_t _height;
		size_t _depth;
		size_t _conv_depth;
		size_t _threshold;
	};

}
#endif
