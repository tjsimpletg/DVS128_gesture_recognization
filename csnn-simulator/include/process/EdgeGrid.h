#ifndef _PROCESS_EDGE_GRID_H
#define _PROCESS_EDGE_GRID_H

#include <filesystem>
#include <iostream>
#include "Process.h"
#include "NumpyReader.h"

//	EdgeGrid - GO CHECK PYTHON CODE IN /src/process/OpticalFlowProcess IN ORDER TO GENERATE EG
namespace process
{

	namespace _priv
	{
		class EdgeGridHelper
		{

		public:
			EdgeGridHelper() = delete;
		};
	}
	/**
	 * @brief EdgeGrid extract the optical flow information from a sequence of frames and constructs the data in a grid form.
	 * 
	 * @param expName The name of the expirement, this name will log the drawn frames.
	 * @param draw A flag that draws the pre-processed information in the build file.
	 * @param frames_total Number of frames that are used to create a grid.
	 * @param vertical_frames Number of vertical frames.
	 * @param horizontal_frames Number of horizontal frames.
	 */
	class EdgeGrid : public UniquePassProcess
	{

	public:
		EdgeGrid();
		EdgeGrid(std::string expName, size_t draw = 0, size_t _frames_total = 1, size_t _vertical_frames = 1, size_t _horizontal_frames = 1);

		virtual Shape compute_shape(const Shape &shape);
		virtual void process_train(const std::string &label, Tensor<float> &sample);
		virtual void process_test(const std::string &label, Tensor<float> &sample);

	private:
		void _process(Tensor<float> &in) const;

		size_t _draw;
		std::string _expName;
		size_t _frames_total;
		size_t _vertical_frames;
		size_t _horizontal_frames;

		size_t _width;
		size_t _height;
		size_t _depth;
		size_t _conv_depth;
	};

}
#endif
