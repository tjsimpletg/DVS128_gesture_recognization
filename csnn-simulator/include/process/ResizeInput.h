#ifndef _PROCESS_RESIZE_INPUT_H
#define _PROCESS_RESIZE_INPUT_H

#include <filesystem>
#include <iostream>
#include "Process.h"
#include "NumpyReader.h"

namespace process
{
	/**
	 * @brief This method resizes the input frames.
	 * 
	 * @param expName The name of the expirement, in order to choose the folder name.
	 * @param frame_size_height the expected output height
	 * @param frame_size_width the expected output width
	 * @param draw If set to 1, the input frames are drawn in the ReDrawInput folder in the build file. Otherwise, nothing is drawn.
	 * @param scalar A constant to scale the input.
	 */
	class ResizeInput : public UniquePassProcess
	{

	public:
		ResizeInput();
		ResizeInput(std::string expName, size_t frame_size_width = 0, size_t frame_size_height = 0, size_t draw = 0, size_t scalar = 1);

		virtual Shape compute_shape(const Shape &shape);
		virtual void process_train(const std::string &label, Tensor<float> &sample);
		virtual void process_test(const std::string &label, Tensor<float> &sample);

	private:
		void _process(const std::string &label, Tensor<float> &in) const;

		size_t _draw;
		size_t _frame_size_width; 
		size_t _frame_size_height;
		std::string _expName;
		//a value used to amplify the optical flow.
		size_t _scaler;
		// To save the drawings
		std::string _file_path;

		size_t _width;
		size_t _height;
		size_t _depth;
		size_t _conv_depth;
	};

}
#endif
