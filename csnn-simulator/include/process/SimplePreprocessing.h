#ifndef _PROCESS_SIMPLE_PREPROCESSING_H
#define _PROCESS_SIMPLE_PREPROCESSING_H

#include <filesystem>
#include <iostream>
#include "Process.h"
#include "NumpyReader.h"

namespace process
{
	/**
	 * @brief This method creates background subtraction between each two consiqutive frames.
	 * In order to use this function, make ure that the input temporal depth is greater than 2, because this process operates on two frames at a time.
	 *
	 * @param expName The name of the expirement, in order to choose the folder name.
	 * @param method A flag if 0 the +ve and -ve pixel changes are separated into 2 channels. If 1 the -ve are changed to abs. If 2 they are kept as is.
	 * @param draw If set to 1, the input frames are drawn in the ReDrawInput folder in the build file. Otherwise, nothing is drawn.
	 */
	class SimplePreprocessing : public UniquePassProcess
	{

	public:
		SimplePreprocessing();
		SimplePreprocessing(std::string expName, size_t method = 0, size_t draw = 0);

		virtual Shape compute_shape(const Shape &shape);
		virtual void process_train(const std::string &label, Tensor<float> &sample);
		virtual void process_test(const std::string &label, Tensor<float> &sample);

	private:
		void _process(const std::string &label, Tensor<float> &in) const;

		size_t _draw;
		std::string _expName;
		// a value used to amplify the optical flow.
		size_t _method;
		// To save the drawings
		std::string _file_path;

		size_t _width;
		size_t _height;
		size_t _depth;
		size_t _conv_depth;
	};

}
#endif
