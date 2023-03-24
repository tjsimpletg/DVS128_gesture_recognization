#ifndef _PROCESS_SALT_PEPPER_H
#define _PROCESS_SALT_PEPPER_H

#include <filesystem>
#include <iostream>
#include "Process.h"
#include "NumpyReader.h"

namespace process
{
	/**
	 * @brief This method takes a frame and amplifies the existing pixels by multiplying them with a scalar value.
	 *
	 * @param expName The name of the expirement, in order to choose the folder name.
	 * @param scalar The value used to scale up the pixels.
	 * @param draw If set to 1, the input frames are drawn in the ReDrawInput folder in the build file. Otherwise, nothing is drawn.
	 */
	class AddSaltPepperNoise : public UniquePassProcess
	{

	public:
		AddSaltPepperNoise();
		AddSaltPepperNoise(std::string expName, size_t salt_scalar = 0, size_t pepper_scalar = 0, size_t draw = 0);

		virtual Shape compute_shape(const Shape &shape);
		virtual void process_train(const std::string &label, Tensor<float> &sample);
		virtual void process_test(const std::string &label, Tensor<float> &sample);

	private:
		void _process(const std::string &label, Tensor<float> &in) const;

		size_t _draw;
		std::string _expName;
		// a value used to amplify the optical flow.
		size_t _salt_scalar;
		size_t _pepper_scalar;
		// To save the drawings
		std::string _file_path;

		size_t _width;
		size_t _height;
		size_t _depth;
		size_t _conv_depth;
	};

}
#endif
