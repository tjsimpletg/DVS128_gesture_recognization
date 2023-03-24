#ifndef _PROCESS_ACCELERATION_H
#define _PROCESS_ACCELERATION_H

#include <filesystem>
#include <iostream>
#include "Process.h"
#include "NumpyReader.h"

namespace process
{

	namespace _priv
	{
		class AccelerationHelper
		{

		public:
			AccelerationHelper() = delete;
		};
	}

	/**
	 * @brief Extracts the optical flow information but as orientation and amplitude values. In drawing this data we use RGB.
	 * 
	 * @param expName The name of the expirement (set this to experiment.name()), in order to save the drawn value in a folder that has this name.
	 * @param draw A flag that draws the pre_procssed information in the build folder in a folder called Input_frames.
	 * @param scalar Scales up the optical flow.
	 */
	class Acceleration : public UniquePassProcess
	{

	public:
		Acceleration();
		Acceleration(std::string expName, size_t draw = 0, size_t scalar = 1);

		virtual Shape compute_shape(const Shape &shape);
		virtual void process_train(const std::string &label, Tensor<float> &sample);
		virtual void process_test(const std::string &label, Tensor<float> &sample);

	private:
		void _process(const std::string & label, Tensor<float> &in) const;
		size_t _draw;
		std::string _expName;
		//a value used to amplify frame size.
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
