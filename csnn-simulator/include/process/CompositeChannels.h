#ifndef _PROCESS_COMPOSITE_CHANNELS_H
#define _PROCESS_COMPOSITE_CHANNELS_H

#include <filesystem>
#include <iostream>
#include "Process.h"
#include "NumpyReader.h"

namespace process
{

	namespace _priv
	{
		class CompositeChannelsHelper
		{

		public:
			CompositeChannelsHelper() = delete;
		};
	}
	/**
	 * @brief CompositeChannels this pre-processing combines the two optical flow displacement channels (in the x and y directions) with the 
	 * grey-scale information of the moving part of the object in the video.
	 * 
	 * @param expName The name of the expirement, this name will log the drawn frames.
	 * @param draw A flag that draws the pre-processed information in the build file.
	 * @param scalar A constant used to augment the intensity of the optical flow if it was too low.
	 */
	class CompositeChannels : public UniquePassProcess
	{

	public:
		CompositeChannels();
		CompositeChannels(std::string _expName, size_t _draw = 0, size_t _scalar = 1);

		virtual Shape compute_shape(const Shape &shape);
		virtual void process_train(const std::string &label, Tensor<float> &sample);
		virtual void process_test(const std::string &label, Tensor<float> &sample);

	private:
		void _process(const std::string & label, Tensor<float> &in) const;

		size_t _draw;
		std::string _expName;
		//a value used to amplify frame size.
		
		std::string _file_path;
		size_t _scaler;
		size_t _width;
		size_t _height;
		size_t _depth;
		size_t _conv_depth;
	};

}
#endif
