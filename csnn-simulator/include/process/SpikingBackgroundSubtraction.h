#ifndef _PROCESS_SPIKING_BACKGROUND_SUBTRACTION_H
#define _PROCESS_SPIKING_BACKGROUND_SUBTRACTION_H

#include <filesystem>
#include <iostream>
#include "Process.h"
#include "Spike.h"
#include "NumpyReader.h"
/**
 * SpikingBackgroundSubtraction
 */
namespace process
{

	namespace _priv
	{
		class SpikingBackgroundSubtractionHelper
		{

		public:
			SpikingBackgroundSubtractionHelper() = delete;
		};
	}

	/**
	 * @brief SpikingBackgroundSubtraction applies background subtraction on the level of spiking timesptamps.
	 * 
	 * @param expName the name of the expirement
	 * @param method 0 for seperating the information into two channels, 1 for keeping the information in the same channel with the _ve vales as abd value
	 */
	class SpikingBackgroundSubtraction : public UniquePassProcess
	{
	public:
		SpikingBackgroundSubtraction();
		SpikingBackgroundSubtraction(std::string expName, size_t method = 0, size_t threshold = 0);

		virtual Shape compute_shape(const Shape &shape);
		virtual void process_train(const std::string &label, Tensor<float> &sample);
		virtual void process_test(const std::string &label, Tensor<float> &sample);

	private:
		void _process(const std::string &label, Tensor<float> &in) const;

		std::string _expName;
		size_t _method;

		size_t _width;
		size_t _height;
		size_t _depth;
		size_t _conv_depth;
		size_t _threshold;
	};

}
#endif
