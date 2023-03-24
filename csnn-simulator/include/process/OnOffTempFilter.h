#ifndef _PROCESS_ON_OFF_Temp_FILTER_H
#define _PROCESS_ON_OFF_Temp_FILTER_H

#include <iostream>
#include "Process.h"
#include "NumpyReader.h"

namespace process
{

	namespace _priv
	{
		class OnOffTempFilterHelper
		{

		public:
			OnOffTempFilterHelper() = delete;

			static Tensor<float> generate_3D_filter(size_t filter_size, size_t tmp_filter_size, float center_dev, float surround_dev, float center_tau, float surround_tau);
			static Tensor<float> generate_spacial_filter(size_t filter_size, float center_dev, float surround_dev);
			static Tensor<float> generate_temporal_filter(size_t tmp_filter_size, float center_tau, float surround_tau);
		};
	}

	/**
	 * @brief On-center/off-center filtering in a pre-processing technique that is similar to DoG filtering to detect edges.
	 * In fact, it is to extract the difference in intensity which helps the SNN encode the image as spikes.
	 * This filter puts the on and off cells in 2 different channels.
	 *
	 * @param filter_size size_t - The size of the on-center/off-center filter in the spacial dimensions.
	 * @param center_dev float - The variance of the Gaussian kernels DoG center.
	 * @param surround_dev float - The variance of the Gaussian kernels DoG surround.
	 */
	class DefaultOnOffTempFilter : public UniquePassProcess
	{

	public:
		DefaultOnOffTempFilter();
		DefaultOnOffTempFilter(size_t filter_size, size_t tmp_filter_size = 1, float center_dev = 1, float surround_dev = 4, float center_tau = 1, float surround_tau = 4);

		virtual Shape compute_shape(const Shape &shape);
		virtual void process_train(const std::string &label, Tensor<float> &sample);
		virtual void process_test(const std::string &label, Tensor<float> &sample);

	private:
		void _process(Tensor<float> &in) const;

		size_t _filter_size;
		size_t _tmp_filter_size;
		float _center_dev;
		float _surround_dev;
		float _center_tau;
		float _surround_tau;

		size_t _width;
		size_t _height;
		size_t _depth;
		size_t _conv_depth;
		Tensor<float> _filter;
		Tensor<float> _spacial_filter;
		Tensor<float> _temporal_filter;
	};

}
#endif
