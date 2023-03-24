#include "process/OnOffTempFilter.h"
#include "Experiment.h"
#include "Math.h"

using namespace process;

/**
 * @brief A on-center/off-center filter mimics the ratina if the eye, where the on-center receptive field shows an excitatory response when
 * stimulated at the center and an inhibitory response when stimulated at the peripheral part. In contrast, the off-center receptive field
 * shows an inhibitory response when stimulated centrally and an excitatory response when stimulated peripherally.
 *
 * @param filter_size The size of the gaussian filter
 * @param center_dev The varience of the gaussian equation at the center.
 * @param surround_dev The varience of the gaussian equation at the peripheral part.
 * @return Tensor<float>
 */
Tensor<float> process::_priv::OnOffTempFilterHelper::generate_spacial_filter(size_t filter_size, float center_dev, float surround_dev)
{

	Tensor<float> d2(Shape({filter_size, filter_size}));
	for (size_t i = 0; i < filter_size; i++)
	{
		for (size_t j = 0; j < filter_size; j++)
		{
			d2.at(i, j) = std::pow((i + 1) - static_cast<float>(filter_size) / 2.0f - 0.5f, 2.0f) + std::pow((j + 1) - static_cast<float>(filter_size) / 2.0f - 0.5f, 2.0f);
		}
	}

	Tensor<float> filter1(Shape({filter_size, filter_size}));
	Tensor<float> filter2(Shape({filter_size, filter_size}));
	float filter_sum1 = 0;
	float filter_sum2 = 0;

	for (size_t i = 0; i < filter_size; i++)
	{
		for (size_t j = 0; j < filter_size; j++)
		{
			filter1.at(i, j) = (1.0f / (2.0f * static_cast<float>(M_PI))) * (1.0f / (center_dev * center_dev)) * (std::exp(-d2.at(i, j) / 2.0f / (center_dev * center_dev)));
			filter_sum1 += filter1.at(i, j);

			filter2.at(i, j) = (1.0f / (2.0f * static_cast<float>(M_PI))) * (1.0f / (surround_dev * surround_dev)) * (std::exp(-d2.at(i, j) / 2.0f / (surround_dev * surround_dev)));
			filter_sum2 += filter2.at(i, j);
		}
	}

	Tensor<float> filter(Shape({filter_size, filter_size}));
	for (size_t i = 0; i < filter_size; i++)
	{
		for (size_t j = 0; j < filter_size; j++)
		{
			// The final filter is the difference between the two DoG filters.
			filter.at(i, j) = filter1.at(i, j) / filter_sum1 - filter2.at(i, j) / filter_sum2;
		}
	}

	return filter;
}

Tensor<float> process::_priv::OnOffTempFilterHelper::generate_temporal_filter(size_t tmp_filter_size, float center_tau, float surround_tau)
{

	Tensor<float> t2(Shape({tmp_filter_size}));
	/**
	 * @brief t2 is the value of (k^2) in the case of 1D temporal filtering.
	 */
	for (size_t k = 0; k < tmp_filter_size; k++)
		t2.at(k) = std::pow((k + 1) - static_cast<float>(tmp_filter_size) / 2.0f - 0.5f, 2.0f);

	Tensor<float> filter1(Shape({tmp_filter_size}));
	Tensor<float> filter2(Shape({tmp_filter_size}));
	float filter_sum1 = 0;
	float filter_sum2 = 0;

	for (size_t k = 0; k < tmp_filter_size; k++)
	{
		filter1.at(k) = 1.0f / std::sqrt(2.0f * static_cast<float>(M_PI)) * (1.0f / center_tau * std::exp(-t2.at(k) / 2.0f / (center_tau * center_tau)));
		filter_sum1 += filter1.at(k);

		filter2.at(k) = 1.0f / std::sqrt(2.0f * static_cast<float>(M_PI)) * (1.0f / surround_tau * std::exp(-t2.at(k) / 2.0f / (surround_tau * surround_tau)));
		filter_sum2 += filter2.at(k);
	}

	Tensor<float> filter(Shape({tmp_filter_size}));

	for (size_t k = 0; k < tmp_filter_size; k++)
	{
		filter.at(k) = filter1.at(k) / filter_sum1 - filter2.at(k) / filter_sum2;
	}

	return filter;
}

/**
 * @brief This generatefilter function creates a difference-of-Gaussians filter by subtracting one Gaussian kernel to another one of different variance.
 *
 * A on-center/off-center filter mimics the ratina if the eye, where the on-center receptive field shows an excitatory response when
 * stimulated at the center and an inhibitory response when stimulated at the peripheral part. In contrast, the off-center receptive field
 * shows an inhibitory response when stimulated centrally and an excitatory response when stimulated peripherally.
 *
 * @param filter_size The size of the gaussian filter
 * @param center_dev The varience of the gaussian equation at the center.
 * @param surround_dev The varience of the gaussian equation at the peripheral part.
 * @return Tensor<float>
 */
Tensor<float> process::_priv::OnOffTempFilterHelper::generate_3D_filter(size_t filter_size, size_t tmp_filter_size, float center_dev, float surround_dev, float center_tau, float surround_tau)
{

	Tensor<float> d2(Shape({filter_size, filter_size}));
	Tensor<float> t2(Shape({tmp_filter_size}));
	/**
	 * @brief d2 is the value of (x^2 + y^2) in the case of 2D spatial filtering, but with 3D spatio-temoral filtering, we also need t2 which corresponds to 1D temporal filtering.
	 */

	for (size_t i = 0; i < filter_size; i++)
		for (size_t j = 0; j < filter_size; j++)
			d2.at(i, j) = std::pow((i + 1) - static_cast<float>(filter_size) / 2.0f - 0.5f, 2.0f) + std::pow((j + 1) - static_cast<float>(filter_size) / 2.0f - 0.5f, 2.0f);

	for (size_t k = 0; k < tmp_filter_size; k++)
		t2.at(k) = std::pow((k + 1) - static_cast<float>(tmp_filter_size) / 2.0f - 0.5f, 2.0f);

	Tensor<float> filter1(Shape({filter_size, filter_size, tmp_filter_size}));
	Tensor<float> filter2(Shape({filter_size, filter_size, tmp_filter_size}));
	float filter_sum1 = 0;
	float filter_sum2 = 0;

	for (size_t i = 0; i < filter_size; i++)
		for (size_t j = 0; j < filter_size; j++)
			for (size_t k = 0; k < tmp_filter_size; k++)
			{
				filter1.at(i, j, k) = (1.0f / std::sqrt(std::pow(2.0f * static_cast<float>(M_PI), 3) * std::pow(center_dev, 4) * std::pow(center_tau, 2))) * (std::exp((-d2.at(i, j) / 2.0f / (center_dev * center_dev)) - (t2.at(k) / 2.0f / (center_tau * center_tau))));
				filter_sum1 += filter1.at(i, j, k);

				filter2.at(i, j, k) = (1.0f / std::sqrt(std::pow(2.0f * static_cast<float>(M_PI), 3) * std::pow(surround_dev, 4) * std::pow(surround_tau, 2))) * (std::exp((-d2.at(i, j) / 2.0f / (surround_dev * surround_dev)) - (t2.at(k) / 2.0f / (surround_tau * surround_tau))));
				filter_sum2 += filter2.at(i, j, k);
			}

	Tensor<float> filter(Shape({filter_size, filter_size, tmp_filter_size}));
	for (size_t i = 0; i < filter_size; i++)
		for (size_t j = 0; j < filter_size; j++)
			for (size_t k = 0; k < tmp_filter_size; k++)
			{
				filter.at(i, j, k) = filter1.at(i, j, k) / filter_sum1 - filter2.at(i, j, k) / filter_sum2;
			}

	return filter;
}

//
//	DefaultOnOffTempFilter
//

static RegisterClassParameter<DefaultOnOffTempFilter, ProcessFactory> _register_1("DefaultOnOffTempFilter");

DefaultOnOffTempFilter::DefaultOnOffTempFilter() : UniquePassProcess(_register_1),
												   _filter_size(0), _tmp_filter_size(0), _center_dev(0), _surround_dev(0), _height(0), _width(0), _depth(0), _conv_depth(0), _spacial_filter(), _temporal_filter()
{
	add_parameter("filter_size", _filter_size);
	add_parameter("tmp_filter_size", _tmp_filter_size);
	add_parameter("center_dev", _center_dev);
	add_parameter("surround_dev", _surround_dev);
	add_parameter("center_tau", _center_tau);
	add_parameter("surround_tau", _surround_tau);
}

DefaultOnOffTempFilter::DefaultOnOffTempFilter(size_t filter_size, size_t tmp_filter_size, float center_dev, float surround_dev, float center_tau, float surround_tau) : DefaultOnOffTempFilter()
{
	parameter<size_t>("filter_size").set(filter_size);
	parameter<size_t>("tmp_filter_size").set(tmp_filter_size);
	parameter<float>("center_dev").set(center_dev);
	parameter<float>("surround_dev").set(surround_dev);
	parameter<float>("center_tau").set(center_tau);
	parameter<float>("surround_tau").set(surround_tau);
}

Shape DefaultOnOffTempFilter::compute_shape(const Shape &shape)
{
	parameter<size_t>("filter_size").ensure_initialized(experiment()->random_generator());
	parameter<size_t>("tmp_filter_size").ensure_initialized(experiment()->random_generator());
	parameter<float>("center_dev").ensure_initialized(experiment()->random_generator());
	parameter<float>("surround_dev").ensure_initialized(experiment()->random_generator());
	parameter<float>("center_tau").ensure_initialized(experiment()->random_generator());
	parameter<float>("surround_tau").ensure_initialized(experiment()->random_generator());

	_height = shape.dim(0);
	_width = shape.dim(1);
	_depth = shape.dim(2);
	_conv_depth = shape.number() > 3 ? shape.dim(3) : 1;

	// _filter = _priv::OnOffTempFilterHelper::generate_3D_filter(_filter_size, _tmp_filter_size, _center_dev, _surround_dev, _center_tau, _surround_tau);
	_spacial_filter = _priv::OnOffTempFilterHelper::generate_spacial_filter(_filter_size, _center_dev, _surround_dev);
	_temporal_filter = _priv::OnOffTempFilterHelper::generate_temporal_filter(_tmp_filter_size, _center_tau, _surround_tau);
	return Shape({_height, _width, _depth * 2, _conv_depth});
}

void DefaultOnOffTempFilter::process_train(const std::string &, Tensor<float> &sample)
{
	_process(sample);
}

void DefaultOnOffTempFilter::process_test(const std::string &, Tensor<float> &sample)
{
	_process(sample);
}

void DefaultOnOffTempFilter::_process(Tensor<InputType> &in) const
{
	Tensor<InputType> out(Shape({_height, _width, _depth * 2, _conv_depth})); // the depth is doubled because there must be two seperate channels for the off cells and the on cells.

	for (size_t x = 0; x < _height; x++)
		for (size_t y = 0; y < _width; y++)
			for (size_t z = 0; z < _depth; z++)
				for (size_t k = 0; k < _conv_depth; k++)
				{
					float v = 0;
					for (size_t fk = 0; fk < _tmp_filter_size; fk++)
					{
						size_t k_in = k + fk > _tmp_filter_size / 2 ? std::min(k + fk - _tmp_filter_size / 2, _conv_depth - 1) : 0;

						v += in.at(x, y, z, k_in) * _temporal_filter.at(fk);
					}

					out.at(x, y, z * 2, k) = std::max<float>(0, v);
					out.at(x, y, z * 2 + 1, k) = std::max<float>(0, -v);
				}

	in = out;
	// Tensor<float>::draw_tensor("/home/melassal/Workspace/CSNN/csnn-simulator-build/test1/", out);
}

// // 1D with error that is giving good results!
// void DefaultOnOffTempFilter::_process(Tensor<InputType> &in) const
// {
// 	Tensor<InputType> out(Shape({_height, _width, _depth * 2, _conv_depth})); // the depth is doubled because there must be two seperate channels for the off cells and the on cells.

// 	for (size_t x = 0; x < _height; x++)
// 	{
// 		for (size_t y = 0; y < _width; y++)
// 		{
// 			for (size_t z = 0; z < _depth; z++)
// 			{
// 				for (size_t k = 0; k < _conv_depth; k++)
// 				{
// 					float v = 0;
// 					for (size_t fx = 0; fx < _filter_size; fx++)
// 						for (size_t fy = 0; fy < _filter_size; fy++)
// 							for (size_t fk = 0; fk < _tmp_filter_size; fk++)
// 							{
// 								size_t k_in = k + fk > _tmp_filter_size / 2 ? std::min(k + fk - _tmp_filter_size / 2, _conv_depth - 1) : 0;

// 								v += in.at(x, y, z, k_in) * _temporal_filter.at(fk);
// 							}

// 					out.at(x, y, z * 2, k) = std::max<float>(0, v);
// 					out.at(x, y, z * 2 + 1, k) = std::max<float>(0, -v);
// 				}
// 			}
// 		}
// 	}

// 	in = out;
// 	// Tensor<float>::draw_tensor("/home/melassal/Workspace/CSNN/csnn-simulator-build/test1/", out);
// }


// void DefaultOnOffTempFilter::_process(Tensor<InputType> &in) const
// {
// 	Tensor<InputType> out(Shape({_height, _width, _depth * 2, _conv_depth})); // the depth is doubled because there must be two seperate channels for the off cells and the on cells.

// 	for (size_t x = 0; x < _height; x++)
// 		for (size_t y = 0; y < _width; y++)
// 			for (size_t z = 0; z < _depth; z++)
// 				for (size_t k = 0; k < _conv_depth; k++)
// 				{
// 					float v = 0;
// 					// for (size_t fx = 0; fx < _filter_size; fx++)
// 					// 	for (size_t fy = 0; fy < _filter_size; fy++)
// 					for (size_t fk = 0; fk < _tmp_filter_size; fk++)
// 					{
// 						// size_t x_in = x + fx > _filter_size / 2 ? std::min(x + fx - _filter_size / 2, _height - 1) : 0;
// 						// size_t y_in = y + fy > _filter_size / 2 ? std::min(y + fy - _filter_size / 2, _width - 1) : 0;
// 						size_t k_in = k + fk > _tmp_filter_size / 2 ? std::min(k + fk - _tmp_filter_size / 2, _conv_depth - 1) : 0;

// 						v += in.at(x, y, z, k_in) * _temporal_filter.at(fk);
// 					}

// 					out.at(x, y, z * 2, k) = std::max<float>(0, v);
// 					out.at(x, y, z * 2 + 1, k) = std::max<float>(0, -v);
// 				}

// 	in = out;

// 	for (size_t x = 0; x < _height; x++)
// 		for (size_t y = 0; y < _width; y++)
// 			for (size_t z = 0; z < _depth; z++)
// 				for (size_t k = 0; k < _conv_depth; k++)
// 				{
// 					float v = 0;
// 					for (size_t fx = 0; fx < _filter_size; fx++)
// 						for (size_t fy = 0; fy < _filter_size; fy++)
// 							for (size_t fk = 0; fk < _tmp_filter_size; fk++)
// 							{
// 								size_t x_in = x + fx > _filter_size / 2 ? std::min(x + fx - _filter_size / 2, _height - 1) : 0;
// 								size_t y_in = y + fy > _filter_size / 2 ? std::min(y + fy - _filter_size / 2, _width - 1) : 0;
// 								// size_t k_in = k + fk > _tmp_filter_size / 2 ? std::min(k + fk - _tmp_filter_size / 2, _conv_depth - 1) : 0;

// 								v += in.at(x_in, y_in, z, k) * _spacial_filter.at(fx, fy);
// 							}

// 					out.at(x, y, z * 2, k) = std::max<float>(0, v);
// 					out.at(x, y, z * 2 + 1, k) = std::max<float>(0, -v);
// 				}

// 	in = out;
// 	// Tensor<float>::draw_tensor("/home/melassal/Workspace/CSNN/csnn-simulator-build/test1/", out);
// }
