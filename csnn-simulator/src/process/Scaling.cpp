#include "process/Scaling.h"

using namespace process;

//
//	FeatureScaling
//

static RegisterClassParameter<FeatureScaling, ProcessFactory> _register_1("FeatureScaling");

FeatureScaling::FeatureScaling() : TwoPassProcess(_register_1), _size(0), _min(), _max()
{
}

Shape FeatureScaling::compute_shape(const Shape &shape)
{
	_size = shape.product();
	_min = Tensor<float>(shape);
	_min.fill(std::numeric_limits<float>::max());
	_max = Tensor<float>(shape);
	_max.fill(std::numeric_limits<float>::min());
	return shape;
}

void FeatureScaling::compute(const std::string &, const Tensor<float> &sample)
{
	for (size_t i = 0; i < _size; i++)
	{
		_min.at_index(i) = std::min(_min.at_index(i), sample.at_index(i));
		_max.at_index(i) = std::max(_max.at_index(i), sample.at_index(i));
	}
}

void FeatureScaling::process_train(const std::string &, Tensor<float> &sample)
{
	//_train_scale_sample_count++; // a counter for the progress bar
	for (size_t i = 0; i < _size; i++)
	{
		sample.at_index(i) = _min.at_index(i) == _max.at_index(i) ? 0 : (sample.at_index(i) - _min.at_index(i)) / (_max.at_index(i) - _min.at_index(i));
	}
	//draw_progress(_train_scale_sample_count, get_train_count());
}
void FeatureScaling::process_test(const std::string &, Tensor<float> &sample)
{
	//_test_scale_sample_count++;  // a counter for the progress bar
	for (size_t i = 0; i < _size; i++)
	{
		sample.at_index(i) = _min.at_index(i) == _max.at_index(i) ? 0 : (sample.at_index(i) - _min.at_index(i)) / (_max.at_index(i) - _min.at_index(i));
	}
	//draw_progress(_test_scale_sample_count, get_test_count());
}

//
//	ChannelScaling
//

static RegisterClassParameter<ChannelScaling, ProcessFactory> _register_2("ChannelScaling");

ChannelScaling::ChannelScaling() : TwoPassProcess(_register_2),
								   _width(0), _height(0), _depth(0), _conv_depth(0), _min(), _max()
{
}

Shape ChannelScaling::compute_shape(const Shape &shape)
{
	_width = shape.dim(0);
	_height = shape.dim(1);
	_depth = shape.dim(2);
	_conv_depth = shape.number() > 3 ? shape.dim(3) : 1;
	_min = Tensor<float>(Shape({_depth}));
	_min.fill(std::numeric_limits<float>::max());
	_max = Tensor<float>(Shape({_depth}));
	_max.fill(std::numeric_limits<float>::min());
	return shape;
}

void ChannelScaling::compute(const std::string &, const Tensor<float> &sample)
{
	for (size_t x = 0; x < _width; x++)
		for (size_t y = 0; y < _height; y++)
			for (size_t z = 0; z < _depth; z++)
				for (size_t k = 0; k < _conv_depth; k++)
				{
					_min.at(z) = std::min(_min.at(z), sample.at(x, y, z, k));
					_max.at(z) = std::max(_max.at(z), sample.at(x, y, z, k));
				}
}

void ChannelScaling::process_train(const std::string &, Tensor<float> &sample)
{
	for (size_t x = 0; x < _width; x++)
		for (size_t y = 0; y < _height; y++)
			for (size_t z = 0; z < _depth; z++)
				for (size_t k = 0; k < _conv_depth; k++)
				{
					sample.at(x, y, z, k) = _max.at(z) == _min.at(z) ? 0 : (sample.at(x, y, z, k) - _min.at(z)) / (_max.at(z) - _min.at(z));
				}
}

void ChannelScaling::process_test(const std::string &, Tensor<float> &sample)
{
	for (size_t x = 0; x < _width; x++)
		for (size_t y = 0; y < _height; y++)
			for (size_t z = 0; z < _depth; z++)
				for (size_t k = 0; k < _conv_depth; k++)
				{
					sample.at(x, y, z, k) = _max.at(z) == _min.at(z) ? 0 : (sample.at(x, y, z, k) - _min.at(z)) / (_max.at(z) - _min.at(z));
				}
}

//
//	SampleScaling
//

static RegisterClassParameter<SampleScaling, ProcessFactory> _register_3("SampleScaling");

SampleScaling::SampleScaling() : TwoPassProcess(_register_3), _size(0), _min(0), _max(0)
{
}

Shape SampleScaling::compute_shape(const Shape &shape)
{
	_size = shape.product();
	_min = std::numeric_limits<float>::max();
	_max = std::numeric_limits<float>::min();
	return shape;
}

void SampleScaling::compute(const std::string &, const Tensor<float> &sample)
{
	for (size_t i = 0; i < _size; i++)
	{
		_min = std::min(_min, sample.at_index(i));
		_max = std::max(_max, sample.at_index(i));
	}
}

void SampleScaling::process_train(const std::string &, Tensor<float> &sample)
{
	if (_min == _max)
	{
		sample.fill(0);
	}
	else
	{
		for (size_t i = 0; i < _size; i++)
		{
			sample.at_index(i) = (sample.at_index(i) - _min) / (_max - _min);
		}
	}
}

void SampleScaling::process_test(const std::string &, Tensor<float> &sample)
{
	if (_min == _max)
	{
		sample.fill(0);
	}
	else
	{
		for (size_t i = 0; i < _size; i++)
		{
			sample.at_index(i) = (sample.at_index(i) - _min) / (_max - _min);
		}
	}
}

//
//	IndependentScaling
//

static RegisterClassParameter<IndependentScaling, ProcessFactory> _register_4("IndependentScaling");

IndependentScaling::IndependentScaling() : UniquePassProcess(_register_4)
{
}

Shape IndependentScaling::compute_shape(const Shape &shape)
{
	return shape;
}

void IndependentScaling::process_train(const std::string &, Tensor<float> &sample)
{
	auto minmax = std::minmax_element(std::begin(sample), std::end(sample));
	float min = *minmax.first;
	float max = *minmax.second;
	size_t size = sample.shape().product();
	for (size_t i = 0; i < size; i++)
	{
		sample.at_index(i) = (sample.at_index(i) - min) / (max - min);
	}
}

void IndependentScaling::process_test(const std::string &, Tensor<float> &sample)
{
	auto minmax = std::minmax_element(std::begin(sample), std::end(sample));
	float min = *minmax.first;
	float max = *minmax.second;
	size_t size = sample.shape().product();
	for (size_t i = 0; i < size; i++)
	{
		sample.at_index(i) = (sample.at_index(i) - min) / (max - min);
	}
}
