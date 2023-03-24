#include "process/MaxScaling.h"

using namespace process;

//
//	MaxScaling
//
static RegisterClassParameter<MaxScaling, ProcessFactory> _register_1("MaxScaling");

MaxScaling::MaxScaling() : UniquePassProcess(_register_1), _width(0), _height(0), _depth(0), _conv_depth(0)
{
}

void MaxScaling::process_train(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

void MaxScaling::process_test(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

Shape MaxScaling::compute_shape(const Shape &shape)
{
	_height = shape.dim(0);
	_width = shape.dim(1);
	_depth = shape.dim(2);
	_conv_depth = shape.dim(3);
	
	return Shape({_height, _width, _depth, _conv_depth});
}

void MaxScaling::_process(const std::string &label, Tensor<InputType> &in) const
{
	Tensor<float>::normalize_tensor(in);
}