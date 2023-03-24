#include "process/SetTemporalDepth.h"

using namespace process;

//
//	SetTemporalDepth
//
static RegisterClassParameter<SetTemporalDepth, ProcessFactory> _register_1("SetTemporalDepth");

SetTemporalDepth::SetTemporalDepth() : UniquePassProcess(_register_1), _expName(""), _width(0), _height(0), _depth(0), _conv_depth(0), _frame_number(0)
{
}

SetTemporalDepth::SetTemporalDepth(std::string expName, size_t frame_number) : SetTemporalDepth()
{
	_frame_number = frame_number;
	_expName = expName;
}

void SetTemporalDepth::process_train(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

void SetTemporalDepth::process_test(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

Shape SetTemporalDepth::compute_shape(const Shape &shape)
{
	_height = shape.dim(0);
	_width = shape.dim(1);
	_depth = shape.dim(2);
	_conv_depth = _frame_number; 
	return Shape({_height, _width, _depth, _conv_depth});
}

void SetTemporalDepth::_process(const std::string &label, Tensor<InputType> &in) const
{
}