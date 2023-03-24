#include "process/Flatten.h"

using namespace process;

//
//	Flatten
//
static RegisterClassParameter<Flatten, ProcessFactory> _register_1("Flatten");

Flatten::Flatten() : UniquePassProcess(_register_1), _product(0)
{
}

void Flatten::process_train(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

void Flatten::process_test(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

// The -1 is because this process looses one frame by getting the differences of 2 frames.
Shape Flatten::compute_shape(const Shape &shape)
{
	_product = shape.product();

	return Shape({_product});
}

void Flatten::_process(const std::string &label, Tensor<InputType> &in) const
{
	// The output fused tensor
	Tensor<InputType> out(Shape({_product}));

	// Flattening
	for (size_t _j = 0; _j < _product; _j++)
		out.at<float>(_j) = in.at_index(_j);

	in = out;
}