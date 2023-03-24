#include "process/Amplification.h"

using namespace process;

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

//
//	Amplification
//
static RegisterClassParameter<Amplification, ProcessFactory> _register("Amplification");

Amplification::Amplification() : UniquePassProcess(_register), _expName(""), _scalar(0), _draw(0), _width(0), _height(0), _depth(0), _conv_depth(0)
{
	add_parameter("draw", _draw);
}

Amplification::Amplification(std::string expName, size_t scalar, size_t draw) : Amplification()
{
	parameter<size_t>("draw").set(draw);
	_scalar = scalar;
	_expName = expName;

	if (draw == 1)
		std::filesystem::create_directories("Input_frames/" + _expName + "/Amp/");

	_file_path = std::filesystem::current_path();
}

void Amplification::process_train(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

void Amplification::process_test(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

Shape Amplification::compute_shape(const Shape &shape)
{
	_height = shape.dim(0);
	_width = shape.dim(1);
	_depth = shape.dim(2);
	_conv_depth = shape.dim(3);
	return Shape({_height, _width, _depth, _conv_depth});
}

void Amplification::_process(const std::string &label, Tensor<InputType> &in) const
{
	Tensor<InputType> out(Shape({_height, _width, _depth, _conv_depth}));
	// CONV_DEPTH by being incremented every frame.
	for (size_t conv = 0; conv < _conv_depth; conv++)
		for (size_t i = 0; i < _height; i++)
			for (size_t j = 0; j < _width; j++)
				for (size_t k = 0; k < _depth; k++)
				{
					out.at(i, j, k, conv) = in.at(i, j, k, conv) * _scalar;
				}

	if (_draw == 1)
		Tensor<float>::draw_tensor(_file_path + "/Input_frames/" + _expName + "/Amp/Amp_" + label + "_", out);

	in = out;
}