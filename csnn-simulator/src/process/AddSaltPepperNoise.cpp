#include "process/AddSaltPepperNoise.h"

using namespace process;

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

//
//	AddSaltPepperNoise
//
static RegisterClassParameter<AddSaltPepperNoise, ProcessFactory> _register("AddSaltPepperNoise");

AddSaltPepperNoise::AddSaltPepperNoise() : UniquePassProcess(_register), _expName(""), _salt_scalar(0), _pepper_scalar(0), _draw(0), _width(0), _height(0), _depth(0), _conv_depth(0)
{
	add_parameter("draw", _draw);
}

AddSaltPepperNoise::AddSaltPepperNoise(std::string expName, size_t salt_scalar, size_t pepper_scalar, size_t draw) : AddSaltPepperNoise()
{
	parameter<size_t>("draw").set(draw);
	_salt_scalar = salt_scalar;
	_pepper_scalar = pepper_scalar;
	_expName = expName;

	if (draw == 1)
		std::filesystem::create_directories("Input_frames/" + _expName + "/S&PNoise/");

	_file_path = std::filesystem::current_path();
}

void AddSaltPepperNoise::process_train(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

void AddSaltPepperNoise::process_test(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

Shape AddSaltPepperNoise::compute_shape(const Shape &shape)
{
	_height = shape.dim(0);
	_width = shape.dim(1);
	_depth = shape.dim(2);
	_conv_depth = shape.dim(3);
	return Shape({_height, _width, _depth, _conv_depth});
}

void AddSaltPepperNoise::_process(const std::string &label, Tensor<InputType> &in) const
{
	Tensor<InputType> out(Shape({_height, _width, _depth, _conv_depth}));

	std::vector<cv::Mat> _frames;
	std::vector<cv::Mat> _out_frames;

	// This function returns a list of frames that have gone through background subtraction.
	Tensor<float>::tensor_to_multi_matrices(_frames, in);

	for (cv::Mat &_frame : _frames)
	{
		for (int counter = 0; counter < _pepper_scalar; ++counter)
			_frame.at<float>(rand() % _frame.rows, rand() % _frame.cols) = 0;

		for (int counter = 0; counter < _salt_scalar; ++counter)
			_frame.at<float>(rand() % _frame.rows, rand() % _frame.cols) = 255;

		_out_frames.push_back(_frame);
	}

	Tensor<float>::multi_matrices_to_tensor(_out_frames, out, _depth);

	if (_draw == 1)
		Tensor<float>::draw_tensor(_file_path + "/Input_frames/" + _expName + "/S&PNoise/S&PNoise_" + label + "_", out);

	in = out;
}