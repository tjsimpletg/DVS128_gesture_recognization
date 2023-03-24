#include "process/SimplePreprocessing.h"

using namespace process;

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

//
//	SimplePreprocessing
//
static RegisterClassParameter<SimplePreprocessing, ProcessFactory> _register_1("SimplePreprocessing");

SimplePreprocessing::SimplePreprocessing() : UniquePassProcess(_register_1), _expName(""), _method(0), _draw(0), _width(0), _height(0), _depth(0), _conv_depth(0)
{
	add_parameter("draw", _draw);
	add_parameter("method", _method);
}

SimplePreprocessing::SimplePreprocessing(std::string expName, size_t method, size_t draw) : SimplePreprocessing()
{
	parameter<size_t>("draw").set(draw);
	parameter<size_t>("method").set(method);
	_method = method;
	_expName = expName;
	if (draw == 1)
		std::filesystem::create_directories("Input_frames/" + _expName + "/SP/");

	_file_path = std::filesystem::current_path();
}

void SimplePreprocessing::process_train(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

void SimplePreprocessing::process_test(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

// The -1 is because this process looses one frame by getting the differences of 2 frames.
Shape SimplePreprocessing::compute_shape(const Shape &shape)
{
	_height = shape.dim(0);
	_width = shape.dim(1);
	_depth = _method == 0 ? shape.dim(2) * 2 : 1;
	_conv_depth = shape.dim(3) == 2 ? 1 : shape.dim(3);

	if (_conv_depth < 2)
	{
		throw std::runtime_error("This pre-processing needs more than 1 frame! increase the video frame count.");
	}

	return Shape({_height, _width, _depth, _conv_depth});
}

void SimplePreprocessing::_process(const std::string &label, Tensor<InputType> &in) const
{
	std::vector<cv::Mat> _frames;
	std::vector<cv::Mat> _out_frames;

	Tensor<InputType> out(Shape({_height, _width, _depth, _conv_depth}));

	// std::vector<Tensor<float>> _out_tensors;
	// // Transforms 1 tensor to multiple.
	// Tensor<float>::tensor_to_tensors(_out_tensors, in);
	// for (int _i = 0; _i < _out_tensors.size(); _i++)
	// {
	// 	Tensor<float>::draw_tensor(_file_path + "/Input_frames/test/" + _expName + "_" + label + std::to_string(_i) + "_", _out_tensors[_i]);
	// }
	// This function returns a list of frames that have gone through background subtraction.
	Tensor<float>::tensor_to_matrices(_frames, in);
	// I will add a frame, in order to conserve the same temporal depth.
	_frames.push_back(_frames[_frames.size() - 2]);

	for (int _i = 0; _i < _frames.size() - 1; _i++)
	{
		cv::Mat difference_frame;
		if (!_frames[_i + 1].empty())
		{
			difference_frame = _frames[_i] - _frames[_i + 1];

			_out_frames.push_back(difference_frame);
		}
	}
	if (_method == 0)
	{
		Tensor<float>::matrices_to_split_sign_tensor(_out_frames, out);
		Tensor<float>::draw_split_tensor(_file_path + "/Input_frames/" + _expName + "/SP/SP_" + label + "_", out);
	}
	else if (_method == 1)
		Tensor<float>::matrices_to_abs_sign_tensor(_out_frames, out);
	else if (_method == 2)
		Tensor<float>::matrices_to_tensor(_out_frames, out);

	// if (_draw == 1)
	// 	Tensor<float>::draw_tensor(_file_path + "/Input_frames/" + _expName + "/SP/SP_" + label + "_", out);

	in = out;
}