#include "process/ResizeInput.h"

using namespace process;

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

//
//	ResizeInput
//
static RegisterClassParameter<ResizeInput, ProcessFactory> _register_1("ResizeInput");

ResizeInput::ResizeInput() : UniquePassProcess(_register_1), _expName(""), _draw(0), _frame_size_width(0), _frame_size_height(0), _scaler(0), _width(0), _height(0), _depth(0), _conv_depth(0)
{
	add_parameter("draw", _draw);
}

ResizeInput::ResizeInput(std::string expName, size_t frame_size_width, size_t frame_size_height, size_t draw, size_t scaler) : ResizeInput()
{
	parameter<size_t>("draw").set(draw);
	_expName = expName;
	_frame_size_width = frame_size_width;
	_frame_size_height = frame_size_height;

	if (draw == 1)
		std::filesystem::create_directories("Input_frames/" + _expName + "/RF/");

	_file_path = std::filesystem::current_path();
}

void ResizeInput::process_train(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

void ResizeInput::process_test(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

// The -1 is because this process looses one frame by getting the differences of 2 frames.
Shape ResizeInput::compute_shape(const Shape &shape)
{
	_height = _frame_size_height == 0 ? shape.dim(0) : _frame_size_height;
	_width = _frame_size_width == 0 ? shape.dim(1) : _frame_size_width;
	_depth = shape.dim(2);
	_conv_depth = shape.dim(3) == 2 ? 1 : shape.dim(3);
	return Shape({_height, _width, _depth, _conv_depth});
}

void ResizeInput::_process(const std::string &label, Tensor<InputType> &in) const
{
	std::vector<cv::Mat> _frames;
	std::vector<cv::Mat> _out_frames;

	Tensor<InputType> out(Shape({_height, _width, _depth, _conv_depth}));

	// This function returns a list of frames made up of the conv depth and the depth frames.
	Tensor<float>::tensor_to_multi_matrices(_frames, in);

	for (int _i = 0; _i < _frames.size(); _i++)
	{
		cv::Mat resized_frame;
		// width = int(img.shape[1] * scale_percent / 100)
		// height = int(img.shape[0] * scale_percent / 100)
		cv::resize(_frames[_i], resized_frame, cv::Size(_width, _height), cv::INTER_LINEAR);
		_out_frames.push_back(resized_frame);
	}

	Tensor<float>::multi_matrices_to_tensor(_out_frames, out, _depth);

	if (_draw == 1)
		Tensor<float>::draw_tensor(_file_path + "/Input_frames/" + _expName + "/RF/RF_" + label + "_", out);

	in = out;
}