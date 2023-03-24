#include "process/EdgeGrid.h"

using namespace process;

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

//
//	EdgeGrid - GO CHECK PYTHON CODE IN /src/process/OpticalFlowProcess IN ORDER TO GENERATE EG
//
static RegisterClassParameter<EdgeGrid, ProcessFactory> _register_1("EdgeGrid");

EdgeGrid::EdgeGrid() : UniquePassProcess(_register_1), _expName(""), _draw(0), _frames_total(0), _vertical_frames(0), _horizontal_frames(0), _width(0), _height(0), _depth(0), _conv_depth(0)
{
	add_parameter("draw", _draw);
	add_parameter("frames_total", _frames_total);
	add_parameter("vertical_frames", _vertical_frames);
	add_parameter("horizontal_frames", _horizontal_frames);
}

EdgeGrid::EdgeGrid(std::string expName, size_t draw, size_t frames_total, size_t vertical_frames, size_t horizontal_frames) : EdgeGrid()
{
	parameter<size_t>("draw").set(draw);
	_expName = expName;
	parameter<size_t>("frames_total").set(frames_total);
	parameter<size_t>("vertical_frames").set(vertical_frames);
	parameter<size_t>("horizontal_frames").set(horizontal_frames);
	_frames_total = frames_total;
	if (draw == 1)
	{
		std::filesystem::create_directories("Input_frames/" + _expName + "/EG/");
	}
}

void EdgeGrid::process_train(const std::string &, Tensor<float> &sample)
{
	_process(sample);
}

void EdgeGrid::process_test(const std::string &, Tensor<float> &sample)
{
	_process(sample);
}

Shape EdgeGrid::compute_shape(const Shape &shape)
{
	_height = shape.dim(0);
	_width = shape.dim(1);
	_depth = shape.dim(2);
	_conv_depth = shape.dim(3);
	return Shape({_height, _width, _depth, _conv_depth / _frames_total});
}

void EdgeGrid::_process(Tensor<InputType> &in) const
{
	std::vector<cv::Mat> _frames;

	// This function returns a list of frames that have gone through background subtraction.
	Tensor<float>::tensor_to_matrices(_frames, in);

	int depthCounter = 0;
	int widthCounter = 0;
	int heightCounter = 0;

	int VERTICAL_FRAMES = _vertical_frames;
	int HORIZONTAL_FRAMES = _horizontal_frames;

	Tensor<InputType> out(Shape({_height * VERTICAL_FRAMES, _width * HORIZONTAL_FRAMES, _depth, 1})); //, _conv_depth / _frames_total}));
	cv::Size _frame_size(_width, _height);

	cv::Mat totalframe = cv::Mat::zeros(_height * VERTICAL_FRAMES, _width * HORIZONTAL_FRAMES, CV_32FC1);
	
	for (int _i = 0; _i < _frames.size() - 1; _i++)
	{
		cv::Mat flow(_frame_size, CV_32FC1);
		cv::Mat flowSum(_frame_size, CV_32FC1);

		cv::calcOpticalFlowFarneback(_frames[_i], _frames[_i + 1], flow, 0.5, 3, 15, 3, 5, 1.2, 0);

		// visualization
		cv::Mat flow_parts[2];
		split(flow, flow_parts);
		//flowSum = std::max(std::abs(flow_parts[0]), std::abs(flow_parts[1]));
	}

	in = out;
}