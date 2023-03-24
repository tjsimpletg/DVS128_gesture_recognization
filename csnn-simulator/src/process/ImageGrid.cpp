#include "process/ImageGrid.h"

using namespace process;

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

//
//	ImageGrid
//
static RegisterClassParameter<ImageGrid, ProcessFactory> _register_1("ImageGrid");

ImageGrid::ImageGrid() : UniquePassProcess(_register_1), _expName(""), _draw(0), _frames_width(0), _frames_height(0),
						 _ig_vertical_frames(0), _ig_horizontal_frames(0), _scaler(0), _frames_total(0), _width(0), _height(0), _depth(0), _conv_depth(0)
{
	add_parameter("draw", _draw);
	add_parameter("frames_total", _frames_total);
	add_parameter("frames_width", _frames_width);
	add_parameter("frames_height", _frames_height);
	add_parameter("ig_vertical_frames", _ig_vertical_frames);
	add_parameter("ig_horizontal_frames", _ig_horizontal_frames);
	add_parameter("scaler", _scaler);
}

ImageGrid::ImageGrid(std::string expName, size_t draw, size_t frames_width, size_t frames_height, size_t frames_total, size_t ig_vertical_frames, size_t ig_horizontal_frames, size_t scaler) : ImageGrid()
{
	parameter<size_t>("draw").set(draw);
	parameter<size_t>("frames_total").set(frames_total);
	parameter<size_t>("frames_width").set(frames_width);
	parameter<size_t>("frames_height").set(frames_height);
	parameter<size_t>("ig_vertical_frames").set(ig_vertical_frames);
	parameter<size_t>("ig_horizontal_frames").set(ig_horizontal_frames);
	parameter<size_t>("scaler").set(scaler);

	_draw = draw;
	_frames_total = frames_total;
	_frames_width = frames_width;
	_frames_height = frames_height;
	_ig_vertical_frames = ig_vertical_frames;
	_ig_horizontal_frames = ig_horizontal_frames;
	_scaler = scaler;
	_expName = expName;

	if (draw == 1)
	{
		std::filesystem::create_directories("Input_frames/" + _expName + "/MG/");
	}
}

void ImageGrid::process_train(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

void ImageGrid::process_test(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

Shape ImageGrid::compute_shape(const Shape &shape)
{
	_height = shape.dim(0);
	_width = shape.dim(1);
	_depth = shape.dim(2);
	_conv_depth = shape.dim(3);

	if (_frames_width != 0 || _frames_height != 0)
		return Shape({_frames_height, _frames_width, _depth, 1});
	else
		return Shape({_height * _ig_vertical_frames, _width * _ig_horizontal_frames, _depth, 1});
}

void ImageGrid::_process(const std::string &label, Tensor<InputType> &in) const
{
	///////////////////////////////
	std::string delimiter = ";.";
	std::string _label = label;
	std::string _experimentName = _label.substr(0, _label.find(delimiter));
	_label.erase(0, _experimentName.length() + delimiter.length());
	std::string _layerIndex = _label.substr(0, _label.find(delimiter));
	_label.erase(0, _layerIndex.length() + delimiter.length());
	//////////////////////////////
	std::string _file_path = std::filesystem::current_path();
	std::vector<cv::Mat> _frames;

	// This function returns a list of frames that have gone through background subtraction.
	if (_depth == 3)
		Tensor<float>::tensor_to_colored_matrices(_frames, in);
	else
		Tensor<float>::tensor_to_matrices(_frames, in);

	float SCALER = _scaler;

	int VERTICAL_FRAMES = _ig_vertical_frames;
	int HORIZONTAL_FRAMES = _ig_horizontal_frames;

	Tensor<InputType> out(Shape({_height * VERTICAL_FRAMES, _width * HORIZONTAL_FRAMES, _depth, 1})); //, _conv_depth / _frames_total}));
	cv::Size _frame_size(_width, _height);
	cv::Size _frame_size2(_frames_width, _frames_height);

	cv::Mat totalframe = cv::Mat::zeros(_height * VERTICAL_FRAMES, _width * HORIZONTAL_FRAMES, CV_32FC1);
	cv::Mat reducedtotalframe = cv::Mat::zeros(_frames_height, _frames_width, CV_32FC1);
	cv::Mat _frameOne, _frameTwo;

	for (int _h = 0; _h < _ig_horizontal_frames; _h++)
		cv::hconcat(_frames[_h], totalframe, totalframe);
	for (int _v = 0; _v < _ig_vertical_frames; _v++)
		cv::vconcat(_frames[_v], totalframe, totalframe);

	int rand_start = rand() % 100;
	cv::imwrite("/home/melassal/Workspace/Datasets/test/frame_" + std::to_string(rand_start) + ".png", totalframe);
	
	if (_frames_width != 0 || _frames_height != 0)
	{
		cv::resize(totalframe, reducedtotalframe, _frame_size2);
		Tensor<float>::matrix_to_tensor(reducedtotalframe, out);
	}
	else
		Tensor<float>::matrix_to_tensor(totalframe, out);

	if (_draw == 1)
		Tensor<float>::draw_nonscaled_tensor(_file_path + "/Input_frames/" + _expName + "/MG/MG_" + _label + "_" + std::to_string(rand() % 100) + "_" + std::to_string(rand() % 100) + "_", out);

	totalframe = cv::Mat::zeros(_height * VERTICAL_FRAMES, _width * HORIZONTAL_FRAMES, CV_32FC1);

	in = out;
}