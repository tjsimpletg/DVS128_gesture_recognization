#include "process/MotionGridV1.h"

using namespace process;

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

//
//	MotionGridV1
//
static RegisterClassParameter<MotionGridV1, ProcessFactory> _register_1("MotionGridV1");

MotionGridV1::MotionGridV1() : UniquePassProcess(_register_1), _expName(""), _draw(0), _frames_width(0), _frames_height(0),
							   _mg_vertical_frames(0), _mg_horizontal_frames(0), _scaler(0), _frames_total(0), _width(0), _height(0), _depth(0), _conv_depth(0)
{
	add_parameter("draw", _draw);
	add_parameter("frames_total", _frames_total);
	add_parameter("frames_width", _frames_width);
	add_parameter("frames_height", _frames_height);
	add_parameter("mg_vertical_frames", _mg_vertical_frames);
	add_parameter("mg_horizontal_frames", _mg_horizontal_frames);
	add_parameter("scaler", _scaler);
}

MotionGridV1::MotionGridV1(std::string expName, size_t draw, size_t frames_width, size_t frames_height, size_t frames_total, size_t mg_vertical_frames, size_t mg_horizontal_frames, size_t scaler) : MotionGridV1()
{
	parameter<size_t>("draw").set(draw);
	parameter<size_t>("frames_total").set(frames_total);
	parameter<size_t>("frames_width").set(frames_width);
	parameter<size_t>("frames_height").set(frames_height);
	parameter<size_t>("mg_vertical_frames").set(mg_vertical_frames);
	parameter<size_t>("mg_horizontal_frames").set(mg_horizontal_frames);
	parameter<size_t>("scaler").set(scaler);

	_draw = draw;
	_frames_total = frames_total;
	_frames_width = frames_width;
	_frames_height = frames_height;
	_mg_vertical_frames = mg_vertical_frames;
	_mg_horizontal_frames = mg_horizontal_frames;
	_scaler = scaler;
	_expName = expName;

	if (draw == 1)
	{
		std::filesystem::create_directories("Input_frames/" + _expName + "/MG/");
	}
}

void MotionGridV1::process_train(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

void MotionGridV1::process_test(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

Shape MotionGridV1::compute_shape(const Shape &shape)
{
	_height = shape.dim(0);
	_width = shape.dim(1);
	_depth = 1;
	_conv_depth = shape.dim(3);

	if (_frames_width != 0 || _frames_height != 0)
		return Shape({_frames_height, _frames_width, _depth, 1});
	else
		return Shape({_height * _mg_vertical_frames, _width * _mg_horizontal_frames, _depth, 1});
}

void MotionGridV1::_process(const std::string &label, Tensor<InputType> &in) const
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
	Tensor<float>::tensor_to_matrices(_frames, in);

	int imagerow = 0;
	int imagecolumn = 0;
	float SCALER = _scaler;

	int VERTICAL_FRAMES = _mg_vertical_frames;
	int HORIZONTAL_FRAMES = _mg_horizontal_frames;
	int maxImagecolumn = ((HORIZONTAL_FRAMES / 4) - 1);
	int maxImagerow = VERTICAL_FRAMES - 1;

	Tensor<InputType> out(Shape({_height * VERTICAL_FRAMES, _width * HORIZONTAL_FRAMES, _depth, 1})); //, _conv_depth / _frames_total}));
	cv::Size _frame_size(_width, _height);
	cv::Size _frame_size2(_frames_width, _frames_height);

	cv::Mat totalframe = cv::Mat::zeros(_height * VERTICAL_FRAMES, _width * HORIZONTAL_FRAMES, CV_32FC1);
	cv::Mat reducedtotalframe = cv::Mat::zeros(_frames_height, _frames_width, CV_32FC1);

	for (int _i = 0; _i < _frames.size() - 1; _i++)
	{
		cv::Mat flow(_frame_size, CV_32FC1);

		cv::calcOpticalFlowFarneback(_frames[_i], _frames[_i + 1], flow, 0.5, 3, 15, 3, 5, 1.2, 0);

		// TODO: change uint8_t to float. and remove the variation limit.
		// visualization
		cv::Mat flow_parts[2];
		split(flow, flow_parts);

		cv::Mat upframe(_frame_size, CV_8UC1, cv::Scalar(0));
		cv::Mat downframe(_frame_size, CV_8UC1, cv::Scalar(0));
		cv::Mat leftframe(_frame_size, CV_8UC1, cv::Scalar(0));
		cv::Mat rightframe(_frame_size, CV_8UC1, cv::Scalar(0));

		for (int y = 0; y < _frame_size.height; y++)
			for (int x = 0; x < _frame_size.width; x++)
			{
				float upframefloat = _scaler * (std::abs(flow_parts[1].at<float>(y, x)) - flow_parts[1].at<float>(y, x)) / 2;
				float downframefloat = _scaler * (std::abs(flow_parts[1].at<float>(y, x)) + flow_parts[1].at<float>(y, x)) / 2;
				float leftframefloat = _scaler * (std::abs(flow_parts[0].at<float>(y, x)) - flow_parts[0].at<float>(y, x)) / 2;
				float rightframefloat = _scaler * (std::abs(flow_parts[0].at<float>(y, x)) + flow_parts[0].at<float>(y, x)) / 2;

				if (upframefloat > 255)
					upframefloat = 255;
				if (downframefloat > 255)
					downframefloat = 255;
				if (leftframefloat > 255)
					leftframefloat = 255;
				if (rightframefloat > 255)
					rightframefloat = 255;

				upframe.at<uint8_t>(y, x) = (uint8_t)(upframefloat);
				downframe.at<uint8_t>(y, x) = (uint8_t)(downframefloat);
				leftframe.at<uint8_t>(y, x) = (uint8_t)(leftframefloat);
				rightframe.at<uint8_t>(y, x) = (uint8_t)(rightframefloat);
			}

		for (int i = 0; i < _width; i++)
			for (int j = 0; j < _height; j++)
			{
				totalframe.at<float>(imagerow * _height + j, (4 * imagecolumn) * _width + i) = upframe.at<uint8_t>(j, i);
				totalframe.at<float>(imagerow * _height + j, (4 * imagecolumn + 1) * _width + i) = downframe.at<uint8_t>(j, i);
				totalframe.at<float>(imagerow * _height + j, (4 * imagecolumn + 2) * _width + i) = leftframe.at<uint8_t>(j, i);
				totalframe.at<float>(imagerow * _height + j, (4 * imagecolumn + 3) * _width + i) = rightframe.at<uint8_t>(j, i);
			}

		imagecolumn = imagecolumn + 1;
		if (imagecolumn > maxImagecolumn)
		{
			imagecolumn = 0;
			imagerow = imagerow + 1;
		}

		if (imagerow > maxImagerow)
		{
			imagerow = 0;

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

			break;
		}
	}
	in = out;
}