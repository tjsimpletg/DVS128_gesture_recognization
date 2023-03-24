#include "process/ReichardtDetector.h"

using namespace process;

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

//
//	ReichardtDetector
//
static RegisterClassParameter<ReichardtDetector, ProcessFactory> _register_1("ReichardtDetector");

ReichardtDetector::ReichardtDetector() : UniquePassProcess(_register_1), _expName(""), _draw(0), _scaler(0), _width(0), _height(0), _depth(0), _conv_depth(0)
{
	add_parameter("draw", _draw);
}

ReichardtDetector::ReichardtDetector(std::string expName, size_t draw, size_t scaler) : ReichardtDetector()
{
	parameter<size_t>("draw").set(draw);
	_expName = expName;

	if (draw == 1)
		std::filesystem::create_directories("Input_frames/" + _expName + "/RD/");

	_file_path = std::filesystem::current_path();
}

void ReichardtDetector::process_train(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

void ReichardtDetector::process_test(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

// The -1 is because this process looses one frame by getting the differences of 2 frames.
Shape ReichardtDetector::compute_shape(const Shape &shape)
{
	_height = shape.dim(0);
	_width = shape.dim(1);
	_depth = 4; // bacuse e take 4 directions
	_conv_depth = shape.dim(3);
	return Shape({_height, _width, _depth, _conv_depth});
}

void ReichardtDetector::_process(const std::string &label, Tensor<InputType> &in) const
{
	std::vector<cv::Mat> _frames;
	std::vector<Tensor<InputType>> _out_frames;

	Tensor<InputType> out(Shape({_height, _width, _depth, _conv_depth}));

	cv::Mat shift_left_current_gray_frame, current_gray_frame, reichardt_left, shift_left_prev_gray_frame, prev_gray_frame, reichardt_right, reichardt_full_h;
	cv::Mat shift_up_current_gray_frame, reichardt_left1, shift_up_prev_gray_frame, reichardt_right1, reichardt_full_v;
	cv::Mat shift_diag1_current_gray_frame, reichardt_left2, shift_diag1_prev_gray_frame, reichardt_right2, reichardt_full_d1;
	cv::Mat shift_diag2_current_gray_frame, reichardt_left3, shift_diag2_prev_gray_frame, reichardt_right3, reichardt_full_d2;

	// This function returns a list of frames that have gone through background subtraction.
	Tensor<float>::tensor_to_matrices(_frames, in);
	_frames.push_back(_frames[_frames.size() - 1]);

	cv::Size _frame_size(_width, _height);

	for (int _i = 0; _i < _frames.size() - 1; _i++)
	{
		std::vector<cv::Mat> _direction_frames;
		Tensor<InputType> _direction_out(Shape({_height, _width, _depth, 1}));

		prev_gray_frame = _frames[_i] / 255;
		current_gray_frame = _frames[_i + 1] / 255;
		// -----------Left - Right direction---------
		cv::copyMakeBorder(current_gray_frame, shift_left_current_gray_frame, 0, 0, 1, 0, cv::BORDER_CONSTANT, cv::Scalar(0));
		// Crop the full image to that image contained by the rectangle cropsizeRect
		cv::Rect cropsizeRect(0, 0, shift_left_current_gray_frame.cols - 1, shift_left_current_gray_frame.rows);
		shift_left_current_gray_frame = shift_left_current_gray_frame(cropsizeRect);
		cv::multiply(prev_gray_frame, shift_left_current_gray_frame, reichardt_left);

		cv::copyMakeBorder(prev_gray_frame, shift_left_prev_gray_frame, 0, 0, 1, 0, cv::BORDER_CONSTANT, cv::Scalar(0));
		// Crop the full image to that image contained by the rectangle cropsizeRect
		shift_left_prev_gray_frame = shift_left_prev_gray_frame(cropsizeRect);
		cv::multiply(current_gray_frame, shift_left_prev_gray_frame, reichardt_right);

		// Horizontal movement
		reichardt_full_h = reichardt_left - reichardt_right;

		// -----------Up - Down direction---------
		cv::copyMakeBorder(current_gray_frame, shift_up_current_gray_frame, 1, 0, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0));
		cv::Rect cropsizeRect1(0, 0, shift_up_current_gray_frame.cols, shift_up_current_gray_frame.rows - 1);
		shift_up_current_gray_frame = shift_up_current_gray_frame(cropsizeRect1);
		cv::multiply(prev_gray_frame, shift_up_current_gray_frame, reichardt_left1);

		cv::copyMakeBorder(prev_gray_frame, shift_up_prev_gray_frame, 1, 0, 0, 0, cv::BORDER_CONSTANT, cv::Scalar(0));
		shift_up_prev_gray_frame = shift_up_prev_gray_frame(cropsizeRect1);
		cv::multiply(current_gray_frame, shift_up_prev_gray_frame, reichardt_right1);

		// Vertical movement
		reichardt_full_v = reichardt_left1 - reichardt_right1;

		// -----------Up right diagonal direction---------
		cv::copyMakeBorder(current_gray_frame, shift_diag1_current_gray_frame, 1, 0, 1, 0, cv::BORDER_CONSTANT, cv::Scalar(0));
		cv::Rect cropsizeRect2(0, 0, shift_diag1_current_gray_frame.cols - 1, shift_diag1_current_gray_frame.rows - 1);
		shift_diag1_current_gray_frame = shift_diag1_current_gray_frame(cropsizeRect2);
		cv::multiply(prev_gray_frame, shift_diag1_current_gray_frame, reichardt_left2);

		cv::copyMakeBorder(prev_gray_frame, shift_diag1_prev_gray_frame, 1, 0, 1, 0, cv::BORDER_CONSTANT, cv::Scalar(0));
		shift_diag1_prev_gray_frame = shift_diag1_prev_gray_frame(cropsizeRect2);
		cv::multiply(current_gray_frame, shift_diag1_prev_gray_frame, reichardt_right2);

		// Diagonal movement
		reichardt_full_d1 = reichardt_left2 - reichardt_right2;

		// -----------Down right diagonal direction---------
		cv::copyMakeBorder(current_gray_frame, shift_diag2_current_gray_frame, 0, 1, 0, 1, cv::BORDER_CONSTANT, cv::Scalar(0));
		cv::Rect cropsizeRect3(1, 1, shift_diag2_current_gray_frame.cols - 1, shift_diag2_current_gray_frame.rows - 1);
		shift_diag2_current_gray_frame = shift_diag2_current_gray_frame(cropsizeRect3);
		cv::multiply(prev_gray_frame, shift_diag2_current_gray_frame, reichardt_left3);

		cv::copyMakeBorder(prev_gray_frame, shift_diag2_prev_gray_frame, 0, 1, 0, 1, cv::BORDER_CONSTANT, cv::Scalar(0));
		shift_diag2_prev_gray_frame = shift_diag2_prev_gray_frame(cropsizeRect3);
		cv::multiply(current_gray_frame, shift_diag2_prev_gray_frame, reichardt_right3);

		// Diagonal movement
		reichardt_full_d2 = reichardt_left3 - reichardt_right3;

		// reichardt_full_h, reichardt_full_v, reichardt_full_d1, reichardt_full_d2
		_direction_frames.push_back(reichardt_full_h);
		_direction_frames.push_back(reichardt_full_v);
		_direction_frames.push_back(reichardt_full_d1);
		_direction_frames.push_back(reichardt_full_d2);

		Tensor<float>::direction_matrices_to_tensor(_direction_frames, _direction_out);

		_out_frames.push_back(_direction_out);
	}

	Tensor<float>::tensors_to_tensor(_out_frames, out);

	if (_draw == 1)
	{
		Tensor<float>::draw_tensor(_file_path + "/Input_frames/" + _expName + "/RD/RD_h_" + label + "_", _out_frames[0]);
		Tensor<float>::draw_tensor(_file_path + "/Input_frames/" + _expName + "/RD/RD_v_" + label + "_", _out_frames[1]);
		Tensor<float>::draw_tensor(_file_path + "/Input_frames/" + _expName + "/RD/RD_d1_" + label + "_", _out_frames[2]);
		Tensor<float>::draw_tensor(_file_path + "/Input_frames/" + _expName + "/RD/RD_d2_" + label + "_", _out_frames[3]);
	}

	in = out;
}