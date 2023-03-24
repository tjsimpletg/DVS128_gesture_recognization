#include "process/LateFusion.h"

using namespace process;

//
//	LateFusion
//
static RegisterClassParameter<LateFusion, ProcessFactory> _register_1("LateFusion");

LateFusion::LateFusion() : UniquePassProcess(_register_1),
						   _expName(""), _draw(0), _fused_frames_number(0), _width(0), _height(0), _depth(0), _conv_depth(0)
{
	add_parameter("draw", _draw);
	add_parameter("fused_frames_number", _fused_frames_number);
}

LateFusion::LateFusion(std::string expName, size_t draw, size_t fused_frames_number) : LateFusion()
{
	parameter<size_t>("draw").set(draw);
	_expName = expName;
	parameter<size_t>("fused_frames_number").set(fused_frames_number);
	_fused_frames_number = fused_frames_number;
	if (draw == 1)
	{
		std::filesystem::create_directories("Input_frames/" + _expName + "/LF/");
		_file_path = std::filesystem::current_path();
	}
}

void LateFusion::process_train(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

void LateFusion::process_test(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

// The -1 is because this process looses one frame by getting the differences of 2 frames.
Shape LateFusion::compute_shape(const Shape &shape)
{
	_height = shape.dim(0);
	_width = shape.dim(1);
	_depth = shape.dim(2);
	_conv_depth = shape.number() > 3 ? shape.dim(3) : 1;
	if (_fused_frames_number < 2)
	{
		throw std::runtime_error("Set the _fused_frames_number variable to a number >= 2 in your late fusion function");
	}
	return Shape({_height * _fused_frames_number, _width, _depth, 1});
}

void LateFusion::_process(const std::string &label, Tensor<InputType> &in) const
{
	// The output fused tensor
	Tensor<InputType> out(Shape({_height * _fused_frames_number, _width, _depth, _conv_depth / _fused_frames_number}));
	// Opencv cannot have a deph of 16
	for (int k = 0; k < _depth; k++)
	{
		// The frames to fuse.
		std::vector<cv::Mat> _frames;
		// Take a single filter input
		Tensor<InputType> in_prime(Shape({_height, _width, 1, _conv_depth}));
		for (int i = 0; i < _height; i++)
			for (int j = 0; j < _width; j++)
				for (int z = 0; z < _conv_depth; z++)
				{
					in_prime.at<float>(i, j, 0, z) = in.at<float>(i, j, k, z);
				}
		// This function returns the sequence of frames as Mats.
		Tensor<float>::tensor_to_matrices(_frames, in_prime);

		cv::Mat totalframe(_height * _fused_frames_number, _width, CV_32F);

		for (int _i = 0; _i < _frames.size(); _i++)
		{
			cv::Mat _frame = _frames[_i];
			for (int r = 0; r < _height; r++) // with each loop take a new line
				_frame.row(r).copyTo(totalframe.row(r * _frames.size() + _i));
		}
		// fused output for one single frame, will be added to out.
		Tensor<InputType> out_prime(Shape({_height * _fused_frames_number, _width, 1, 1}));

		Tensor<float>::matrix_to_tensor(totalframe, out_prime);

		if (_draw == 1)
			Tensor<float>::draw_tensor(_file_path + "/Input_frames/" + _expName + "/LF/LF " + label + std::to_string(k) + ".png", out_prime);

		for (int i = 0; i < _height; i++)
			for (int j = 0; j < _width; j++)
			{
				out.at<float>(i, j, k, 0) = out_prime.at<float>(i, j, 0, 0);
			}

		totalframe = cv::Mat::zeros(_height * _fused_frames_number, _width, CV_32F);
	}
	in = out;
}