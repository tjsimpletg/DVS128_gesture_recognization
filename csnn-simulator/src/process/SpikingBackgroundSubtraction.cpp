#include "process/SpikingBackgroundSubtraction.h"

using namespace process;

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

//
//	SpikingBackgroundSubtraction
//
static RegisterClassParameter<SpikingBackgroundSubtraction, ProcessFactory> _register_1("SpikingBackgroundSubtraction");

SpikingBackgroundSubtraction::SpikingBackgroundSubtraction() : UniquePassProcess(_register_1), _width(0), _height(0), _depth(0), _conv_depth(0), _expName(""), _method(0), _threshold(0)
{
	add_parameter("threshold", _threshold);
}
SpikingBackgroundSubtraction::SpikingBackgroundSubtraction(std::string expName, size_t method, size_t threshold) : SpikingBackgroundSubtraction()
{
	parameter<size_t>("threshold").set(threshold);
	_expName = expName;
	_method = method;
}

void SpikingBackgroundSubtraction::process_train(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

void SpikingBackgroundSubtraction::process_test(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

Shape SpikingBackgroundSubtraction::compute_shape(const Shape &shape)
{
	_height = shape.dim(0);
	_width = shape.dim(1);
	_depth = shape.dim(2);
	_conv_depth = _method == 0 ? shape.dim(3) * 2 : shape.dim(3);

	return Shape({_height, _width, _depth, _conv_depth});
}

void SpikingBackgroundSubtraction::_process(const std::string &label, Tensor<Time> &in) const
{
	Tensor<Time> out(Shape({_height, _width, _depth, _conv_depth}));

	// A tensor to recive the 3D time tensors
	std::vector<Tensor<Time>> time_tensors_in;
	std::vector<Tensor<Time>> time_tensors_out;
	// This function seperates one 4D time tensors into multiple 3D time tensors
	Tensor<Time>::time_tensor_to_tensors(in, time_tensors_in);

	time_tensors_in.push_back(time_tensors_in[0]);

	size_t size = time_tensors_in[0].shape().product();
	for (size_t _i = 0; _i < time_tensors_in.size() - 1; _i++)
	{
		Tensor<Time> t1 = time_tensors_in[_i];
		Tensor<Time> t2 = time_tensors_in[_i + 1];
		// The splitted positive and negative time changes
		Tensor<Time> t1_out(Shape({_height, _width, _depth, 1}));
		Tensor<Time> t2_out(Shape({_height, _width, _depth, 1}));
		if (_method == 0)
		{
			for (size_t i = 0; i < size; i++)
			{
				Time t = t2.at_index(i) - t1.at_index(i);

				if (t > 0)
				{
					t1_out.at_index(i) = t;
					t2_out.at_index(i) = INFINITE_TIME;
				}
				else if (t < 0)
				{
					t1_out.at_index(i) = INFINITE_TIME;
					t2_out.at_index(i) = -t;
				}
				else
				{
					t1_out.at_index(i) = INFINITE_TIME;
					t2_out.at_index(i) = INFINITE_TIME;
				}
			}
			time_tensors_out.push_back(t1_out);
			time_tensors_out.push_back(t2_out);
		}
		if (_method == 1)
		{
			for (size_t i = 0; i < size; i++)
			{
				Time t = t2.at_index(i) - t1.at_index(i);

				if (t > 0)
					t1_out.at_index(i) = t;
				else if (t < 0)
					t1_out.at_index(i) = std::abs(t);
				else
					t1_out.at_index(i) = INFINITE_TIME;
			}
			time_tensors_out.push_back(t1_out);
		}
	}
	Tensor<Time>::time_tensors_to_tensor(time_tensors_out, out);
	in = out;
}

// void SpikingBackgroundSubtraction::_process(const std::string &label, Tensor<Time> &in) const
// {
// 	Tensor<Time> t1(Shape({_height, _width, _depth, 1}));
// 	Tensor<Time> t2(Shape({_height, _width, _depth, 1}));

// 	Tensor<Time> t1_out(Shape({_height, _width, _depth, 1}));

// 	// This function returns a list of frames that have gone through background subtraction.
// 	Tensor<Time>::split_time_tensor(t1, t2, in);

// 	size_t size = t1.shape().product();
// 	for (size_t i = 0; i < size; i++)
// 	{
// 		Time t = t2.at_index(i) - t1.at_index(i);

// 		if (t != 0 && t1 != INFINITE_TIME && t2 != INFINITE_TIME)
// 		{
// 			t1_out.at_index(i) = t2.at_index(i);
// 		}
// 		else
// 		{
// 			t1_out.at_index(i) = INFINITE_TIME;
// 		}
// 	}

// 	in = t1_out;
// }

// void SpikingBackgroundSubtraction::_process(const std::string &label, Tensor<Time> &in) const
// {
// 	Tensor<Time> t1(Shape({_height, _width, _depth, 1}));
// 	Tensor<Time> t2(Shape({_height, _width, _depth, 1}));

// 	Tensor<Time> t1_out(Shape({_height, _width, _depth, 1}));
// 	Tensor<Time> t2_out(Shape({_height, _width, _depth, 1}));

// 	Tensor<Time> out(Shape({_height, _width, _depth, 2}));

// 	// This function returns a list of frames that have gone through background subtraction.
// 	Tensor<Time>::split_time_tensor(t1, t2, in);

// 	size_t size = t1.shape().product();
// 	for (size_t i = 0; i < size; i++)
// 	{
// 		Time t = t2.at_index(i) - t1.at_index(i);

// 		if (t > 0)
// 		{
// 			t1_out.at_index(i) = t2.at_index(i);
// 			t2_out.at_index(i) = INFINITE_TIME;
// 		}
// 		else if (t < 0)
// 		{
// 			t1_out.at_index(i) = INFINITE_TIME;
// 			t2_out.at_index(i) = t1.at_index(i);
// 		}
// 		else
// 		{
// 			t1_out.at_index(i) = INFINITE_TIME;
// 			t2_out.at_index(i) = INFINITE_TIME;
// 		}
// 	}

// 	Tensor<Time>::join_time_tensor(t1_out, t2_out, out);
// 	in = out;
// }