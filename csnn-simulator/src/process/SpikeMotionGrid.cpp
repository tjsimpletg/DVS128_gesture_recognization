#include "process/SpikeMotionGrid.h"

using namespace process;

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

//
//	SpikeMotionGrid
//
static RegisterClassParameter<SpikeMotionGrid, ProcessFactory> _register_1("SpikeMotionGrid");

SpikeMotionGrid::SpikeMotionGrid() : UniquePassProcess(_register_1), _width(0), _height(0), _depth(0), _conv_depth(0), _threshold(0)
{
	add_parameter("threshold", _threshold);
}
SpikeMotionGrid::SpikeMotionGrid(size_t threshold) : SpikeMotionGrid()
{
	parameter<size_t>("threshold").set(threshold);
}

void SpikeMotionGrid::process_train(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

void SpikeMotionGrid::process_test(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

Shape SpikeMotionGrid::compute_shape(const Shape &shape)
{
	_height = shape.dim(0);
	_width = shape.dim(1);
	_depth = shape.dim(2);
	_conv_depth = 1; // shape.dim(3);

	return Shape({_height, _width, _depth, _conv_depth});
}

void SpikeMotionGrid::_process(const std::string &label, Tensor<Time> &in) const
{
	Tensor<Time> t1(Shape({_height, _width, _depth, 1}));
	Tensor<Time> t2(Shape({_height, _width, _depth, 1}));
	Tensor<Time> out(Shape({_height, _width, _depth, 1}));

	// This function returns a list of frames that have gone through background subtraction.
	Tensor<Time>::split_time_tensor(t1, t2, in);

	size_t size = t1.shape().product();
	for (size_t i = 0; i < size; i++)
	{
		Time t = std::abs(t2.at_index(i) - t1.at_index(i)) > _threshold ? t2.at_index(i) - t1.at_index(i) : 0;
		out.at_index(i) = t;
	}

	in = out;
}