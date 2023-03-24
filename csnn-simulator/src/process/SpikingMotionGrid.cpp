#include "process/SpikingMotionGrid.h"

using namespace process;

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>

//
//	SpikingMotionGrid
//
static RegisterClassParameter<SpikingMotionGrid, ProcessFactory> _register_1("SpikingMotionGrid");

SpikingMotionGrid::SpikingMotionGrid() : UniquePassProcess(_register_1), _width(0), _height(0), _depth(0), _conv_depth(0), _mg_vertical_frames(0), _mg_horizontal_frames(0), _scaler(0), _expName(""), _threshold(0)
{
	add_parameter("threshold", _threshold);
	add_parameter("mg_vertical_frames", _mg_vertical_frames);
	add_parameter("mg_horizontal_frames", _mg_horizontal_frames);
	add_parameter("scaler", _scaler);
}
SpikingMotionGrid::SpikingMotionGrid(std::string expName, size_t threshold, size_t mg_vertical_frames, size_t mg_horizontal_frames, size_t scaler) : SpikingMotionGrid()
{
	parameter<size_t>("threshold").set(threshold);

	parameter<size_t>("mg_vertical_frames").set(mg_vertical_frames);
	parameter<size_t>("mg_horizontal_frames").set(mg_horizontal_frames);
	parameter<size_t>("scaler").set(scaler);

	_mg_vertical_frames = mg_vertical_frames;
	_mg_horizontal_frames = mg_horizontal_frames;
	_scaler = scaler;
	_expName = expName;
}

void SpikingMotionGrid::process_train(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

void SpikingMotionGrid::process_test(const std::string &label, Tensor<float> &sample)
{
	_process(label, sample);
}

Shape SpikingMotionGrid::compute_shape(const Shape &shape)
{
	_height = shape.dim(0);
	_width = shape.dim(1);
	_depth = shape.dim(2);
	_conv_depth = shape.dim(3);

	return Shape({_height * _mg_vertical_frames, _width * _mg_horizontal_frames, _depth, 1});
}

void SpikingMotionGrid::_process(const std::string &label, Tensor<Time> &in) const
{

	int imagerow = 0;
	int imagecolumn = 0;
	float SCALER = _scaler;

	int VERTICAL_FRAMES = _mg_vertical_frames;
	int HORIZONTAL_FRAMES = _mg_horizontal_frames;
	int maxImagecolumn = ((HORIZONTAL_FRAMES / 4) - 1);
	int maxImagerow = VERTICAL_FRAMES - 1;

	Tensor<InputType> totalframe(Shape({_height * VERTICAL_FRAMES, _width * HORIZONTAL_FRAMES, _depth, 1}));
	Tensor<InputType> out(Shape({_height * VERTICAL_FRAMES, _width * HORIZONTAL_FRAMES, _depth, 1}));

	for (int _i = 0; _i < in.shape().dim(3); _i++)
	{
		Tensor<Time> t1(Shape({_height, _width, _depth, 1}));
		Tensor<Time> t2(Shape({_height, _width, _depth, 1}));

		Tensor<Time> t1_out(Shape({_height, _width, _depth, 1}));
		Tensor<Time> t2_out(Shape({_height, _width, _depth, 1}));

		for (size_t i = 0; i < _height; i++)
			for (size_t j = 0; j < _width; j++)
				for (size_t k = 0; k < _depth; k++)
				{
					t1.at(i, j, k, 0) = in.at(i, j, k, _i);
					t2.at(i, j, k, 0) = in.at(i, j, k, _i + 1);
				}

		size_t size = t1.shape().product();
		for (size_t i = 0; i < size; i++)
		{
			Time t = t2.at_index(i) - t1.at_index(i);

			if (t > 0)
			{
				t1_out.at_index(i) = t2.at_index(i);
				t2_out.at_index(i) = INFINITE_TIME;
			}
			else if (t < 0)
			{
				t1_out.at_index(i) = INFINITE_TIME;
				t2_out.at_index(i) = t2.at_index(i);
			}
			else
			{
				t1_out.at_index(i) = INFINITE_TIME;
				t2_out.at_index(i) = INFINITE_TIME;
			}
		}
		for (int i = 0; i < _width; i++)
			for (int j = 0; j < _height; j++)
			{
				totalframe.at<Time>(imagerow * _height + j, (4 * imagecolumn) * _width + i) = t1_out.at<Time>(j, i);
				totalframe.at<Time>(imagerow * _height + j, (4 * imagecolumn + 1) * _width + i) = t2_out.at<Time>(j, i);
				totalframe.at<Time>(imagerow * _height + j, (4 * imagecolumn + 2) * _width + i) = t1_out.at<Time>(j, i);
				totalframe.at<Time>(imagerow * _height + j, (4 * imagecolumn + 3) * _width + i) = t2_out.at<Time>(j, i);
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
			out = totalframe;
			break;
		}
	}
	in = out;
}