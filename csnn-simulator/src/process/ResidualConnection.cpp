#include "process/ResidualConnection.h"

using namespace process;

static RegisterClassParameter<ResidualConnection, ProcessFactory> _register("ResidualConnection");

ResidualConnection::ResidualConnection() : UniquePassProcess(_register), _width(0), _height(0), _depth(0), _conv_depth(0), _layer_name("")
{
}

ResidualConnection::ResidualConnection(std::string exp_name, std::string layer_name) : ResidualConnection()
{
	// Get the saved input data
	_file_path = std::filesystem::current_path();
	_layer_name = layer_name;

	if (layer_name.empty())
		_file_path = _file_path + "/ResInput/" + exp_name;
	else
		_file_path = _file_path + "/ResInput/" + exp_name + "-" + layer_name;

	_data_list.push_back(_file_path + "_train.json");
	_data_list.push_back(_file_path + "_test.json");
}

Shape ResidualConnection::compute_shape(const Shape &shape)
{
	_width = shape.dim(0);
	_height = shape.dim(1);
	_depth = shape.dim(2);
	_conv_depth = shape.number() > 3 ? shape.dim(3) : 1;

	return Shape({_width, _height, _depth, _conv_depth});
}

void ResidualConnection::process_train(const std::string &, Tensor<float> &sample)
{
	// load original input.
	if (_train_set.empty())
	{
		LoadPairVector(_data_list[0], _train_set);
		_train_res_sample_count = 0;
	}

	_train_res_sample_count++;
	_process(sample, _train_set[_train_res_sample_count - 1].second);
	//draw_progress(_train_res_sample_count, get_train_count());
}

void ResidualConnection::process_test(const std::string &, Tensor<float> &sample)
{
	// load original input.
	if (_test_set.empty())
	{
		LoadPairVector(_data_list[1], _test_set);
		_test_res_sample_count = 0;
	}

	_test_res_sample_count++;
	_process(sample, _test_set[_test_res_sample_count - 1].second);
	//draw_progress(_test_res_sample_count, get_test_count());
}

void ResidualConnection::_process(Tensor<float> &in, Tensor<float> &original_sample) const
{
	Tensor<float> out = Tensor<float>(Shape({_width, _height, _depth, _conv_depth}));

	for (size_t x = 0; x < _width; x++)
		for (size_t y = 0; y < _height; y++)
			for (size_t z = 0; z < _depth; z++)
				for (size_t k = 0; k < _conv_depth; k++)
				{
					out.at(x, y, z, k) = std::max(in.at(x, y, z, k), original_sample.at(x, y, 0, k));
				}
	in = out;
}