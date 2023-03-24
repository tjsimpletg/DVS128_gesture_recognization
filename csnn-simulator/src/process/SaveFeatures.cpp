#include "process/SaveFeatures.h"

using namespace process;

static RegisterClassParameter<SaveFeatures, ProcessFactory> _register("SaveFeatures");

SaveFeatures::SaveFeatures() : UniquePassProcess(_register), _width(0), _height(0), _depth(0), _conv_depth(0)
{
}

SaveFeatures::SaveFeatures(std::string exp_name, std::string layer_name) : SaveFeatures()
{
	_exp_name = exp_name;
	_layer_name = layer_name;
	std::filesystem::create_directories("SaveFeatures/" + _exp_name + "/");
	_file_path = std::filesystem::current_path();
}

Shape SaveFeatures::compute_shape(const Shape &shape)
{
	_width = shape.dim(0);
	_height = shape.dim(1);
	_depth = shape.dim(2);
	_conv_depth = shape.number() > 3 ? shape.dim(3) : 1;

	return Shape({_width, _height, _depth, _conv_depth});
}

void SaveFeatures::process_train(const std::string &label, Tensor<float> &sample)
{
	std::string delimiter = ";.";
	std::string _label = label;
	if (label.find(delimiter) != std::string::npos)
	{
		_label.erase(0, _exp_name.length() + delimiter.length());
		std::string _layerIndex = _label.substr(0, _label.find(delimiter));
		_label.erase(0, _layerIndex.length() + delimiter.length());
	}
	if (_train_save_sample_count == 0)
		_general_shape = sample.shape().dim(0);
	else if (_general_shape != sample.shape().dim(0))
	{
		_general_shape = sample.shape().dim(0);
		_train_save_sample_count = 0;
		_test_save_sample_count = 0;
	}

	_train_save_sample_count++; // a counter for the progress bar
	draw_progress(_train_save_sample_count, get_train_count());
	SaveFeature(_file_path + "/SaveFeatures/" + _exp_name + "/" + _exp_name + "_" + _layer_name + "_train.json", _label, sample, _train_save_sample_count, get_train_count());
}

void SaveFeatures::process_test(const std::string &label, Tensor<float> &sample)
{
	_test_save_sample_count++; // a counter for the progress bar
	draw_progress(_test_save_sample_count, get_test_count());
	SaveFeature(_file_path + "/SaveFeatures/" + _exp_name + "/" + _exp_name + "_" + _layer_name + "_test.json", label, sample, _test_save_sample_count, get_test_count());
}

void SaveFeatures::_process(Tensor<float> &in) const
{
}