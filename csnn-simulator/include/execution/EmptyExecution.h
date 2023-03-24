#ifndef _EMPTY_EXECUTION_H
#define _EMPTY_EXECUTION_H

#include "tool/Operations.h"
#include "SparseTensor.h"
#include "Experiment.h"
#include "SpikeConverter.h"
// #include "include/dataset/Image.h"
/**
 * @brief EmptyExecution Is the execution policy that manages the sequential exicution of the functions declared in the expirements of the apps folder.
 * It runs the pre-processing functions and uses functions such as _process_sample that starts the traning of the layers, this function calls the train function in the convolution class.
 * It also calls functions such as _update_data, that takes the resulting information from the training, converts the information back into float values from spikes, and trains/tests the SVM with this information.
 * @param experiment The name of the experiment
 * @param save_features A flag that saves the extracted features in a .json file
 * @param draw_features A flag that draws the extraxcted features 
 */
class EmptyExecution
{

public:
	typedef Experiment<EmptyExecution> ExperimentType;

	EmptyExecution(ExperimentType &experiment);
	EmptyExecution(ExperimentType &experiment, bool save_features, bool draw_features);

	void process(size_t refresh_interval);
	Tensor<Time> compute_time_at(size_t i) const;

private:

	void _load_data();
	void _process_output(size_t index);

	ExperimentType &_experiment;
	bool _save_features;
	bool _draw_features;
	std::string _file_path;

	std::vector<std::pair<std::string, SparseTensor<float>>> _train_set;
	std::vector<std::pair<std::string, SparseTensor<float>>> _test_set;
};

#endif
