#ifndef _PROCESS_EXECUTION_H
#define _PROCESS_EXECUTION_H

#include "tool/Operations.h"
#include "SparseTensor.h"
#include "Experiment.h"
#include "SpikeConverter.h"
// #include "include/dataset/Image.h"
/**
 * @brief ProcessExecution Is the exicution policy that manages the sequential exicution of the functions declared in the expirements of the apps folder.
 * It runs the pre-processing functions and uses functions such as _process_sample that starts the traning of the layers, this function calls the train function in the convolution class.
 * It also calls functions such as _update_data, that takes the resulting information from the training, converts the information back into float values from spikes, and trains/tests the SVM with this information.
 * @param experiment The application that takes three main parameters of it's own (argc, argv, name)
 * @param save_features A flag that saves the extracted features in a .json file in the build folder, this flag is needed when using Two-stream methods.
 * @param draw_features A flag that draws the extracted features in the build folder.
 */
class ProcessExecution
{

public:
	typedef Experiment<ProcessExecution> ExperimentType;

	ProcessExecution(ExperimentType &experiment);
	ProcessExecution(ExperimentType &experiment, bool save_features, bool draw_features);
	ProcessExecution(ExperimentType &experiment,bool save_input, bool save_features, bool draw_features);

	void process(size_t refresh_interval);

	Tensor<Time> compute_time_at(size_t i) const;

private:
	void _prepare(size_t layer_target_index);
	void _process_sample(const std::string &label, const std::vector<Spike> &in, size_t layer_index);
	void _update_data(size_t layer_index, size_t refresh_interval);

	void _load_data();

	void _process_train_data(AbstractProcess &process, std::vector<std::pair<std::string, SparseTensor<float>>> &data, size_t refresh_interval);
	void _process_test_data(AbstractProcess &process, std::vector<std::pair<std::string, SparseTensor<float>>> &data);
	void _set_temporal_depth(AbstractProcess const &process, std::vector<std::pair<std::string, SparseTensor<float>>> &data);
	void _process_output(size_t index);

	ExperimentType &_experiment;
	bool _save_input;
	bool _save_features;
	bool _draw_features;
	std::string _file_path;

	std::vector<std::pair<std::string, SparseTensor<float>>> _train_set;
	std::vector<std::pair<std::string, SparseTensor<float>>> _test_set;
};

#endif
