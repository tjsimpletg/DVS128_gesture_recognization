#include "execution/FusedExecution.h"
#include "Math.h"

FusedExecution::FusedExecution(ExperimentType &experiment) : _experiment(experiment), _train_set(), _test_set()
{
	_file_path = std::filesystem::current_path();
}

FusedExecution::FusedExecution(ExperimentType &experiment, bool save_features, bool draw_features) : _experiment(experiment), _save_features(save_features), _draw_features(draw_features), _train_set(), _test_set()
{
	_file_path = std::filesystem::current_path();
}

void FusedExecution::process(size_t refresh_interval)
{
	_load_data();

	std::vector<size_t> train_index;
	for (size_t i = 0; i < _train_set.size(); i++)
	{
		train_index.push_back(i);
	}

	for (size_t i = 0; i < _experiment.process_number(); i++)
	{
		_experiment.print() << "Process " << _experiment.process_at(i).factory_name() << "." << _experiment.process_at(i).class_name();
		if (!_experiment.process_at(i).name().empty())
		{
			_experiment.print() << " (" << _experiment.process_at(i).name() << ")";
		}
		_experiment.print() << std::endl;

		_process_output(i);
	}
	try
	{
		_train_set.clear();
		_test_set.clear();
	}
	catch (std::string ex)
	{
		std::cout << "hmm.. clearing the dataset didn't work" << std::endl;
	}
}

Tensor<Time> FusedExecution::compute_time_at(size_t i) const
{
	throw std::runtime_error("Unimplemented");
}

void FusedExecution::_load_data()
{
	for (Input *input : _experiment.train_data())
	{
		size_t count = 0;
		while (input->has_next())
		{
			auto entry = input->next();
			_train_set.emplace_back(entry.first, to_sparse_tensor(entry.second));
			count++;
		}
		_experiment.log() << "Load " << count << " train samples from " << input->to_string() << std::endl;
		input->close();
	}

	for (Input *input : _experiment.test_data())
	{
		size_t count = 0;
		while (input->has_next())
		{
			auto entry = input->next();
			_test_set.emplace_back(entry.first, to_sparse_tensor(entry.second));
			count++;
		}
		_experiment.log() << "Load " << count << " test samples from " << input->to_string() << std::endl;
		input->close();
	}
}

void FusedExecution::_process_output(size_t index)
{
	for (size_t i = 0; i < _experiment.output_count(); i++)
	{
		if (_experiment.output_at(i).index() == index)
		{
			Output &output = _experiment.output_at(i);

			std::cout << "Output " << output.name() << std::endl;

			std::vector<std::pair<std::string, SparseTensor<float>>> output_train_set;
			std::vector<std::pair<std::string, SparseTensor<float>>> output_test_set;

			for (std::pair<std::string, SparseTensor<float>> &entry : _train_set)
			{
				Tensor<float> current = from_sparse_tensor(entry.second);
				output_train_set.emplace_back(entry.first, to_sparse_tensor(output.converter().process(current)));
			}

			for (std::pair<std::string, SparseTensor<float>> &entry : _test_set)
			{
				Tensor<float> current = from_sparse_tensor(entry.second);
				output_test_set.emplace_back(entry.first, to_sparse_tensor(output.converter().process(current)));
			}

			for (Process *process : output.postprocessing())
			{
				_experiment.print() << "Process " << process->class_name() << std::endl;
			}

			std::string _mainExpName = _experiment.name();
			if (_mainExpName.find("fused") != std::string::npos)
			{
				size_t _loc = _mainExpName.find("_fused");
				_mainExpName = _mainExpName.erase(_loc, 6);
			}

			if (_save_features || _draw_features)
			{
				std::filesystem::create_directories(_file_path + "/ExtractedFeatures/" + _mainExpName + "/train/");
				std::filesystem::create_directories(_file_path + "/ExtractedFeatures/" + _mainExpName + "/test/");
				std::filesystem::create_directories(_file_path + "/ExtractedFeatures/" + _mainExpName + "/Fused_Result/train/");
				std::filesystem::create_directories(_file_path + "/ExtractedFeatures/" + _mainExpName + "/Fused_Result/test/");
			}
			if (_save_features)
			{
				SavePairVector(_file_path + "/ExtractedFeatures/" + _mainExpName + "/Fused_Result/train/" + _experiment.name() + ".json", output_train_set);
				SavePairVector(_file_path + "/ExtractedFeatures/" + _mainExpName + "/Fused_Result/test/" + _experiment.name() + ".json", output_test_set);
			}
			if (_draw_features)
			{
				DrawSparseFeatures(_file_path + "/ExtractedFeatures/" + _mainExpName + "/Fused_Result/train/" + _experiment.name() + "_", output_train_set);
				DrawSparseFeatures(_file_path + "/ExtractedFeatures/" + _mainExpName + "/Fused_Result/test/" + _experiment.name() + "_", output_test_set);
			}

			for (Analysis *analysis : output.analysis())
			{
				_experiment.log() << output.name() << ", analysis " << analysis->class_name() << ":" << std::endl;

				size_t n = analysis->train_pass_number();

				for (size_t i = 0; i < n; i++)
				{
					analysis->before_train_pass(i);
					for (std::pair<std::string, SparseTensor<float>> &entry : output_train_set)
					{
						Tensor<float> current = from_sparse_tensor(entry.second);
						analysis->process_train_sample(entry.first, current, i);
					}
					analysis->after_train_pass(i);
				}

				if (n == 0)
				{
					analysis->after_test();
				}
				else
				{
					analysis->before_test();
					// In the old simulator version, late fusion happens here.
					for (std::pair<std::string, SparseTensor<float>> &entry : output_test_set)
					{
						Tensor<float> current = from_sparse_tensor(entry.second);
						analysis->process_test_sample(entry.first, current);
					}
					analysis->after_test();
				}
			}
		}
	}
}