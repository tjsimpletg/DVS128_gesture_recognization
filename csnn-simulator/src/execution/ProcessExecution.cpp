#include "execution/ProcessExecution.h"
#include "Math.h"

ProcessExecution::ProcessExecution(ExperimentType &experiment) : _experiment(experiment), _train_set(), _test_set()
{
	_file_path = std::filesystem::current_path();
}

ProcessExecution::ProcessExecution(ExperimentType &experiment, bool save_features, bool draw_features) : _experiment(experiment), _save_features(save_features), _draw_features(draw_features), _train_set(), _test_set()
{
	_file_path = std::filesystem::current_path();
}

void ProcessExecution::process(size_t refresh_interval)
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

		_process_train_data(_experiment.process_at(i), _train_set, refresh_interval);
		_process_test_data(_experiment.process_at(i), _test_set);
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

Tensor<Time> ProcessExecution::compute_time_at(size_t i) const
{
	throw std::runtime_error("Unimplemented");
}

void ProcessExecution::_load_data()
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


void ProcessExecution::_process_train_data(AbstractProcess &process, std::vector<std::pair<std::string, SparseTensor<float>>> &data, size_t refresh_interval)
{
	size_t n = process.train_pass_number();


	for (size_t i = 0; i < n; i++)
	{

		size_t total_size = 0;
		size_t total_capacity = 0;

		for (size_t j = 0; j < data.size(); j++)
		{
			Tensor<float> current = from_sparse_tensor(data[j].second);
			process.process_train_sample(_experiment.name() + ";." + std::to_string(process.index()) + ";." + data[j].first, current, i, j, data.size());
			data[j].second = to_sparse_tensor(current);

			total_size += data[j].second.values().size();
			total_capacity += data[j].second.values().size();

			if (j % 10000 == 10000 - 1)
			{
				std::cout << static_cast<double>(total_size) / 10000.0 << "/" << static_cast<double>(total_capacity) / 10000.0 << std::endl;
				total_size = 0;
				total_capacity = 0;
			}

			if (i == n - 1 && data[j].second.shape() != process.shape() && process.class_name() != "LateFusion")
			{
				throw std::runtime_error("Unexpected shape (actual: " + data[j].second.shape().to_string() + ", expected at " + process.class_name() + ": " + process.shape().to_string() + ")");
			}

			_experiment.tick(process.index(), i * data.size() + j);

			if ((i * data.size() + j) % refresh_interval == 0)
			{
				_experiment.refresh(process.index());
			}
		}
	}
}

void ProcessExecution::_process_test_data(AbstractProcess &process, std::vector<std::pair<std::string, SparseTensor<float>>> &data)
{
	for (size_t j = 0; j < data.size(); j++)
	{
		Tensor<float> current = from_sparse_tensor(data[j].second);
		process.process_test_sample(data[j].first, current, j, data.size());
		data[j].second = to_sparse_tensor(current);

		if (data[j].second.shape() != process.shape() && process.class_name() != "LateFusion")
		{
			throw std::runtime_error("Unexpected shape (actual: " + data[j].second.shape().to_string() + ", expected: " + process.shape().to_string() + ")");
		}
	}
}

void ProcessExecution::_process_output(size_t index)
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
				_process_train_data(*process, output_train_set, std::numeric_limits<size_t>::max());
				_process_test_data(*process, output_test_set);
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