#include "execution/SparseIntermediateExecutionNew.h"
#include "Math.h"

SparseIntermediateExecutionNew::SparseIntermediateExecutionNew(ExperimentType &experiment) : _experiment(experiment), _train_set(), _test_set()
{
	_file_path = std::filesystem::current_path();
}

SparseIntermediateExecutionNew::SparseIntermediateExecutionNew(ExperimentType &experiment, bool allow_residual_connections, bool save_features, bool save_timestamps, bool draw_features) : _experiment(experiment), _allow_residual_connections(allow_residual_connections), _save_features(save_features), _save_timestamps(save_timestamps), _draw_features(draw_features), _train_set(), _test_set()
{
	_file_path = std::filesystem::current_path();
}

void SparseIntermediateExecutionNew::process(size_t refresh_interval)
{
	_load_data();
	if (_allow_residual_connections == true)
	{
		std::filesystem::create_directories(_file_path + "/ResInput/");
		SaveInputPairVector(_file_path + "/ResInput/" + _experiment.name() + "_train.json", _train_set);
		SaveInputPairVector(_file_path + "/ResInput/" + _experiment.name() + "_test.json", _test_set);
	}
	std::vector<size_t> train_index;
	for (size_t i = 0; i < _train_set.size(); i++)
	{
		train_index.push_back(i);
	}

	for (size_t i = 0; i < _experiment.process_number(); i++)
	{
		auto start = std::chrono::system_clock::now();

		_experiment.print() << "Process " << _experiment.process_at(i).factory_name() << "." << _experiment.process_at(i).class_name();
		if (!_experiment.process_at(i).name().empty())
		{
			_experiment.print() << " (" << _experiment.process_at(i).name() << ")";
		}
		_experiment.print() << std::endl;

		_process_train_data(_experiment.process_at(i), _train_set, refresh_interval);
		_process_test_data(_experiment.process_at(i), _test_set);
		_process_output(i);

		auto end = std::chrono::system_clock::now();

		std::chrono::duration<double> elapsed_seconds = end - start;
		std::cout << "--------------" + _experiment.process_at(i).name() + " time: ";
		std::cout << elapsed_seconds.count() << std::endl;
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

Tensor<Time> SparseIntermediateExecutionNew::compute_time_at(size_t i) const
{
	throw std::runtime_error("Unimplemented");
}

void SparseIntermediateExecutionNew::_load_data()
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

void SparseIntermediateExecutionNew::_set_temporal_depth(AbstractProcess const &process, std::vector<std::pair<std::string, SparseTensor<float>>> &data)
{
	std::vector<std::pair<std::string, SparseTensor<float>>> joined_train_set;

	size_t sample_join_buffer = 0;
	size_t _height = data[0].second.shape().dim(0);
	size_t _width = data[0].second.shape().dim(1);
	size_t _depth = data[0].second.shape().dim(2);
	size_t _conv_depth = data[0].second.shape().dim(3);
	size_t _temporal_depth = process.shape().dim(3);

	Tensor<float> _sample_buffer = Tensor<float>(Shape({_height, _width, _depth, _temporal_depth}));

	for (std::pair<std::string, SparseTensor<float>> &entry : data)
	{
		Tensor<float> current = from_sparse_tensor(entry.second);

		if (_temporal_depth > 1 && entry.second.shape().number() > 3) // 3D convolution
		{
			for (int i = 0; i < _height; i++)
				for (int j = 0; j < _width; j++)
					for (int z = 0; z < _depth; z++)
						for (int k = 0; k < _conv_depth; k++)
						{
							_sample_buffer.at(i, j, z, sample_join_buffer) = current.at(i, j, z, k);
						}
			sample_join_buffer++;

			if (sample_join_buffer == _temporal_depth)
			{
				joined_train_set.emplace_back(entry.first, to_sparse_tensor(_sample_buffer));
				_sample_buffer = Tensor<float>(Shape({_height, _width, _depth, _temporal_depth}));
				sample_join_buffer = 0;
			}
		}
	}

	data = joined_train_set;
}

void SparseIntermediateExecutionNew::_process_train_data(AbstractProcess &process, std::vector<std::pair<std::string, SparseTensor<float>>> &data, size_t refresh_interval)
{
	size_t n = process.train_pass_number();

	if (n == 0)
	{
		throw std::runtime_error("train_pass_number() should be > 0");
	}
	// during training, n = epochs
	for (size_t i = 0; i < n; i++)
	{

		size_t total_size = 0;
		size_t total_capacity = 0;
		if (process.class_name() == "SetTemporalDepth")
			_set_temporal_depth(process, data);

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

void SparseIntermediateExecutionNew::_process_test_data(AbstractProcess &process, std::vector<std::pair<std::string, SparseTensor<float>>> &data)
{
	if (process.class_name() == "SetTemporalDepth")
		_set_temporal_depth(process, data);

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

void SparseIntermediateExecutionNew::_process_output(size_t index)
{
	for (size_t i = 0; i < _experiment.output_count(); i++)
	{
		if (_experiment.output_at(i).index() == index)
		{
			Output &output = _experiment.output_at(i);

			std::cout << "Output " << output.name() << std::endl;

			std::vector<std::pair<std::string, SparseTensor<float>>> output_train_set;
			std::vector<std::pair<std::string, SparseTensor<float>>> output_test_set;

			std::string _mainExpName = _experiment.name();
			if (_save_timestamps)
			{
				std::filesystem::create_directories(_file_path + "/ExtractedTimestamps/" + _mainExpName + "/test/");
				std::filesystem::create_directories(_file_path + "/ExtractedTimestamps/" + _mainExpName + "/train/");
				SavePairVector(_file_path + "/ExtractedTimestamps/" + _mainExpName + "/train/" + _experiment.name() + "_timestamps.json", _train_set);
				SavePairVector(_file_path + "/ExtractedTimestamps/" + _mainExpName + "/test/" + _experiment.name() + "_timestamps.json", _test_set);
			}

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
				if (process->class_name() == "ResidualConnection" && _allow_residual_connections != true)
				{
					throw std::runtime_error("You need to set the _allow_residual_connections flag to true in the experiment in order to use residual connections.");
				}
				_experiment.print() << "Process " << process->class_name() << std::endl;
				_process_train_data(*process, output_train_set, std::numeric_limits<size_t>::max());
				_process_test_data(*process, output_test_set);
			}

			if (_mainExpName.find("time") != std::string::npos)
			{
				size_t _loc = _mainExpName.find("_time");
				_mainExpName = _mainExpName.erase(_loc, 5);
			}

			if (_save_features)
			{
				std::filesystem::create_directories(_file_path + "/ExtractedFeatures/" + _mainExpName + "/test/");
				std::filesystem::create_directories(_file_path + "/ExtractedFeatures/" + _mainExpName + "/train/");
				SavePairVector(_file_path + "/ExtractedFeatures/" + _mainExpName + "/train/" + _experiment.name() + ".json", output_train_set);
				SavePairVector(_file_path + "/ExtractedFeatures/" + _mainExpName + "/test/" + _experiment.name() + ".json", output_test_set);
			}
			if (_draw_features)
			{
				std::filesystem::create_directories(_file_path + "/ExtractedFeatures/" + _mainExpName + "/train_Features/" + output.name() + "/");
				std::filesystem::create_directories(_file_path + "/ExtractedFeatures/" + _mainExpName + "/test_Features/" + output.name() + "/");
				DrawSparseFeatures(_file_path + "/ExtractedFeatures/" + _mainExpName + "/train_Features/" + output.name() + "/" + _experiment.name() + "_", output_train_set);
				DrawSparseFeatures(_file_path + "/ExtractedFeatures/" + _mainExpName + "/test_Features/" + output.name() + "/" + _experiment.name() + "_", output_test_set);
			}
			if (_allow_residual_connections == true)
			{
				std::filesystem::create_directories(_file_path + "/ResInput/");
				SaveInputPairVector(_file_path + "/ResInput/" + _experiment.output_at(i).name() + "_train.json", _train_set);
				SaveInputPairVector(_file_path + "/ResInput/" + _experiment.output_at(i).name() + "_test.json", _test_set);
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