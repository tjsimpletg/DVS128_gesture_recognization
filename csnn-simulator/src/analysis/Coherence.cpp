#include "analysis/Coherence.h"
#include "Experiment.h"

using namespace analysis;

static RegisterClassParameter<Coherence, AnalysisFactory> _register("Coherence");

Coherence::Coherence() : NoPassAnalysis(_register)
{
}

void Coherence::resize(const Shape &)
{
}

void Coherence::process()
{
	experiment().log() << "===Coherence===" << std::endl;

	if (experiment().process_at(layer_index()).has_parameter("w") && experiment().process_at(layer_index()).is_type<Tensor<float>>("w"))
	{
		const Tensor<float> &w = experiment().process_at(layer_index()).parameter<Tensor<float>>("w").get();

		if (w.shape().number() == 4)
		{

			size_t width = w.shape().dim(0);
			size_t height = w.shape().dim(1);
			size_t depth = w.shape().dim(2);
			size_t n = w.shape().dim(3);

			std::vector<float> list;

			for (size_t i = 0; i < n; i++)
			{
				for (size_t j = i + 1; j < n; j++)
				{
					float value = 0;
					float ni = 0;
					float nj = 0;

					for (size_t x = 0; x < width; x++)
					{
						for (size_t y = 0; y < height; y++)
						{
							for (size_t z = 0; z < depth; z++)
							{
								value += w.at(x, y, z, i) * w.at(x, y, z, j);
								ni += w.at(x, y, z, i) * w.at(x, y, z, i);
								nj += w.at(x, y, z, j) * w.at(x, y, z, j);
							}
						}
					}

					list.push_back(value / (std::numeric_limits<float>::epsilon() + std::sqrt(ni) * std::sqrt(nj)));
				}
			}

			std::sort(std::begin(list), std::end(list));

			experiment().log() << "N: " << list.size() << std::endl;
			experiment().log() << "Min: " << list.front() << std::endl;
			experiment().log() << "Q1: " << list.at(std::min(list.size() - 1, (list.size() * 1) / 4)) << std::endl;
			experiment().log() << "Q2: " << list.at(std::min(list.size() - 1, (list.size() * 2) / 4)) << std::endl;
			experiment().log() << "Q3: " << list.at(std::min(list.size() - 1, (list.size() * 3) / 4)) << std::endl;
			experiment().log() << "Max: " << list.back() << std::endl;
		}
		else if (w.shape().number() == 5)
		{

			size_t width = w.shape().dim(0);
			size_t height = w.shape().dim(1);
			size_t depth = w.shape().dim(2);
			size_t n = w.shape().dim(3);
			size_t conv_depth = w.shape().dim(4);

			std::vector<float> list;

			for (size_t i = 0; i < n; i++)
			{
				for (size_t j = i + 1; j < n; j++)
				{
					float value = 0;
					float ni = 0;
					float nj = 0;

					for (size_t x = 0; x < width; x++)
					{
						for (size_t y = 0; y < height; y++)
						{
							for (size_t z = 0; z < depth; z++)
							{
								for (size_t k = 0; k < conv_depth; k++)
								{
									value += w.at(x, y, z, i, k) * w.at(x, y, z, j, k);
									ni += w.at(x, y, z, i, k) * w.at(x, y, z, i, k);
									nj += w.at(x, y, z, j, k) * w.at(x, y, z, j, k);
								}
							}
						}

						list.push_back(value / (std::numeric_limits<float>::epsilon() + std::sqrt(ni) * std::sqrt(nj)));
					}
				}

				std::sort(std::begin(list), std::end(list));

				experiment().log() << "N: " << list.size() << std::endl;
				experiment().log() << "Min: " << list.front() << std::endl;
				experiment().log() << "Q1: " << list.at(std::min(list.size() - 1, (list.size() * 1) / 4)) << std::endl;
				experiment().log() << "Q2: " << list.at(std::min(list.size() - 1, (list.size() * 2) / 4)) << std::endl;
				experiment().log() << "Q3: " << list.at(std::min(list.size() - 1, (list.size() * 3) / 4)) << std::endl;
				experiment().log() << "Max: " << list.back() << std::endl;
			}
		}
		else
		{
			experiment().log() << "Incompatible w shape." << std::endl;
		}
	}
	else
	{
		experiment().log() << "No w parameter in this layer." << std::endl;
	}

	experiment().log() << std::endl;
}
