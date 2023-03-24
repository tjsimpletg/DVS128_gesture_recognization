#ifndef _RESIDUAL_CONNECTION_H
#define _RESIDUAL_CONNECTION_H

#include "Process.h"
#include "tool/Operations.h"

namespace process
{
	// counters for the progress bar & res connection sample fusion
	static int _train_res_sample_count = 0;
	static int _test_res_sample_count = 0;

	/**
	 * @brief
	 * @param exp_name The name of the expirement
	 */
	class ResidualConnection : public UniquePassProcess
	{

	public:
		ResidualConnection();
		ResidualConnection(std::string exp_name, std::string layer_name);

		virtual Shape compute_shape(const Shape &shape);
		virtual void process_train(const std::string &label, Tensor<float> &sample);
		virtual void process_test(const std::string &label, Tensor<float> &sample);

	private:
		void _process(Tensor<float> &sample, Tensor<float> &original_sample) const;

		std::string _file_path;
		std::string _layer_name;

		std::vector<std::string> _data_list;

		std::vector<std::pair<std::string, Tensor<float>>> _train_set;
		std::vector<std::pair<std::string, Tensor<float>>> _test_set;

		size_t _width;
		size_t _height;
		size_t _depth;
		size_t _conv_depth;
	};

}

#endif
