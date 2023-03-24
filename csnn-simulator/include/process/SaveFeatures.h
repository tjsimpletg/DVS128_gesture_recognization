#ifndef _SAVE_FEATURES_H
#define _SAVE_FEATURES_H

#include "Process.h"
#include "tool/Operations.h"

namespace process
{

	/**
	 * @brief This function allows saving the features in their full size.
	 * This function is a post-processing, so it can only be used after output conversion (the process)
	 * 
	 * @param exp_name the name of the experiment
	 * @param layer_name the name of the layer
	 */
	// counters for the progress bar
	static int _train_save_sample_count = 0;
	static int _test_save_sample_count = 0;
	static int _general_shape = 0;
	class SaveFeatures : public UniquePassProcess
	{

	public:
		SaveFeatures();
		SaveFeatures(std::string exp_name, std::string layer_name = "Default");

		virtual Shape compute_shape(const Shape &shape);
		virtual void process_train(const std::string &label, Tensor<float> &sample);
		virtual void process_test(const std::string &label, Tensor<float> &sample);

	private:
		void _process(Tensor<float> &sample) const;
		
		std::string _file_path;
		std::string _exp_name;
		std::string _layer_name;

		size_t _width;
		size_t _height;
		size_t _depth;
		size_t _conv_depth;
	};
}

#endif
