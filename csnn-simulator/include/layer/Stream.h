#ifndef _STREAM_H
#define _STREAM_H

#include "Stdp.h"
#include "Layer.h"
#include "Stdp.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include "tool/Operations.h"
#include "plot/Threshold.h"
#include "plot/Evolution.h"
#include <thread>         // std::this_thread::sleep_for
#include <chrono> 

namespace layer
{
	/**
	 * @brief  This is only a filler layer to pass some dimentions to the SVM.
	 */
	class Stream : public Layer4D
	{

	public:
		/**
		* @brief Construct a new Convolution 3 D object
	 	* 
	 	*/
		Stream();

		Stream(size_t filter_width, size_t filter_height, size_t filter_depth, size_t filter_number = 6);

		Stream(const Stream &that) = delete;
		Stream &operator=(const Stream &that) = delete;

		virtual Shape compute_shape(const Shape &previous_shape);

		virtual size_t train_pass_number() const;
		virtual void process_train_sample(const std::string &label, Tensor<float> &sample, size_t current_pass, size_t current_index, size_t number);
		virtual void process_test_sample(const std::string &label, Tensor<float> &sample, size_t current_index, size_t number);

		virtual void train(const std::string &label, const std::vector<Spike> &input_spike, const Tensor<Time> &input_time, std::vector<Spike> &output_spike); //, size_t layer_index, size_t epoch_index );
		virtual void test(const std::string &label, const std::vector<Spike> &input_spike, const Tensor<Time> &input_time, std::vector<Spike> &output_spike);

		virtual Tensor<float> reconstruct(const Tensor<float> &t) const;

	
	};

} // namespace layer
#endif
