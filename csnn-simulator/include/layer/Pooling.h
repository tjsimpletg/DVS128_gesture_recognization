#ifndef _LAYER_POOLING_H
#define _LAYER_POOLING_H

#include "Layer.h"

namespace layer
{

	class Pooling : public Layer3D
	{

	public:
		Pooling();
		Pooling(size_t filter_width, size_t filter_height, size_t stride_x = 2, size_t stride_y = 2, size_t padding_x = 0, size_t padding_y = 0);

		virtual Shape compute_shape(const Shape &previous_shape);

		virtual size_t train_pass_number() const;
		virtual void process_train_sample(const std::string &label, Tensor<float> &sample, size_t current_pass, size_t current_index, size_t number);
		virtual void process_test_sample(const std::string &label, Tensor<float> &sample, size_t current_index, size_t number);

		virtual void train(const std::string &label, const std::vector<Spike> &input_spike, const Tensor<Time> &input_time, std::vector<Spike> &output_spike);
		virtual void test(const std::string &label, const std::vector<Spike> &input_spike, const Tensor<Time> &input_time, std::vector<Spike> &output_spike);
		virtual Tensor<float> reconstruct(const Tensor<float> &t) const;

	private:
		void _exec(const std::vector<Spike> &input_spike, std::vector<Spike> &output_spike);

		Tensor<bool> _inh;
	};

	/**
	 * @brief  Pooling has many benifits, one is to reduce the dimensionality of the data across the layers (When the data is smaller, it's faster to process)
	 * Pooling also maximizes the edges and global features the SNN is interested in.
	 * This pooling layer can be used for 3D data like videos, -This si in progress- in order to perform temporal pooling.
	 * 
	 * Example: a 2x2 filter with a stride of 2, this reduces our input by a factor of 4. 
	 * So, for each 4 pixels, 4 neurons are used, and we choose only one, which has the maximum value in case of max-pooling or the average in case of 
	 * sum-pooling. So we end up with less pixels and need less neurons.
	 * 
	 * @param filter_width the width of the pooling filter.
	 * @param filter_hight the height of the pooling filter.
	 * @param filter_conv_depth the temporal depth of the pooling filter.
	 * @param stride_x the filters step to move in x direction.
	 * @param stride_y the filters step to move in y direction.
	 * @param stride_k the filters step to move in time dim.
	 */
	class Pooling3D : public Layer4D
	{

	public:
		Pooling3D();
		Pooling3D(size_t filter_width, size_t filter_height, size_t filter_conv_depth = 1, size_t stride_x = 1, size_t stride_y = 1, size_t stride_k = 1, size_t padding_x = 0, size_t padding_y = 0, size_t padding_k = 0);

		virtual Shape compute_shape(const Shape &previous_shape);

		virtual size_t train_pass_number() const;
		virtual void process_train_sample(const std::string &label, Tensor<float> &sample, size_t current_pass, size_t current_index, size_t number);
		virtual void process_test_sample(const std::string &label, Tensor<float> &sample, size_t current_index, size_t number);

		virtual void train(const std::string &label, const std::vector<Spike> &input_spike, const Tensor<Time> &input_time, std::vector<Spike> &output_spike);
		virtual void test(const std::string &label, const std::vector<Spike> &input_spike, const Tensor<Time> &input_time, std::vector<Spike> &output_spike);
		virtual Tensor<float> reconstruct(const Tensor<float> &t) const;

	private:
		void _exec(const std::vector<Spike> &input_spike, std::vector<Spike> &output_spike);

		Tensor<bool> _inh;
	};

}

#endif
