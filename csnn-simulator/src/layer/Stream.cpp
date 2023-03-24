#include "layer/Stream.h"
#include "Experiment.h"
#include <execution>
#include <mutex>

using namespace layer;

static RegisterClassParameter<Stream, LayerFactory> _register("Stream");
/**
 * Stream is only a filler layer
 */
Stream::Stream() : Layer4D(_register)
{
}

Stream::Stream(size_t filter_width, size_t filter_height, size_t filter_depth, size_t filter_number) : Layer4D(_register, filter_width, filter_height, filter_depth, filter_number, 1, 1, 1, 0, 0, 0)
{
}

/**
 * @brief The convolutional layer is initialised with random weights. The w tensor takes the same dimentions as the filerts.
 *
 * @param previous_shape
 * @param random_generator
 */

Shape Stream::compute_shape(const Shape &previous_shape)
{
	Layer4D::compute_shape(previous_shape);

	return Shape({_width, _height, _depth, _conv_depth});
}

size_t Stream::train_pass_number() const
{
	return 0;
}

void Stream::process_train_sample(const std::string &label, Tensor<float> &sample, size_t current_pass, size_t current_index, size_t number)
{
}

void Stream::process_test_sample(const std::string &label, Tensor<float> &sample, size_t current_index, size_t number)
{
}

void Stream::train(const std::string &label, const std::vector<Spike> &input_spike, const Tensor<Time> &input_time, std::vector<Spike> &output_spike)
{
}

void Stream::test(const std::string &, const std::vector<Spike> &input_spike, const Tensor<Time> &input_time, std::vector<Spike> &output_spike)
{
}

// This function is not extended because it's only for drawing.
Tensor<float> Stream::reconstruct(const Tensor<float> &t) const
{
}