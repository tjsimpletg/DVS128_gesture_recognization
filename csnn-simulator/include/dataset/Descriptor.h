#ifndef _DATASET_DESCRIPTOR_H
#define _DATASET_DESCRIPTOR_H

#include <cassert>
#include <limits>
#include <tuple>

#include "Input.h"
#include "tool/Operations.h"

/**
 * @brief An image sample in DESCRIPTOR is 28x28 and with a depth of 1 correspoiding to Greyscale.
 * 
 * @param DESCRIPTOR_WIDTH is the width of the frame. 
 * @param DESCRIPTOR_HEIGHT is the height of the frame. 
 * @param DESCRIPTOR_DEPTH is the number of channels. 
 * @param CONV_DEPTH is the depth of the convolution in case of 3D convolution. 
 */
#define DESCRIPTOR_WIDTH 1
#define DESCRIPTOR_HEIGHT 1
#define DESCRIPTOR_DEPTH 1
#define CONV_DEPTH 1

namespace dataset
{

	/**
	 * @brief This class monitors introducing the dataset into the program, 
	 * It is responsible for counting the number of samples. This is the first step, even before transforming the data into spikes.
	 */
	class Descriptor : public Input
	{

	public:
		Descriptor(const std::string &descriptor_folder_path, size_t max_read = std::numeric_limits<size_t>::max());

		virtual bool has_next() const;
		virtual std::pair<std::string, Tensor<InputType>> next();
		virtual void reset();
		virtual void close();

		size_t size() const;
		virtual std::string to_string() const;

		virtual const Shape &shape() const;

	private:
		uint32_t swap(uint32_t v);

		std::string _descriptor_folder_path;

		std::vector<std::pair<std::string, Tensor<float>>> _descriptors;

		std::pair<std::string, Tensor<float>> _current_descriptor;
		uint32_t _size;
		uint32_t _cursor;

		Shape _shape;

		uint32_t _max_read;
	};

}

#endif
