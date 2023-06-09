#ifndef _DATASET_SPIKINGVIDEO_H
#define _DATASET_SPIKINGVIDEO_H

#include <istream>
#include <iostream>
#include <string>
#include <cassert>
#include <fstream>
#include <limits>
#include <tuple>

#include "Tensor.h"
#include "Input.h"


#define FRAME_WIDTH 128
#define FRAME_HEIGHT 128
#define FRAME_NUMBER 20
#define VIDEO_DEPTH 2
#define CONV_DEPTH 1

namespace dataset {

	class SpikingVideo : public Input {

	public:
		SpikingVideo(const std::string& videos_npy_filename, const std::string& label_npy_filename);

		virtual bool has_next() const;
		virtual std::pair<std::string, Tensor<InputType>> next();
		virtual void reset();
		virtual void close();

		size_t size() const;
		virtual std::string to_string() const;

		virtual const Shape& shape() const;

	private:

		std::string _videos_npy_filename;
		std::string _label_npy_filename;

		uint32_t _size;
		uint32_t _cursor;
		uint32_t _label_cursor;
		Shape _shape;
		std::vector<float> _data;
		std::vector<float> _label;
		std::vector<unsigned long> _data_shape;
		std::vector<unsigned long> _label_shape;
		bool _is_fortran{false};
	};

}

#endif