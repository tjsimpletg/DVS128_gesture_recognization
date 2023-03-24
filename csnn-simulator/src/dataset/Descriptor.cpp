#include "dataset/Descriptor.h"
using namespace dataset;

Descriptor::Descriptor(const std::string & descriptor_folder_path, size_t max_read) : _descriptor_folder_path(descriptor_folder_path), 
	_size(0), _cursor(0), _shape({DESCRIPTOR_WIDTH, DESCRIPTOR_HEIGHT, DESCRIPTOR_DEPTH, CONV_DEPTH}), _max_read(max_read)
{
	LoadPairVector(descriptor_folder_path, _descriptors);

	size_t _width = _descriptors[0].second.shape().dim(0);
	size_t _height = _descriptors[0].second.shape().dim(1);
	size_t _depth = _descriptors[0].second.shape().dim(2);
	size_t _conv_depth = _descriptors[0].second.shape().dim(3);
	_shape = Shape(std::vector<size_t>({_width, _height, _depth, _conv_depth}));
	_size = _descriptors.size();
}

bool Descriptor::has_next() const
{
	return _cursor < size();
}

std::pair<std::string, Tensor<InputType>> Descriptor::next()
{

	_current_descriptor = _descriptors[_cursor];
	std::pair<std::string, Tensor<InputType>> out(_current_descriptor.first, _shape);
	out.second = _current_descriptor.second;

	_cursor++;

	return out;
}

void Descriptor::reset()
{
	_cursor = 0;
}

void Descriptor::close()
{

}

size_t Descriptor::size() const
{
	return std::min(_size, _max_read);
}

std::string Descriptor::to_string() const
{
	return "Descriptor(" + _descriptor_folder_path + ")[" + std::to_string(size()) + "]";
}

const Shape &Descriptor::shape() const
{
	return _shape;
}


uint32_t Descriptor::swap(uint32_t v)
{
	return ((v & 0xFF) << 24) | ((v & 0xFF00) << 8) | ((v & 0xFF0000) >> 8) | ((v & 0xFF000000) >> 24);
}
