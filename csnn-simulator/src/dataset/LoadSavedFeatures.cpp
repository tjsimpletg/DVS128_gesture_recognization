#include "dataset/LoadSavedFeatures.h"
using namespace dataset;

LoadSavedFeatures::LoadSavedFeatures(const std::string &folder_path, size_t max_read) : _folder_path(folder_path),
                                                                                        _size(0), _cursor(0), _shape({1, 1, 1, 1}), _max_read(max_read)
{
    // Get the saved locations of the featues
    for (const auto &file : std::filesystem::directory_iterator(_folder_path))
    {
        std::string _file_path = file.path();
        _data_list.push_back(_file_path);
    }

    // load the saved spatial features.
    LoadPairVector(_data_list[0], _features);

    _shape = _features[0].second.shape();
    _size = _features.size();
}

bool LoadSavedFeatures::has_next() const
{
    return _cursor < size();
}

std::pair<std::string, Tensor<InputType>> LoadSavedFeatures::next()
{
    std::string _label = _features[_cursor].first;

    std::pair<std::string, Tensor<InputType>> out(_label, _shape);

    out.second = _features[_cursor].second;

    _cursor++;

    return out;
}

void LoadSavedFeatures::reset()
{
    _cursor = 0;
}

void LoadSavedFeatures::close()
{
}

size_t LoadSavedFeatures::size() const
{
    return std::min(_size, _max_read);
}

std::string LoadSavedFeatures::to_string() const
{
    return "LoadSavedFeatures(" + _folder_path + ")[" + std::to_string(size()) + "]";
}

const Shape &LoadSavedFeatures::shape() const
{
    return _shape;
}

uint32_t LoadSavedFeatures::swap(uint32_t v)
{
    return ((v & 0xFF) << 24) | ((v & 0xFF00) << 8) | ((v & 0xFF0000) >> 8) | ((v & 0xFF000000) >> 24);
}
