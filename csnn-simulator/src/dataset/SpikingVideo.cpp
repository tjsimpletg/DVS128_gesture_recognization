#include "dataset/SpikingVideo.h"
#include <iostream>
#include <string>
#include "tool/npy.hpp"

using namespace dataset;

SpikingVideo::SpikingVideo(const std::string& videos_npy_filename, const std::string& label_npy_filename)
        :_videos_npy_filename(videos_npy_filename),
         _label_npy_filename(label_npy_filename),
         _size(0), 
         _cursor(0),
         _label_cursor(0),
         _shape({FRAME_HEIGHT,FRAME_WIDTH,VIDEO_DEPTH,FRAME_NUMBER})
{
    LoadNpy(_videos_npy_filename,_data,_all_shape);
    LoadNpy(_label_npy_filename,_label,_all_shape);
}


bool SpikingVideo::has_next() const
{
	return _cursor < size();
}

std::pair<std::string, Tensor<InputType>> SpikingVideo::next()
{
    size_t _current_label = _label[_label_cursor];
    std::pair<std::string, Tensor<InputType>> out(std::to_string(static_cast<size_t>(_current_label)), _shape);
    for (size_t i = 0; i < FRAME_NUMBER; i++)
    {
        for (size_t j = 0; j < VIDEO_DEPTH; j++)
        { 
            for (size_t y = 0; y < FRAME_HEIGHT; y++)
	        {
		        for (size_t x = 0; x < FRAME_WIDTH; x++)
		        {
			        float pixel;
			
			        out.second.at(y,x,j,i) = static_cast<InputType>(pixel);
		        }
	        }
        }
    }

	// Tensor<float>::draw_Mnist_tensor("/home/melassal/Workspace/Draw/Mnist/Raw_" + std::to_string(label) + "_" + std::to_string(_cursor) + "_", out.second);

	_cursor+=FRAME_HEIGHT*FRAME_WIDTH*VIDEO_DEPTH*FRAME_NUMBER;
    _label_cursor++;
	return out;
}



size_t SpikingVideo::size() const
{
	return _all_shape[0]*_all_shape[1]*_all_shape[2]*_all_shape[3]*_all_shape[4];
}

void SpikingVideo::reset()
{
	_cursor = 0;
    _label_cursor=0;
}

const Shape &SpikingVideo::shape() const
{
	return _shape;
}





void SpikingVideo::LoadNpy(std::string &npy_file_path,std::vector<float> &_data, std::vector<unsigned long> &shape)
{
    bool is_fortran{false};
    npy::LoadArrayFromNumpy(npy_file_path, shape, is_fortran, _data);
}

std::string SpikingVideo::to_string() const
{
	return "SpikingVideo(" + std::to_string(_cursor) + ", " + std::to_string(_label_cursor) + ")";
}

void SpikingVideo::close()
{
    ///
}