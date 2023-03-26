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
    npy::LoadArrayFromNumpy(_videos_npy_filename, _data_shape, _is_fortran, _data);
    npy::LoadArrayFromNumpy(_label_npy_filename,_label_shape,_is_fortran,_label);
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
                    pixel = _data[i*FRAME_WIDTH*VIDEO_DEPTH*FRAME_NUMBER*+j*FRAME_WIDTH*VIDEO_DEPTH+y*FRAME_WIDTH+x];
			        out.second.at(y,x,j,i) = static_cast<InputType>(pixel);
		        }
	        }
        }
    }

	//Tensor<float>::draw_nonscaled_tensor("/home/zhang/S2/RP/csnn-simulator-build/test/Raw_" + std::to_string(_current_label) + "_" + std::to_string(_cursor/(FRAME_HEIGHT*FRAME_WIDTH*VIDEO_DEPTH*FRAME_NUMBER)) + "_", out.second);

	_cursor+=FRAME_HEIGHT*FRAME_WIDTH*VIDEO_DEPTH*FRAME_NUMBER;
    _label_cursor++;
	return out;
}



size_t SpikingVideo::size() const
{
	return _data_shape[0]*FRAME_HEIGHT*FRAME_WIDTH*VIDEO_DEPTH*FRAME_NUMBER;
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



std::string SpikingVideo::to_string() const
{
	return "SpikingVideo(" + std::to_string(_cursor) + ", " + std::to_string(_label_cursor) + ")";
}

void SpikingVideo::close()
{
    ///
}