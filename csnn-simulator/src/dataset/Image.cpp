#include "dataset/Image.h"
#include <iostream>

using namespace dataset;

/**
 * @brief Get the image dataset from the input path,
 * We assign each folder name as a label to it's contained videos, (example: all the videos inside a folder names boxing will have a label indicating that they are boxing videos)
 *
 * @param video_folder_name gets the names of the sub-folders inside the main folder,
 * @param max_read
 */
Image::Image(const std::string &images_folder_path, const size_t &temporal_depth, const size_t &frame_size_width, const size_t &frame_size_height, const size_t &grey, size_t max_read) : _images_folder_path(images_folder_path), _temporal_depth(temporal_depth), _frame_size_width(frame_size_width), _frame_size_height(frame_size_height), _grey(grey), _threshould(2000), _size(0),
																																														  _frame_cursor(0), _folder_cursor(0), _shape({IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH, CONV_DEPTH}), _max_read(max_read)
{
	std::vector<cv::String> fnSize;
	std::string _file_path = "";
	for (const auto &file : std::filesystem::directory_iterator(_images_folder_path))
	{
		_file_path = file.path();
		_images_path_list.push_back(_file_path);
		_label_list.push_back(_file_path.substr(_file_path.find_last_of("/") + 1));
	}

	cv::glob(_images_path_list[0] + "/*.*", fnSize, false);
	// cv::glob(_images_path_list[0] + "/*.png", fnSize, false);

	// Get the correct frame shape.
	cv::Mat _size_frame;
	_size_frame = cv::imread(fnSize[0]);

	_temporal_depth = temporal_depth == 0 ? 1 : temporal_depth;

	if (_grey == 1)
		cv::cvtColor(_size_frame, _size_frame, cv::COLOR_BGR2GRAY);

	size_t _width = _frame_size_width == 0 ? _size_frame.cols : _frame_size_width;
	size_t _height = _frame_size_height == 0 ? _size_frame.rows : _frame_size_height;
	size_t _depth = _size_frame.channels();
	size_t _conv_depth = _temporal_depth;
	_shape = Shape(std::vector<size_t>({_height, _width, _depth, _conv_depth}));
}

bool Image::has_next() const
{
	return (_frame_cursor < size() || _folder_cursor < _images_path_list.size());
}

std::pair<std::string, Tensor<InputType>> Image::next()
{
	// Get folder path.
	_current_images_path = _images_path_list[_folder_cursor];
	// Get images in the folder.
	std::vector<cv::String> fn;
	// cv::glob(_current_images_path + "/*.png", fn, false);
	cv::glob(_current_images_path + "/*.*", fn, false);
	// Get total number of images.
	size_t _total_frames_in_folder = fn.size();

	cv::Mat frame;

	size_t _label = assign_label_to_sample(_current_images_path);
	std::pair<std::string, Tensor<InputType>> out(std::to_string(static_cast<size_t>(_label)), _shape);

	for (size_t _im = 0; _im < _temporal_depth; _im++)
	{
		frame = cv::imread(fn[_frame_cursor]);

		if (frame.empty())
		{
			break;
		}

		if (_frame_size_height != 0 || _frame_size_width != 0)
			cv::resize(frame, frame, cv::Size(_frame_size_width, _frame_size_height));

		if (_grey == 1)
			cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

		// _frame_number loops over the CONV_DEPTH by being incremented every frame.
		for (int i = 0; i < frame.rows; i++)
			for (int j = 0; j < frame.cols; j++)
				for (int k = 0; k < frame.channels(); k++)
				{
					if (frame.channels() > 1)
						out.second.at(i, j, k, _im) = (frame.at<cv::Vec3b>(i, j)[k]);
					else
						out.second.at(i, j, k, _im) = (frame.at<unsigned char>(i, j));
				}
		_frame_cursor++;
	}

	// Tensor<float>::draw_colored_tensor("/home/melassal/Workspace/CSNN _2./csnn-simulator-build/test/frame" + out.first + "_" + std::to_string(_frame_cursor) + "_", out.second);
	// Tensor<float>::draw_nonscaled_tensor("/home/melassal/Workspace/CSNN _2./csnn-simulator-build/test/frame" + out.first + "_" + std::to_string(_frame_cursor) + "_", out.second);

	if (_frame_cursor >= _total_frames_in_folder)
	{
		_folder_cursor++;
		if (_folder_cursor < _images_path_list.size())
			_frame_cursor = 0;
	}

	return out;
}

void Image::reset()
{
	_frame_cursor = 0;
}

/**
 * @brief This function is used to assign number labels to videos, if the video name changes,
 * thus there is a new action, thus we need a new label.
 *
 * @param _current_images_path the video being saved.
 * @return uint32_t returnd the assigned label to the video.
 */
uint32_t Image::assign_label_to_sample(std::string _current_images_path)
{
	std::string _label = _current_images_path.substr(_current_images_path.find_last_of("/") + 1);
	size_t _label_index = std::distance(_label_list.begin(), std::find(_label_list.begin(), _label_list.end(), _label));
	return _label_index;
}

// /**
//  * @brief This function checs if the frame contains enough movement to be taken into consideration.
//  *
//  * @param _current_images_path the video being saved.
//  * @return uint32_t returnd the assigned label to the video.
//  */
// uint32_t Image::check_movement_threshould(cv::Mat frame)
// {
// 		cv::Scalar sum = cv::sum(frame);

// 		if (sum(0) > _threshould)
// 		{
// 			return ;
// 		}
// 	}
// }

void Image::close()
{
	// _label_file.close();
	// _image_file.close();
}

size_t Image::size() const
{
	return std::min(_size, _max_read);
}

std::string Image::to_string() const
{
	return "Image(" + _images_folder_path + ")";
}

const Shape &Image::shape() const
{
	return _shape;
}

uint32_t Image::swap(uint32_t v)
{
	// return ((v & 0xFF) << 24) | ((v & 0xFF00) << 8) | ((v & 0xFF0000) >> 8) | ((v & 0xFF000000) >> 24);
}