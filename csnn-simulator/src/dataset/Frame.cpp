#include "dataset/Frame.h"
#include <iostream>

using namespace dataset;

/**
 * @brief Construct a new Frame:: Video object takes the input path,
 * We assign each folder name as a label to it's contained videos, (example: all the videos inside a folder names boxing will have a label indicating that they are boxing videos)
 *
 * @param video_folder_name gets the names of the sub-folders inside the main folder,
 * @param max_read
 * @param frame_size_width
 * @param frame_size_height
 */
Frame::Frame(const std::string &video_folder_path, const size_t &frame_per_video, const size_t &frame_gap, const size_t &threshold,
			 const size_t &frame_size_width, const size_t &frame_size_height, const size_t &sample_per_video, std::string exp_name, const size_t &draw, size_t max_read) : _video_folder_path(video_folder_path), _frame_per_video(frame_per_video), _frame_gap(frame_gap), _frame_gap_counter(0),
																																										   _frame_size_width(frame_size_width), _frame_size_height(frame_size_height), _sample_per_video(sample_per_video), _draw(draw), _exp_name(exp_name), _frame_preprocess(0), _frame_number(0), _threshold(threshold),
																																										   _cursor(0), _cursor_count(0), _label_count(0), _shape({FRAME_WIDTH, FRAME_HEIGHT, FRAME_DEPTH, CONV_DEPTH}), _max_read(max_read)
{
	for (const auto &file : std::filesystem::directory_iterator(_video_folder_path))
	{
		std::string _file_path = file.path();
		_action_list.push_back(_file_path.substr(_video_folder_path.length() + 1));

		for (const auto &_file : std::filesystem::directory_iterator(_file_path))
			_video_list.push_back(_file.path());
	}

	std::sort(_video_list.begin(), _video_list.end());
	std::sort(_action_list.begin(), _action_list.end());

	_size = _video_list.size();

	// Get the coorect frame shape.
	cv::VideoCapture capture(_video_list[0]);

	cv::Mat _size_frame;
	capture >> _size_frame;

	cv::cvtColor(_size_frame, _size_frame, cv::COLOR_BGR2GRAY);

	size_t _width = _frame_size_width == 0 ? _size_frame.cols : _frame_size_width;
	size_t _height = _frame_size_height == 0 ? _size_frame.rows : _frame_size_height;
	size_t _depth = _size_frame.channels();
	size_t _conv_depth = 1; // _frame_per_video;
	_shape = Shape(std::vector<size_t>({_height, _width, _depth, _conv_depth}));
}

bool Frame::has_next() const
{
	return _cursor < size();
}

std::pair<std::string, Tensor<InputType>> Frame::next()
{
	_current_video_name = _video_list[_cursor];
	cv::VideoCapture capture(_current_video_name);

	if (!capture.isOpened())
		std::cout << "Unable to open file!" << std::endl;

	_frame_number = 0;
	int size[3] = {_shape.dim(0), _shape.dim(1), _shape.dim(2)};

	// frame dimentions
	cv::Mat frame(_shape.dim(2), size, CV_32F, cv::Scalar(0)), skipFrame(_shape.dim(2), size, CV_32F, cv::Scalar(0)), spframe(_shape.dim(2), size, CV_32F, cv::Scalar(0));
	// start video from a different point by adding a shift
	set_frame_gap(_cursor_count * 3, capture, skipFrame);

	size_t _label = assign_label_to_sample(_current_video_name);
	// make shape dims of frame.
	std::pair<std::string, Tensor<InputType>> out(std::to_string(static_cast<size_t>(_label)), _shape);
	// extract a frame
	capture >> frame;

	cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

	if (_frame_size_height != 0 || _frame_size_width != 0)
		cv::resize(frame, frame, cv::Size(_frame_size_width, _frame_size_height));

	int fluct = 0;

	// loop.
	while (true)
	{
		// speed up the action.
		cv::Mat next_frame(_shape.dim(2), size, CV_32F, cv::Scalar(0));

		capture >> next_frame;

		// count number of frames
		if (frame.empty() || next_frame.empty() || _cursor_count == _frame_per_video)
		{
			_frame_number = 0;
			// _cursor_count = 0;
			break;
		}

		cv::cvtColor(next_frame, next_frame, cv::COLOR_BGR2GRAY);

		if (_frame_size_height != 0 || _frame_size_width != 0)
			cv::resize(next_frame, next_frame, cv::Size(_frame_size_width, _frame_size_height));

		// if (_current_video_name.find("running") != std::string::npos || _current_video_name.find("jogging") != std::string::npos || _current_video_name.find("walking") != std::string::npos)
		if (_threshold > 0)
			if (movement_threshold(frame, next_frame))
			{
				frame = next_frame;
				fluct = 0;
				continue;
			}

		if (_frame_gap > 0)
			set_frame_gap(_frame_gap, capture, skipFrame);

		// frame = frame_preprocess(1, frame, next_frame);

		if (!frame.empty())
		{
			// _frame_number loops over the CONV_DEPTH by being incremented every frame.
			for (int i = 0; i < frame.rows; i++)
				for (int j = 0; j < frame.cols; j++)
					for (int k = 0; k < frame.channels(); k++)
					{
						if (frame.channels() > 1)
							out.second.at(i, j, k, 0) = (frame.at<cv::Vec3b>(i, j)[k]) / static_cast<InputType>(std::numeric_limits<uint8_t>::max());
						else
							out.second.at(i, j, k, 0) = (frame.at<unsigned char>(i, j)) / static_cast<InputType>(std::numeric_limits<uint8_t>::max());
					}
			_frame_number++;
		}
		fluct = 1;
		frame = next_frame;
	}
	if (_frame_per_video > 0)
	{
		if (_cursor_count == _frame_per_video)
		{
			_cursor++;
			_cursor_count = 0;
		}
		_cursor_count++;
	}
	else
		_cursor++;

	if (fluct == 1 && _draw == 1)
		save_as_images(out);

	_frame_gap_counter = 0;
	return out;
}

void Frame::save_as_images(std::pair<std::string, Tensor<InputType>> out)
{
	_file_path = std::filesystem::current_path();
	if (_current_video_name.find("train") != std::string::npos)
	{
		_current_video_name = _current_video_name.substr(_video_folder_path.length() + 1);
		std::string _action = _current_video_name.substr(0, _current_video_name.find("/"));
		std::filesystem::create_directories("Input_frames/" + _exp_name + "/train/" + _action + "/");
		Tensor<float>::draw_tensor(_file_path + "/Input_frames/" + _exp_name + "/train/" + _action + "/" + _action + "_" + std::to_string(_cursor) + "_" + std::to_string(_frame_number), out.second);
	}
	if (_current_video_name.find("test") != std::string::npos)
	{
		_current_video_name = _current_video_name.substr(_video_folder_path.length() + 1);
		std::string _action = _current_video_name.substr(0, _current_video_name.find("/"));
		std::filesystem::create_directories("Input_frames/" + _exp_name + "/test/" + _action + "/");
		Tensor<float>::draw_tensor(_file_path + "/Input_frames/" + _exp_name + "/test/" + _action + "/" + _action + "_" + std::to_string(_cursor) + "_" + std::to_string(_frame_number), out.second);
	}
}

/**
 * @brief This function is used to assign number labels to videos, if the video name changes,
 * thus there is a new action, thus we need a new label.
 *
 * @param _current_video_name the video being saved.
 * @return uint32_t returnd the assigned label to the video.
 */
uint32_t Frame::assign_label_to_sample(std::string _current_video_name)
{
	_current_video_name = _current_video_name.substr(_video_folder_path.length() + 1);
	std::string _action = _current_video_name.substr(0, _current_video_name.find("/"));
	_label_count = std::distance(_action_list.begin(), find(_action_list.begin(), _action_list.end(), _action));

	return _label_count;
}

/**
 * @brief Skipping frames from the video to speed up the action.
 *
 * @param _frame_gap The number of skipped frames
 * @param capture the capture object of opencv that inputs the video
 * @param skipFrame the skipped frame.
 */
void Frame::set_frame_gap(int _frame_gap, cv::VideoCapture capture, cv::Mat skipFrame)
{
	int frame_Number = capture.get(cv::CAP_PROP_FRAME_COUNT);
	int frame_sets = (frame_Number / 2);

	// _frame_gap_counter++;
	// // skip to the second half of the video
	// if (_frame_gap_counter == (_frame_per_video / 2))
	// 	for (int i = 0; i < frame_sets; i++)
	// 		capture >> skipFrame;

	for (int i = 0; i < _frame_gap; i++)
		capture >> skipFrame;
}

/**
 * @brief Checking the movement threshold.
 *
 * @param frame the first frame
 * @param next_frame the next frame
 */
bool Frame::movement_threshold(cv::Mat frame, cv::Mat next_frame)
{
	cv::Mat difference;
	difference = frame - next_frame; // diference of second and third frame
	cv::Scalar sum = cv::sum(difference);
	int _sum = sum(0);
	if (sum(0) < _threshold)
	{
		return true;
	}
	return false;
}

cv::Mat Frame::frame_preprocess(int _frame_preprocess, cv::Mat frame, cv::Mat next_frame)
{
	// covert frame into greyscale.
	// cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
	if (_frame_preprocess == 0)
		return frame;
	else if (_frame_preprocess == 1)
	{
		cv::Mat difference;

		// cv::cvtColor(next_frame, next_frame, cv::COLOR_BGR2GRAY);
		difference = frame - next_frame; // diference of second and third frame

		cv::Scalar sum = cv::sum(difference);

		if (sum(0) > _threshold)
		{
			return difference;
		}
	}
}

void Frame::reset()
{
	_cursor = 0;
	_label_count = 0;
}

void Frame::close()
{
	// _label_file.close();
	// _image_file.close();
}

size_t Frame::size() const
{
	return std::min(_size, _max_read);
}

std::string Frame::to_string() const
{
	return "Video(" + _video_folder_path + ")[" + std::to_string(size()) + "]";
}

const Shape &Frame::shape() const
{
	return _shape;
}

uint32_t Frame::swap(uint32_t v)
{
	// return ((v & 0xFF) << 24) | ((v & 0xFF00) << 8) | ((v & 0xFF0000) >> 8) | ((v & 0xFF000000) >> 24);
}