#include "tool/VideoFrameSelector.h"

using namespace tool;

VideoFrameSelector::VideoFrameSelector(const std::string &video_folder_path, size_t &frame_per_video) : _video_folder_path(video_folder_path)
{

	// The used didn't decide the number of frames so we get the min.
	if (frame_per_video != 0)
	{
		std::vector<std::string> _video_list;
		for (const auto &file : std::filesystem::directory_iterator(_video_folder_path))
		{
			std::string _file_path = file.path();

			for (const auto &_action_file : std::filesystem::directory_iterator(_file_path))
				for (const auto &_file : std::filesystem::directory_iterator(_action_file))
					_video_list.push_back(_file.path());
		}

		std::sort(_video_list.begin(), _video_list.end());
		size_t _size = _video_list.size();

		int _minimum_buffer = 0;
		for (size_t _vid_i = 0; _vid_i < _size; _vid_i++)
		{
			int counter = 0;
			cv::Mat frame;
			cv::VideoCapture cap(_video_list[_vid_i]);
			int frame_Number = cap.get(cv::CAP_PROP_FRAME_COUNT);

			if (_minimum_buffer == 0)
				_minimum_buffer = frame_Number;
			else if (frame_Number < _minimum_buffer)
				_minimum_buffer = frame_Number;
		}
		frame_per_video = _minimum_buffer;
	}
}

std::string VideoFrameSelector::to_string() const
{
	return "Video(" + _video_folder_path + ")";
}