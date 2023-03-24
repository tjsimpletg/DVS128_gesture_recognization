#ifndef _DATASET_FRAME_H
#define _DATASET_FRAME_H

#include <filesystem>
#include <string>
#include <cassert>
#include <fstream>
#include <limits>
#include <tuple>
#include <math.h>

#include "Tensor.h"
#include "Input.h"

/**
 * @brief An image sample in FRAME is 28x28 and with a depth of 1 correspoiding to Greyscale.
 * 
 * @param FRAME_WIDTH is the width of the frame. 
 * @param FRAME_HEIGHT is the height of the frame. 
 * @param FRAME_DEPTH is the number of channels. 
 * @param CONV_DEPTH is the depth of the convolution in case of 3D convolution. 
 */
#define FRAME_WIDTH 1
#define FRAME_HEIGHT 1
#define FRAME_DEPTH 1
#define CONV_DEPTH 1

namespace dataset
{

	/**
	 * @brief This class fills the videos into tensors
	 * @param video_folder_name video location.
	 * @param frame_per_video video sequence number, is set to zero, the whole video is taken (after some size calculation).
	 * @param frame_gap video frames that are skipped to speed up the action.
	 * @param frame_size_width video frame width size that is set to zero takes the default size.
	 * @param frame_size_height video frame height size that is set to zero takes the default size.
	 */
	class Frame : public Input
	{

	public:
		Frame(const std::string &video_folder_name, const size_t &frame_per_video, const size_t &frame_gap = 0, const size_t &threshold = 0,
			  const size_t &frame_size_width = 0, const size_t &frame_size_height = 0,const size_t &sample_per_video = 0,
			  std::string _exp_name = "", const size_t &draw = 0, size_t max_read = std::numeric_limits<size_t>::max());
		/**
		 * @brief A function that is called to fetch the next sample if the size is not reached.
		 * This function can be seen in the OptimizedLayerByLayer class in the load function.
		 */
		virtual bool has_next() const;

		/**
		 * @brief loads a video sample using openCV.
		 * @return std::pair<std::string, Tensor<InputType>> 
		 */
		virtual std::pair<std::string, Tensor<InputType>> next();
		/**
		 * @brief Gets the video label from the video location (if the video is in a folder names boxing then it's a boxing video).
		 * @param _current_video_name 
		 */
		virtual uint32_t assign_label_to_sample(std::string _current_video_name);
		/**
		 * @brief Set the frame gap object decides how many frames to skip between each two consiquetive frames.
		 * @param frame_gap 
		 */
		// virtual void set_frame_gap(cv::FrameCapture capture, cv::Mat skipFrame);
		virtual void set_frame_gap(int _frame_gap, cv::VideoCapture capture, cv::Mat skipFrame);

		virtual bool movement_threshold(cv::Mat frame, cv::Mat next_frame);
		/**
		 * @brief This function is responsible for pre-processing the vodei frames. 
		 * @param _frame_preprocess if process is 0 then no pre-processing, if process is one then background subtraction.
		 * process is an int not a boolean in case we add more pre-processing possibilities in the future.
		 */
		virtual cv::Mat frame_preprocess(int _frame_preprocess, cv::Mat frame, cv::Mat next_frame);
		/**
		 * @brief This function allows saving the video frames used in the expirement as an image dataset.
		 * The location of this dataset is in the build folder.
		 */
		virtual void save_as_images(std::pair<std::string, Tensor<InputType>> out);

		virtual void reset();
		virtual void close();

		size_t size() const;
		virtual std::string to_string() const;

		virtual const Shape &shape() const;

	private:
		uint32_t swap(uint32_t v);
		// The path of the folder that contains the videos.
		std::string _video_folder_path;
		uint32_t _frame_size_width;
		uint32_t _frame_size_height;

		// A list that contains all the video names.
		std::vector<std::string> _video_list;
		// A list that contains all the action names.
		std::vector<std::string> _action_list;

		uint32_t _cursor;
		uint32_t _cursor_count;
		uint32_t _sample_per_video;

		uint32_t _size;
		// counter for the number of frames that form a video sample
		uint32_t _frame_number;
		// each video has a specific label.
		uint32_t _label_count;

		Shape _shape;

		std::string _current_video_name;
		// checks if the video name has changed to assign a new label.
		std::string _video_name_buffer;
		// The total number of frames that represent one video.
		int _frame_per_video;
		// The number of skipped frames between each two selected frames.
		int _frame_gap;
		// A counter used in the frage-gap function to indicate when half the frames are taken from the begining of the video.
		int _frame_gap_counter;
		int _frame_preprocess;
		std::string _file_path;

		//The threshold that records the video activity. (if less than this threshold, the activity is not significant and the frame is not recorded.)
		int _draw;
		std::string _exp_name;
		int _threshold;

		uint32_t _max_read;
	};

}

#endif
