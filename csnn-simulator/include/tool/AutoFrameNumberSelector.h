#ifndef _TOOL_AUTO_FRAME_NUMBER_SELECTOR_H
#define _TOOL_AUTO_FRAME_NUMBER_SELECTOR_H

#include <filesystem>
#include <fstream>
#include "Tensor.h"
#include "InputTool.h"

namespace tool
{
    /**
     * @brief This tool is used to get the minimum number of frames found in both the test & train video datasets.
     * This is needed because all the input should be of same dimention, thus all the videos should have the same length.
     * 
     * @param video_folder_path The path of the train and test datasets.
     * @param frame_per_video The maximum number of frames per video, if 0 takes the whole video (all the frames).
     */
    class AutoFrameNumberSelector : public InputTool
    {

    public:
        AutoFrameNumberSelector();
        AutoFrameNumberSelector(const std::string & video_folder_path,  size_t & frame_per_video);

		virtual std::string to_string() const;

    private:
        std::string _video_folder_path;
    };
}

#endif