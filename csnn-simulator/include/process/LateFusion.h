#ifndef _PROCESS_LATE_FUSION_H
#define _PROCESS_LATE_FUSION_H

#include <filesystem>
#include "Process.h"

namespace process
{
	/**
	 * @brief Late fusion is a type of feature fusion that is used to conserve the temporal component of a sequence of frames.
	 * For this function to be useful, the join_frames function in the execution policy should be greater than or equal to 2. 
	 * 
	 * @param expName The name of the expirement, this name will log the drawn frames.
	 * @param draw A flag that draws the pre-processed information in the build file.
	 * @param fused_frames_number Number of frames that are fused together to create one sample.
	 */
	class LateFusion : public UniquePassProcess
	{

	public:
		LateFusion();
		LateFusion(std::string expName, size_t draw = 0, size_t _fused_frames_number = 1);

		virtual Shape compute_shape(const Shape &shape);
		virtual void process_train(const std::string &label, Tensor<float> &sample);
		virtual void process_test(const std::string &label, Tensor<float> &sample);

	private:
		void _process(const std::string &label, Tensor<float> &t) const;

		size_t _draw;
		// To save the drawings
		std::string _file_path;
		std::string _expName;
		size_t _fused_frames_number;

		size_t _width;
		size_t _height;
		size_t _depth;
		size_t _conv_depth;
	};

}

#endif
