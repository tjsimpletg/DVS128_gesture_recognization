#ifndef _PROCESS_FLATTEN_H
#define _PROCESS_FLATTEN_H

#include <filesystem>
#include "Process.h"

namespace process
{
	/**
	 * @brief This post-processing flattens by changing the dimentions of the tensor into product x 1 x 1
	 */
	class Flatten : public UniquePassProcess
	{

	public:
		Flatten();

		virtual Shape compute_shape(const Shape &shape);
		virtual void process_train(const std::string &label, Tensor<float> &sample);
		virtual void process_test(const std::string &label, Tensor<float> &sample);

	private:
		void _process(const std::string &label, Tensor<float> &t) const;

		size_t _product;
	};

}

#endif
