#ifndef _ANALYSIS_ACTIVITY_H
#define _ANALYSIS_ACTIVITY_H

#include "Analysis.h"

namespace analysis {
	/**
 	* @brief Activity means the neuronal activity. In the world of SNNs, 
	* the neuron that fires tends to silence it's peers, so in order to keep a state of homeostasis, 
	* analysing the activity of teh neurons is important.
 	*/
	class Activity : public UniquePassAnalysis {

	public:
		Activity();

		void resize(const Shape&);

		void before_train();
		void process_train(const std::string& label, const Tensor<float>& sample);
		void after_train();

		void before_test();
		void process_test(const std::string& label, const Tensor<float>& sample);
		void after_test();

	private:
		void _reset();
		void _process(const Tensor<float>& sample);
		void _print();

		double _sparsity;
		double _activity;
		size_t _quiet;
		size_t _count;
		size_t _size;
	};
}
#endif

