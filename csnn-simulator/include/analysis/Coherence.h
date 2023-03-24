#ifndef _ANALYSIS_COHERENCE_H
#define _ANALYSIS_COHERENCE_H

#include "Analysis.h"

namespace analysis {

	/**
 	* @brief Coherence measures the ammount of redundant features. The less the coherence, the more features leared by the SNN.
	* The training should have  acertain incoherence with SNNs.
 	*/
	class Coherence : public NoPassAnalysis {

	public:
		Coherence();

		virtual void resize(const Shape& shape);

		virtual void process();
	};

}

#endif
