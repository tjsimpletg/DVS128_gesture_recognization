#ifndef _STDP_MULTIPLICATIVE_H
#define _STDP_MULTIPLICATIVE_H

#include "Stdp.h"

namespace stdp {

	class Multiplicative : public STDP {

	public:
		Multiplicative();
		Multiplicative(float alpha, float beta);

		virtual float process(float w, const Time pre, Time post);
		virtual void adapt_parameters(float factor);
	private:
		float _alpha;
		float _beta;
	};

}
#endif
