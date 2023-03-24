#include "stdp/Multiplicative.h"

using namespace stdp;

static RegisterClassParameter<Multiplicative, STDPFactory> _register("Multiplicative");

Multiplicative::Multiplicative() : STDP(_register), _alpha(0), _beta(0) {
    add_parameter("alpha", _alpha);
    add_parameter("beta", _beta);
}


Multiplicative::Multiplicative(float alpha, float beta) : Multiplicative() {
    parameter<float>("alpha").set(alpha);
    parameter<float>("beta").set(beta);
}

float Multiplicative::process(float w, const Time pre, Time post) {
	float v = pre <= post ? w+_alpha*std::exp(-_beta*w) :  w-_alpha*std::exp(_beta*(w-1.0f));
	return std::max<float>(0, std::min<float>(1, v));
}

void Multiplicative::adapt_parameters(float factor) {
	_alpha *= factor;
}
