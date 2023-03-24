#include "stdp/Biological.h"

using namespace stdp;

static RegisterClassParameter<Biological, STDPFactory> _register("Biological");

Biological::Biological() : STDP(_register), _alpha(0), _tau(0) {
	add_parameter("alpha", _alpha);
	add_parameter("tau", _tau);
}


Biological::Biological(float alpha, float tau) : Biological() {
	parameter<float>("alpha").set(alpha);
	parameter<float>("tau").set(tau);
}

float Biological::process(float w, const Time pre, Time post) {
	float v = pre <= post ? w+_alpha*std::exp(-(post-pre)/_tau) :  w-_alpha*std::exp(-(pre-post)/_tau);
	return std::max<float>(0, std::min<float>(1, v));
}

void Biological::adapt_parameters(float factor) {
	_alpha *= factor;
}
