#ifndef _INPUT_TOOL_H
#define _INPUT_TOOL_H

#include <string>
#include "Tensor.h"

class InputTool
{

public:
	InputTool()
	{
	}

	virtual ~InputTool()
	{
	}

	virtual std::string to_string() const = 0;
};

#endif
