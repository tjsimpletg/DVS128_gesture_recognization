#include "SpikeConverter.h"

void SpikeConverter::to_spike(const Tensor<Time> &in, std::vector<Spike> &out)
{
	size_t width = in.shape().dim(0);
	size_t height = in.shape().dim(1);
	size_t depth = in.shape().dim(2);
	size_t conv_depth = in.shape().number() > 3 ? in.shape().dim(3) : 1;

	if (in.shape().number() == 3)
		for (size_t x = 0; x < width; x++)
		{
			for (size_t y = 0; y < height; y++)
			{
				for (size_t z = 0; z < depth; z++)
				{
					Time t = in.at(x, y, z);
					if (t != INFINITE_TIME)
					{
						out.emplace_back(t, x, y, z);
					}
				}
			}
		}
	else
		for (size_t x = 0; x < width; x++)
		{
			for (size_t y = 0; y < height; y++)
			{
				for (size_t z = 0; z < depth; z++)
				{
					for (size_t k = 0; k < conv_depth; k++)
					{
						Time t = in.at(x, y, z, k);
						if (t != INFINITE_TIME)
						{
							out.emplace_back(t, x, y, z, k);
						}
					}
				}
			}
		}

	std::sort(std::begin(out), std::end(out), TimeComparator());
}

void SpikeConverter::to_spike(const Tensor<Time> &in, std::vector<Spike> &out, size_t x_start, size_t y_start, size_t x_end, size_t y_end)
{
	size_t width = in.shape().dim(0);
	size_t height = in.shape().dim(1);
	size_t depth = in.shape().dim(2);
	size_t conv_depth = in.shape().number() > 3 ? in.shape().dim(3) : 1;
	if (in.shape().number() == 3)
		for (size_t x = x_start; x < std::min(width, x_end); x++)
			for (size_t y = y_start; y < std::min(height, y_end); y++)
				for (size_t z = 0; z < depth; z++)
				{
					Time t = in.at(x, y, z);
					if (t != INFINITE_TIME)
					{
						out.emplace_back(t, x - x_start, y - y_start, z);
					}
				}
	else
		for (size_t x = x_start; x < std::min(width, x_end); x++)
			for (size_t y = y_start; y < std::min(height, y_end); y++)
				for (size_t z = 0; z < depth; z++)
					for (size_t k = 0; k < conv_depth; k++)
					{
						Time t = in.at(x, y, z, k);
						if (t != INFINITE_TIME)
						{
							out.emplace_back(t, x - x_start, y - y_start, z, k);
						}
					}

	std::sort(std::begin(out), std::end(out), TimeComparator());
}

void SpikeConverter::from_spike(const std::vector<Spike> &in, Tensor<Time> &out)
{
	out.fill(INFINITE_TIME);
	if (out.shape().number() == 3)
		for (const Spike &spike : in)
		{
			out.at(spike.x, spike.y, spike.z) = spike.time;
		}
	else
		for (const Spike &spike : in)
		{
			out.at(spike.x, spike.y, spike.z, spike.k) = spike.time;
		}
}
