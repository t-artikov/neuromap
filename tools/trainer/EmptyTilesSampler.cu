#include <fstream>

#include "Common.h"
#include "EmptyTilesSampler.h"

namespace
{

__global__ void get_sample(
	uint32_t n_elements,
	uint32_t resolution,
	tcnn::default_rng_t rng,
	const uint8_t * __restrict__ data,
	float * __restrict__ coords,
	float * __restrict__ colors)
{
	const auto i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

	const auto coord_idx = i * 2;
	const auto color_idx = i * 3;

	rng.advance(i * 100);

	float x;
	float y;
	uint32_t c;

	while (true)
	{
		x = rng.next_float();
		y = rng.next_float();
		const auto ix = static_cast<uint32_t>(x * resolution);
		const auto iy = static_cast<uint32_t>(y * resolution);
		const auto subindex = ix % 4;
		c = (data[resolution / 4 * iy + ix / 4] >> (subindex * 2)) & 3;
		if (c != 0) break;
	}

	coords[coord_idx] = x;
	coords[coord_idx + 1] = y;

	if (c == 1)
	{
		colors[color_idx] = 1.0f;
		colors[color_idx + 1] = 0.0f;
		colors[color_idx + 2] = 0.0f;
	}
	else if (c == 2)
	{
		colors[color_idx] = 0.0f;
		colors[color_idx + 1] = 1.0f;
		colors[color_idx + 2] = 0.0f;
	}
	else
	{
		colors[color_idx] = 0.0f;
		colors[color_idx + 1] = 0.0f;
		colors[color_idx + 2] = 1.0f;
	}
}

}

EmptyTilesSampler::EmptyTilesSampler(const std::string & path)
	: data_(to_gpu(read_file_content<uint8_t>(path)))
	, resolution_(static_cast<uint32_t>(std::sqrt(data_.size() * 4)))
{}

void EmptyTilesSampler::sample(
		cudaStream_t stream,
		tcnn::default_rng_t & rng,
		uint32_t count,
		float * coords,
		float * colors) const
{
	tcnn::linear_kernel(get_sample, 0, stream, count, resolution_, rng, data_.data(), coords, colors);
	rng.advance(count * 100);
}

uint32_t EmptyTilesSampler::resolution() const
{
	return resolution_;
}
