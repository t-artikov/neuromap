#include "CombinedTilesSampler.h"

CombinedTilesSampler::CombinedTilesSampler(const uint32_t batch_size, const std::string & empty_tiles_path, const std::string & tiles_path)
	: batch_size_(batch_size)
	, empty_tiles_sampler_(empty_tiles_path)
	, detailed_tiles_sampler_(batch_size_ / 2, empty_tiles_sampler_.resolution(), tiles_path)
{
}

uint32_t CombinedTilesSampler::batch_size() const
{
	return batch_size_;
}

void CombinedTilesSampler::sample(
	cudaStream_t stream,
	tcnn::default_rng_t & rng,
	float * coords,
	float * colors)
{
	const auto empty_count = batch_size_ / 2;
	empty_tiles_sampler_.sample(stream, rng, empty_count, coords, colors);
	detailed_tiles_sampler_.sample(coords + empty_count * 2, colors + empty_count * 3);
}
