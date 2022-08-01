#pragma once
#include "EmptyTilesSampler.h"
#include "DetailedTilesSampler.h"

class CombinedTilesSampler
{
public:
	CombinedTilesSampler(uint32_t batch_size, const std::string & empty_tiles_path, const std::string & tiles_path);
	void sample(cudaStream_t stream, tcnn::default_rng_t & rng, float * coords, float * colors);

	uint32_t batch_size() const;

private:
	const uint32_t batch_size_;
	EmptyTilesSampler empty_tiles_sampler_;
	DetailedTilesSampler detailed_tiles_sampler_;
};
