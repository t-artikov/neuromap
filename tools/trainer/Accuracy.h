#pragma once
#include <tiny-cuda-nn/random.h>

class CombinedTilesSampler;
class Model;

void print_accuracy(
	cudaStream_t stream,
	tcnn::default_rng_t & rng,
	CombinedTilesSampler & tiles_sampler,
	Model & model);
