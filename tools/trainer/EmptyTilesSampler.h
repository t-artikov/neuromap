#include <string>
#include <cstdint>

#include <tiny-cuda-nn/gpu_memory.h>
#include <tiny-cuda-nn/random.h>

class EmptyTilesSampler
{
public:
	EmptyTilesSampler(const std::string & path);
	void sample(cudaStream_t stream, tcnn::default_rng_t & rng, uint32_t count, float * coords, float * colors) const;

	uint32_t resolution() const;

private:
	tcnn::GPUMemory<uint8_t> data_;
	const uint32_t resolution_;
};
