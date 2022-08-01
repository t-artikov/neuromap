#pragma once

#include <memory>
#include <utility>
#include <vector>

#include <tiny-cuda-nn/gpu_matrix.h>
#include <tiny-cuda-nn/random.h>

#include "HostMemory.h"
#include "ThreadSafeQueue.h"

namespace DetailedTilesSamplerImpl
{

struct Tile
{
	static constexpr uint32_t resolution = 32;
	static constexpr float inv_resolution = 1.f / resolution;
	static constexpr uint32_t pixel_count = resolution * resolution;
	static constexpr uint32_t data_size = pixel_count / 4;

	uint32_t x;
	uint32_t y;
	std::array<uint8_t, data_size> data;
};

struct Coord
{
	float x;
	float y;
};
struct Color
{
	float r;
	float g;
	float b;
};

struct Batch
{
	HostMemory<Coord> coords;
	HostMemory<Color> colors;
};

class Generator;
using GeneratorPtr = std::shared_ptr<Generator>;

} // namespace DetailedTilesSamplerImpl

class DetailedTilesSampler
{
public:
	DetailedTilesSampler(uint32_t batch_size, uint32_t resolution, const std::string & path);
	void sample(float * coords, float * colors);
	~DetailedTilesSampler();

private:
	using Tile = DetailedTilesSamplerImpl::Tile;
	using Coord = DetailedTilesSamplerImpl::Coord;
	using Color = DetailedTilesSamplerImpl::Color;
	using Batch = DetailedTilesSamplerImpl::Batch;
	using GeneratorPtr = DetailedTilesSamplerImpl::GeneratorPtr;
	std::pair<Coord, Color> get_sample(tcnn::default_rng_t & rng) const;

	const uint32_t batch_size_;
	std::vector<Tile> tiles_;
	ThreadSafeQueue<Batch> free_batches_;
	ThreadSafeQueue<Batch> batches_;
	std::vector<GeneratorPtr> generators_;
};
