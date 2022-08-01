#include <thread>

#include "Common.h"
#include "DetailedTilesSampler.h"

using namespace DetailedTilesSamplerImpl;

namespace
{

inline Color to_color(uint8_t index)
{
	switch (index)
	{
		case 1: return Color{1.f, 0.f, 0.f};
		case 2: return Color{0.f, 1.f, 0.f};
		case 3: return Color{0.f, 0.f, 1.f};
	}
	return {};
}

inline Color get_color(const Tile & tile, uint32_t x, uint32_t y)
{
	const auto subindex = x % 4;
	const auto c = (tile.data[tile.resolution / 4 * (tile.resolution - 1 - y) + x / 4] >> (subindex * 2)) & 3;
	return to_color(c);
}

}

namespace DetailedTilesSamplerImpl
{

class Generator
{
public:
	Generator(uint32_t resolution, const std::vector<Tile> & tiles, ThreadSafeQueue<Batch> & free_batches, ThreadSafeQueue<Batch> & batches)
		: tiles_(tiles)
		, tile_count_(tiles.size())
		, inv_resolution_(1.f / resolution)
		, free_batches_(free_batches)
		, batches_(batches)
		, thread_(&Generator::run, this)
		, rng_(rand())
	{
	}

	std::pair<Coord, Color> get_sample()
	{
		const auto i = rng_.next_uint(tile_count_);
		const auto & tile = tiles_[i];
		const auto pixel_x = rng_.next_uint(tile.resolution);
		const auto pixel_y = rng_.next_uint(tile.resolution);
		const auto x = (static_cast<float>(tile.x) + (pixel_x + 0.5f) * tile.inv_resolution) * inv_resolution_;
		const auto y = (static_cast<float>(tile.y) + (pixel_y + 0.5f) * tile.inv_resolution) * inv_resolution_;
		const auto color = get_color(tile, pixel_x, pixel_y);
		return {Coord{x, y}, color};
	}

	~Generator()
	{
		thread_.join();
	}

private:
	void run()
	{
		while (true)
		{
			auto batch = free_batches_.pop();
			auto coords = batch.coords.data();
			auto colors = batch.colors.data();
			if (batch.coords.empty())
			{
				return;
			}
			for (size_t i = 0, n = batch.coords.size(); i < n; ++i)
			{
				const auto sample = get_sample();
				coords[i] = sample.first;
				colors[i] = sample.second;
			}
			batches_.push(std::move(batch));
		}
	}

	std::thread thread_;
	const std::vector<Tile> & tiles_;
	const uint32_t tile_count_;
	const float inv_resolution_;
	ThreadSafeQueue<Batch> & free_batches_;
	ThreadSafeQueue<Batch> & batches_;
	tcnn::default_rng_t rng_;
};

}

DetailedTilesSampler::DetailedTilesSampler(uint32_t batch_size, uint32_t resolution, const std::string & path)
	: batch_size_(batch_size)
	, tiles_(read_file_content<Tile>(path))
{
	constexpr auto generator_count = 8;
	constexpr auto sample_queue_size = generator_count * 2;
	for (auto i = 0; i < sample_queue_size; ++i)
	{
		free_batches_.push(Batch{HostMemory<Coord>(batch_size), HostMemory<Color>(batch_size)});
	}
	generators_.reserve(generator_count);
	for (auto i = 0; i < generator_count; ++i)
	{
		generators_.push_back(std::make_shared<Generator>(resolution, tiles_, free_batches_, batches_));
	}
}

void DetailedTilesSampler::sample(float * coords, float * colors)
{
	auto batch = batches_.pop();
	CUDA_CHECK_THROW(cudaMemcpy(coords, batch.coords.data(), batch.coords.size() * sizeof(Coord), cudaMemcpyHostToDevice));
	CUDA_CHECK_THROW(cudaMemcpy(colors, batch.colors.data(), batch.colors.size() * sizeof(Color), cudaMemcpyHostToDevice));
	free_batches_.push(std::move(batch));
}

DetailedTilesSampler::~DetailedTilesSampler()
{
	for (size_t i = 0; i < generators_.size(); ++i)
	{
		free_batches_.push({});
	}
}
