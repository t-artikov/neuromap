#include <chrono>

#include <cxxopts.hpp>

#include "Accuracy.h"
#include "CombinedTilesSampler.h"
#include "Model.h"

void train(cudaStream_t stream, tcnn::default_rng_t & rng, Model & model, uint32_t steps, CombinedTilesSampler & tiles_sampler)
{
	const auto batch_size = tiles_sampler.batch_size();
	const auto n_input_dims = 2;
	const auto n_output_dims = 3;

	tcnn::GPUMatrix<float> coords(n_input_dims, batch_size);
	tcnn::GPUMatrix<float> colors(n_output_dims, batch_size);

	tcnn::GPUMemory<float> max_levels(batch_size);
	model.set_max_levels(max_levels.data());

	auto begin = std::chrono::steady_clock::now();

	float loss_sum = 0;
	uint32_t loss_counter = 0;

	std::cout << "Beginning training with " << steps << " steps." << std::endl;

	uint32_t interval = 1000;
	for (uint32_t i = 1; i <= steps; ++i)
	{
		bool print_loss = i % interval == 0;

		tiles_sampler.sample(stream, rng, coords.data(), colors.data());
		tcnn::generate_random_uniform<float>(stream, rng, batch_size, max_levels.data(), 3.0f / model.level_count(), 1.5f);

		bool get_loss = i % 100 == 0;
		const auto loss = model.training_step(stream, coords, colors, get_loss);
		if (get_loss)
		{
			loss_sum += loss;
			++loss_counter;
		}

		if (print_loss)
		{
			std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
			std::cout << "Step#" << i << ": " << "loss=" << loss_sum/(float)loss_counter << " time=" << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

			loss_sum = 0;
			loss_counter = 0;
			begin = std::chrono::steady_clock::now();
		}
	}
	model.set_max_levels(nullptr);
}

int main(int argc, char* argv[])
{
	cxxopts::Options options_parser("trainer");
	options_parser.add_options()
		("config", "", cxxopts::value<std::string>())
		("tiles", "", cxxopts::value<std::string>())
		("empty_tiles", "", cxxopts::value<std::string>())
		("steps", "", cxxopts::value<unsigned int>()->default_value("10000"));
	const auto opt = options_parser.parse(argc, argv);

	CombinedTilesSampler tiles_sampler(
		1 << 16,
		opt["empty_tiles"].as<std::string>(),
		opt["tiles"].as<std::string>()
	);

	Model model(opt["config"].as<std::string>());
	const auto n_training_steps = opt["steps"].as<uint32_t>();

	tcnn::default_rng_t rng(123);

	cudaStream_t stream;
	CUDA_CHECK_THROW(cudaStreamCreate(&stream));

	train(stream, rng, model, n_training_steps, tiles_sampler);
	model.save();
	print_accuracy(stream, rng, tiles_sampler, model);
}
