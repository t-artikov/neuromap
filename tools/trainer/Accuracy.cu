#include <iomanip>

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/gpu_matrix.h>

#include "Accuracy.h"
#include "CombinedTilesSampler.h"
#include "Model.h"

namespace
{

__device__ inline int max_index(float v0, float v1, float v2)
{
	if (v0 > v1)
	{
		return v0 > v2 ? 0 : 2;
	}
	else
	{
		return v1 > v2 ? 1 : 2;
	}
}

__global__ void calculate_difference(uint32_t n_elements, float* __restrict__ a, float* __restrict__ b, float* __restrict__ result)
{
	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n_elements) return;

	const uint32_t j = i * 3;
	const auto a_max = max_index(a[j], a[j + 1], a[j + 2]);
	const auto b_max = max_index(b[j], b[j + 1], b[j + 2]);
	result[i] = a_max == b_max ? 1.0f : 0.0f;
}

double calculate_accuracy(cudaStream_t stream, tcnn::default_rng_t & rng, CombinedTilesSampler & tiles_sampler, Model & model, float level)
{
	constexpr auto n_input_dims = 2;
	constexpr auto n_output_dims = 3;
	constexpr auto n_iter = 1000;
	const auto batch_size = tiles_sampler.batch_size();

	tcnn::GPUMatrix<float> coords(n_input_dims, batch_size);
	tcnn::GPUMatrix<float> target(n_output_dims, batch_size);
	tcnn::GPUMatrix<float> prediction(n_output_dims, batch_size);
	tcnn::GPUMemory<float> difference(batch_size);

	model.set_max_level(level);

	double total_sum = 0.0;
	for (int i = 0; i < n_iter; ++i)
	{
		tiles_sampler.sample(stream, rng, coords.data(), target.data());
		model.inference(stream, coords, prediction);
		tcnn::linear_kernel(calculate_difference, 0, stream, batch_size, target.data(), prediction.data(), difference.data());
		const double sum = tcnn::reduce_sum(difference.data(), batch_size, stream);
		total_sum += sum / batch_size;
	}
	return total_sum / n_iter;
}

}

void print_accuracy(cudaStream_t stream, tcnn::default_rng_t & rng, CombinedTilesSampler & tiles_sampler, Model & model)
{
	std::cout << "calculating accuracy..." << std::endl;
	for (auto level = 2; level <= model.level_count(); ++level)
	{
		const auto accuracy = calculate_accuracy(stream, rng, tiles_sampler, model, level / static_cast<float>(model.level_count()));
		std::cout << "accuracy_" << level << "=" << std::setprecision(8) << accuracy * 100.0 << std::endl;
	}
}
