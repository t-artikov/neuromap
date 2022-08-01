#include <fstream>

#include <tiny-cuda-nn/config.h>

#include "Model.h"
#include "SoftmaxCrossEntropyLoss.h"

using json = nlohmann::json;
using precision_t = tcnn::network_precision_t;

namespace
{

template <class T>
std::vector<T> to_cpu(const T * gpu_data, const size_t size)
{
	std::vector<T> result(size);
	cudaMemcpy(result.data(), gpu_data, size * sizeof(T), cudaMemcpyDeviceToHost);
	return result;
}

std::shared_ptr<tcnn::Loss<precision_t>> create_loss(const json & opts)
{
	const auto type = opts.value("otype", std::string());
	if (type == "SoftmaxCrossEntropyLoss")
	{
		return std::make_shared<SoftmaxCrossEntropyLoss<precision_t>>();
	}

	return std::shared_ptr<tcnn::Loss<precision_t>>{tcnn::create_loss<precision_t>(opts)};
}

}

Model::Model(const std::string & config_path)
{
	std::ifstream config_file(config_path);
	json config = json::parse(config_file);

	json encoding_opts = config.value("encoding", json::object());
	json loss_opts = config.value("loss", json::object());
	json optimizer_opts = config.value("optimizer", json::object());
	json network_opts = config.value("network", json::object());
	hidden_layer_neuron_count_ = network_opts.value("n_neurons", 0);
	encoding_level_count_ = encoding_opts.value("n_levels", 0);

	std::shared_ptr<tcnn::Loss<precision_t>> loss{create_loss(loss_opts)};
	std::shared_ptr<tcnn::Optimizer<precision_t>> optimizer{tcnn::create_optimizer<precision_t>(optimizer_opts)};
	network_ = std::make_shared<tcnn::NetworkWithInputEncoding<precision_t>>(2, 3, encoding_opts, network_opts);
	encoding_ = &dynamic_cast<tcnn::GridEncoding<precision_t>&>(*network_->encoding());

	trainer_ = std::make_shared<tcnn::Trainer<float, precision_t, precision_t>>(network_, optimizer, loss);
}

unsigned int Model::level_count() const
{
	return encoding_level_count_;
}

float Model::training_step(
	cudaStream_t stream,
	const tcnn::GPUMatrix<float> & input,
	const tcnn::GPUMatrix<float> & target,
	bool calculate_loss)
{
	auto ctx = trainer_->training_step(stream, input, target);
	if (calculate_loss)
	{
		return trainer_->loss(stream, *ctx);
	}
	return 0.0f;
}

void Model::inference(cudaStream_t stream,
	const tcnn::GPUMatrix<float> & input,
	tcnn::GPUMatrix<float> & output)
{
	network_->inference(stream, input, output);
}

void Model::set_max_levels(float * levels)
{
	encoding_->set_max_level_gpu(levels);
}

void Model::set_max_level(float level)
{
	encoding_->set_max_level_gpu(nullptr);
	encoding_->set_max_level(level);
}

void Model::save()
{
	const auto encoding_params_count = encoding_->n_params();
	const auto network_params_count = trainer_->model()->n_params() - encoding_params_count;
	{
		std::ofstream f("network.data", std::ios::binary);
		const auto count = network_params_count - hidden_layer_neuron_count_ * 4;
		auto params = to_cpu(trainer_->params(), count);
		f.write(reinterpret_cast<const char*>(params.data()), count * sizeof(float));
	}
	{
		const auto json = trainer_->serialize();
		json::binary_t params = json["params_binary"];
		std::ofstream f("encoding.data", std::ios::binary);
		f.write(reinterpret_cast<const char*>(
			params.data()) + network_params_count * sizeof(precision_t),
			encoding_params_count * sizeof(precision_t));
	}
}
