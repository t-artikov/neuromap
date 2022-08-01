#pragma once
#include <tiny-cuda-nn/encodings/grid.h>
#include <tiny-cuda-nn/network_with_input_encoding.h>
#include <tiny-cuda-nn/trainer.h>

class Model
{
public:
	Model(const std::string & config_path);
	float training_step(
		cudaStream_t stream,
		const tcnn::GPUMatrix<float> & input,
		const tcnn::GPUMatrix<float> & target,
		bool calculate_loss);
	void inference(cudaStream_t stream, const tcnn::GPUMatrix<float> & input, tcnn::GPUMatrix<float> & output);
	unsigned int level_count() const;
	void set_max_levels(float * levels);
	void set_max_level(float level);
	void save();

private:
	using precision_t = tcnn::network_precision_t;
	std::shared_ptr<tcnn::NetworkWithInputEncoding<precision_t>> network_;
	std::shared_ptr<tcnn::Trainer<float, precision_t, precision_t>> trainer_;
	tcnn::GridEncoding<precision_t> * encoding_ = nullptr;
	unsigned int hidden_layer_neuron_count_ = 0;
	unsigned int encoding_level_count_ = 0;
};
