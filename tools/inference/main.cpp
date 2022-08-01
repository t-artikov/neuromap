#include <algorithm>
#include <cassert>
#include <iostream>
#include <fstream>
#include <vector>

#include <cxxopts.hpp>
#include <json/json.hpp>
#include <stbi/stb_image_write.h>

#include "hf.h"

using json = nlohmann::json;

std::vector<float> load_floats(const std::string & filename)
{
	std::ifstream f(filename, std::ios_base::binary);
	f.seekg(0, std::ios::end);
	const size_t size = f.tellg();
	f.seekg(0, std::ios::beg);

	std::vector<float> result(size / sizeof(float));
	f.read(reinterpret_cast<char*>(result.data()), size);
	return result;
}

std::vector<float> load_hp_floats(const std::string & filename)
{
	std::ifstream f(filename, std::ios_base::binary);
	f.seekg(0, std::ios::end);
	const size_t size = f.tellg();
	f.seekg(0, std::ios::beg);

	std::vector<uint16_t> params(size / sizeof(uint16_t));
	f.read(reinterpret_cast<char*>(params.data()), size);

	std::vector<float> result(params.size());
	std::transform(params.begin(), params.end(), result.begin(), hf2float);
	return result;
}

struct Config
{
	struct Encoding
	{
		unsigned int n_levels;
		unsigned int n_features_per_level;
		unsigned int log2_hashmap_size;
		unsigned int base_resolution;
	} encoding;

	struct Network
	{
		unsigned int n_neurons;
		unsigned int n_hidden_layers;
		std::string activation;
		std::string output_activation;
	} network;
};

Config load_config(const std::string & filename)
{
	std::ifstream f(filename);
	const auto json = json::parse(f);
	json::object_t encoding = json["encoding"];
	json::object_t network = json["network"];
	return Config {
		.encoding = {
			.n_levels = encoding["n_levels"],
			.n_features_per_level = encoding["n_features_per_level"],
			.log2_hashmap_size = encoding["log2_hashmap_size"],
			.base_resolution = encoding["base_resolution"]
		},
		.network = {
			.n_neurons = network["n_neurons"],
			.n_hidden_layers = network["n_hidden_layers"],
			.activation = network["activation"],
			.output_activation = network["output_activation"]
		}
	};
}

struct Point
{
	float x;
	float y;
};

struct Rect
{
	float left;
	float top;
	float right;
	float bottom;

	float width() const
	{
		return right - left;
	}

	float height() const
	{
		return bottom - top;
	}
};

struct Color
{
	float r;
	float g;
	float b;
};

struct Image
{
	unsigned int width;
	unsigned int height;
	std::vector<Color> pixels;
};

Color quantize(const Color & c)
{
	constexpr Color r = {1.0f, 0.0f, 0.0f};
	constexpr Color g = {0.0f, 1.0f, 0.0f};
	constexpr Color b = {0.0f, 0.0f, 1.0f};
	if (c.r > c.g)
	{
		return c.r > c.b ? r : b;
	}
	else
	{
		return c.g > c.b ? g : b;
	}
}

float dot_product(const float * a, const float * b, const size_t n)
{
	float result = 0.f;
	for (size_t i = 0; i < n; ++i)
	{
		result += a[i] * b[i];
	}
	return result;
}

void linear_transform(const std::vector<float> & input, const float * weights, std::vector<float> & output)
{
	const auto m = input.size();
	for (size_t i = 0, n = output.size(); i < n; ++i)
	{
		output[i] = dot_product(input.data(), weights + i * m, m);
	}
}

void relu(std::vector<float> & values)
{
	for (auto & v : values)
	{
		v = std::max(0.f, v);
	}
}

void sigmoid(std::vector<float> & values)
{
	for (auto & v : values)
	{
		v = 1.0f / (1.0f + exp(-v));
	}
}

void sine(std::vector<float> & values)
{
	for (auto & v : values)
	{
		v = std::sin(v);
	}
}

void tanh(std::vector<float> & values)
{
	for (auto & v : values)
	{
		v = std::tanh(v);
	}
}

void none(std::vector<float> &)
{
}

using Activation = void (*)(std::vector<float> &);

Activation get_activation(const std::string & name)
{
	if (name == "ReLU")
	{
		return relu;
	}
	else if (name == "Sigmoid")
	{
		return sigmoid;
	}
	else if (name == "Sine")
	{
		return sine;
	}
	else if (name == "Tanh")
	{
		return tanh;
	}
	else if (name == "None")
	{
		return none;
	}

	assert(false);
	return nullptr;
}

template <typename T>
T div_round_up(T val, T divisor)
{
	return (val + divisor - 1) / divisor;
}

template <typename T>
T next_multiple(T val, T divisor)
{
	return div_round_up(val, divisor) * divisor;
}

class Encoding
{
public:
	class Layer
	{
	public:
		Layer(unsigned int resolution, unsigned int feature_count, bool use_hash, std::vector<float> params)
			: resolution_(resolution)
			, feature_count_(feature_count)
			, params_(std::move(params))
			, output_(feature_count, 0.f)
			, use_hash_(use_hash)
		{
		}

		const std::vector<float> & apply(const Point & p) const
		{
			const float x = p.x * (resolution_ - 1);
			const float y = p.y * (resolution_ - 1);
			const unsigned int ix = static_cast<unsigned int>(x);
			const unsigned int iy = static_cast<unsigned int>(y);
			const float kx = x - ix;
			const float ky = y - iy;
			for (unsigned int i = 0; i < feature_count_; ++i)
			{
				const float v00 = param(ix, iy, i);
				const float v10 = param(ix + 1, iy, i);
				const float v01 = param(ix, iy + 1, i);
				const float v11 = param(ix + 1, iy + 1, i);
				const float v_0 = v00 + (v10 - v00) * kx;
				const float v_1 = v01 + (v11 - v01) * kx;
				output_[i] = v_0 + (v_1 - v_0) * ky;
			}
			return output_;
		}

		uint32_t hash(uint32_t x, uint32_t y) const
		{
			if (use_hash_)
			{
				return (y * 2654435761) ^ x;
			}
			else
			{
				return y * resolution_ + x;
			}
		}

		float param(unsigned int x, unsigned int y, unsigned int i) const
		{
			const size_t index = (hash(x, y) * feature_count_ + i);
			return params_[index % params_.size()];
		}

	private:
		const unsigned int resolution_;
		const unsigned int feature_count_;
		const std::vector<float> params_;
		const bool use_hash_;
		mutable std::vector<float> output_;
	};

	Encoding(const Config::Encoding & config, const std::vector<float> & params)
		: feature_count_(config.n_features_per_level)
		,  output_(config.n_levels * config.n_features_per_level, 1.f)
	{
		size_t params_offset = 0;
		auto resolution = config.base_resolution;
		const auto hash_size = size_t(1) << config.log2_hashmap_size;
		layers_.reserve(config.n_levels);
		for (unsigned int i = 0; i < config.n_levels; ++i)
		{
			auto use_hash = false;
			auto params_count = static_cast<size_t>(next_multiple(resolution * resolution, 8u));
			if (params_count > hash_size)
			{
				use_hash = true;
				params_count = hash_size;
			}
			params_count *= config.n_features_per_level;

			assert(params.size() >= params_offset + params_count);
			std::vector<float> layer_params(params.begin() + params_offset, params.begin() + params_offset + params_count);
			layers_.push_back(Layer(resolution, config.n_features_per_level, use_hash, std::move(layer_params)));
			params_offset += params_count;
			resolution = (resolution - 1) * 2 + 1;
		}
	}

	unsigned int output_size() const
	{
		return output_.size();
	}

	const std::vector<float> & apply(const Point & p, float level) const
	{
		unsigned int offset = 0;
		unsigned int index = 0;
		for (const auto & layer : layers_)
		{
			const auto mask = level - index + 1.f;
			if (mask <= 0.f)
			{
				std::fill(output_.begin() + offset, output_.begin() + offset + feature_count_, 0.f);
			}
			else
			{
				const auto & output = layer.apply(p);
				std::copy(output.begin(), output.end(), output_.begin() + offset);
				if (mask < 1.f)
				{
					for (auto it = output_.begin() + offset; it != output_.begin() + offset + feature_count_; ++it)
					{
						*it *= mask;
					}
				}
			}
			offset += feature_count_;
			index++;
		}
		return output_;
	}

private:
	const unsigned int feature_count_;
	std::vector<Layer> layers_;
	mutable std::vector<float> output_;
};

class Network
{
public:
	Network(const Config::Network & config, std::vector<float> weights, unsigned int input_size)
		: activation_(get_activation(config.activation))
		, output_activation_(get_activation(config.output_activation))
		, weights_(std::move(weights))
	{
		values_.reserve(config.n_hidden_layers + 2);
		values_.emplace_back(input_size, 0.f);
		for (unsigned int i = 0; i < config.n_hidden_layers; i++)
		{
			values_.emplace_back(config.n_neurons, 0.f);
		}
		values_.emplace_back(4, 0.f);

		size_t expected_weight_count = 0;
		for (size_t i = 0; i < values_.size() - 1; i++)
		{
			expected_weight_count += values_[i].size() * values_[i + 1].size();
		}
		assert(expected_weight_count == weights_.size());
	}

	size_t weight_count() const
	{
		return weights_.size();
	}

	Color apply(const std::vector<float> & input) const
	{
		assert(input.size() == values_.front().size());
		std::copy(input.begin(), input.end(), values_.front().begin());

		size_t weights_offset = 0;
		for (size_t i = 0; i < values_.size() - 1; i++)
		{
			const auto & input = values_[i];
			auto & output = values_[i + 1];
			linear_transform(input, weights_.data() + weights_offset, output);
			if (i != values_.size() - 2)
			{
				activation_(output);
			}
			else
			{
				output_activation_(output);
			}
			weights_offset += input.size() * output.size();
		}

		const auto & output = values_.back();
		return Color{output[0], output[1], output[2]};
	}

private:
	const Activation activation_;
	const Activation output_activation_;
	std::vector<float> weights_;
	mutable std::vector<std::vector<float>> values_;
};

void save_image(const Image & image, const std::string & filename)
{
	const auto to_byte = [](const float v) {
		return static_cast<u_int8_t>(std::max(0.f, std::min(v, 1.0f)) * 255.f + 0.5f);
	};
	std::vector<uint8_t> pixels;
	pixels.reserve(image.pixels.size() * 3);
	for (const auto & color : image.pixels)
	{
		pixels.push_back(to_byte(color.r));
		pixels.push_back(to_byte(color.g));
		pixels.push_back(to_byte(color.b));
	}
	stbi_write_png(filename.c_str(), image.width, image.height, 3, pixels.data(), image.width * 3);
}

Image generate_image(const Encoding & encoding, const Network & network, const Rect & rect, unsigned int resolution, float level)
{
	const auto width = rect.width() >= rect.height() ? resolution : static_cast<unsigned int>(resolution * rect.width() / rect.height());
	const auto height = rect.height() >= rect.width() ? resolution : static_cast<unsigned int>(resolution * rect.height() / rect.width());
	Image image {width, height, std::vector<Color>(width * height)};
	for (unsigned int iy = 0; iy < height; ++iy)
	{
		const auto y = (iy + 0.5f) / height * rect.height() + rect.top;
		for (unsigned int ix = 0; ix < width; ++ix)
		{
			const auto x = (ix + 0.5f) / width * rect.width() + rect.left;
			const auto & encoded = encoding.apply({x, y}, level);
			const auto color = quantize(network.apply(encoded));
			image.pixels[iy * width + ix] = color;
		}
	}
	return image;
}

Rect get_rect(const cxxopts::ParseResult & opt)
{
	const auto rect = Rect {
		opt["left"].as<float>(),
		opt["top"].as<float>(),
		opt["right"].as<float>(),
		opt["bottom"].as<float>()
	};
	if (rect.left > rect.right || rect.top > rect.bottom)
	{
		throw std::runtime_error("Invalid rect");
	}
	return rect;
}

int main(int argc, const char ** argv)
{
	cxxopts::Options options_parser("inference");
	options_parser.add_options()
		("config", "", cxxopts::value<std::string>())
		("encoding", "", cxxopts::value<std::string>()->default_value("encoding.data"))
		("network", "", cxxopts::value<std::string>()->default_value("network.data"))
		("left", "", cxxopts::value<float>()->default_value("0.0"))
		("top", "", cxxopts::value<float>()->default_value("0.0"))
		("right", "", cxxopts::value<float>()->default_value("1.0"))
		("bottom", "", cxxopts::value<float>()->default_value("1.0"))
		("resolution", "", cxxopts::value<unsigned int>()->default_value("2000"))
		("level", "", cxxopts::value<float>()->default_value("1000.0"))
		("output", "", cxxopts::value<std::string>()->default_value("inference.png"));
	const auto opt = options_parser.parse(argc, argv);

	const auto config = load_config(opt["config"].as<std::string>());
	auto encoding_params = load_hp_floats(opt["encoding"].as<std::string>());
	auto network_params = load_floats(opt["network"].as<std::string>());
	const auto resolution = opt["resolution"].as<unsigned int>();
	const auto level = opt["level"].as<float>();
	const auto output = opt["output"].as<std::string>();

	const auto encoded_size = config.encoding.n_levels * config.encoding.n_features_per_level;
	const Encoding encoding(config.encoding, std::move(encoding_params));
	const Network network(config.network, std::move(network_params), encoded_size);

	const auto rect = get_rect(opt);
	const auto image = generate_image(encoding, network, rect, resolution, level);
	save_image(image, output);
}
