#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <unordered_map>

#include <cxxopts.hpp>
#include <stb_image_write.h>

struct Rect
{
	double left;
	double top;
	double right;
	double bottom;

	double width() const
	{
		return right - left;
	}

	double height() const
	{
		return bottom - top;
	}
};


struct Color
{
	uint32_t value = 0xFF000000;
};


Color to_color(uint8_t index)
{
	switch (index)
	{
		case 1: return Color{0xFF0000FF};
		case 2: return Color{0xFF00FF00};
		case 3: return Color{0xFFFF0000};
	}
	return Color{0xFFFFFFFF};
}

struct Tile
{
	static constexpr uint32_t resolution = 32;
	static constexpr uint32_t pixel_count = resolution * resolution;
	static constexpr uint32_t data_size = pixel_count / 4;

	uint32_t x;
	uint32_t y;
	std::array<uint8_t, data_size> data;

	Color get_color(double x, double y) const
	{
		const auto ix = static_cast<uint32_t>(x * resolution);
		const auto iy = static_cast<uint32_t>(y * resolution);
		const auto subindex = ix % 4;
		const auto c = (data[resolution / 4 * (resolution - 1 - iy) + ix / 4] >> (subindex * 2)) & 3;
		return to_color(c);
	}
};

namespace std
{

template<>
struct hash<pair<uint32_t, uint32_t>>{
	size_t operator() (pair<uint32_t, uint32_t> v) const
	{
		return std::hash<uint64_t>()(static_cast<uint64_t>(v.first) << 32 | v.second);
	}
};
}

struct Map
{
	using PositionToIndexMap = std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t>;

	unsigned int resolution;
	std::vector<uint8_t> empty_tiles;
	std::vector<Tile> tiles;
	PositionToIndexMap position_to_tile_index;

	Color get_color(double x, double y) const
	{
		const auto ix = static_cast<uint32_t>(x * resolution);
		const auto iy = static_cast<uint32_t>(y * resolution);
		const auto subindex = ix % 4;
		const auto c = (empty_tiles[resolution / 4 * iy + ix / 4] >> (subindex * 2)) & 3;
		if (c)
		{
			return to_color(c);
		}

		const auto it = position_to_tile_index.find({ix, iy});
		if (it != position_to_tile_index.end())
		{
			const auto & tile = tiles[it->second];
			return tile.get_color(x * resolution - ix, y * resolution - iy);
		}

		return Color{0xFFFFFFFF};
	}
};

struct Image
{
	unsigned int width;
	unsigned int height;
	std::vector<Color> pixels;
};

template <class T>
std::vector<T> read_file_content(const std::string & path)
{
	std::ifstream f(path, std::ios_base::binary);
	f.seekg(0, std::ios::end);
	const size_t size = f.tellg();
	f.seekg(0, std::ios::beg);
	std::vector<T> result(size / sizeof(T));
	f.read(reinterpret_cast<char*>(result.data()), size);
	if (!f)
	{
		throw std::runtime_error("Failded to read file content");
	}
	return result;
}

Map load_map(const std::string & empty_tiles_path, const std::string & tiles_path)
{
	(void)tiles_path;
	auto empty_tiles = read_file_content<uint8_t>(empty_tiles_path);
	const auto resolution = static_cast<unsigned int>(std::sqrt(empty_tiles.size() * 4));
	auto tiles = read_file_content<Tile>(tiles_path);
	Map::PositionToIndexMap position_to_tile_index;

	uint32_t index = 0;
	for (const auto & tile : tiles)
	{
		position_to_tile_index.insert({std::make_pair(tile.x, tile.y), index});
		++index;
	}

	return Map {
		resolution,
		std::move(empty_tiles),
		std::move(tiles),
		std::move(position_to_tile_index)
	};
}

Image generate_image(const Map & map, const Rect & rect, unsigned int resolution)
{
	const auto width = rect.width() >= rect.height() ? resolution : static_cast<unsigned int>(resolution * rect.width() / rect.height());
	const auto height = rect.height() >= rect.width() ? resolution : static_cast<unsigned int>(resolution * rect.height() / rect.width());
	Image image {width, height, std::vector<Color>(width * height)};
	for (unsigned int iy = 0; iy < height; ++iy)
	{
		const auto y = (iy + 0.5) / height * rect.height() + rect.top;
		for (unsigned int ix = 0; ix < width; ++ix)
		{
			const auto x = (ix + 0.5) / width * rect.width() + rect.left;
			const auto color = map.get_color(x, y);
			image.pixels[iy * width + ix] = color;
		}
	}
	return image;
}

void save_image(const Image & image, const std::string & path)
{
	const auto ok = stbi_write_png(path.c_str(), image.width, image.height, 4, image.pixels.data(), image.width * 4);
	if (!ok)
	{
		throw std::runtime_error("Failed to save image");
	}
}

Rect get_rect(const cxxopts::ParseResult & opt)
{
	const auto rect = Rect {
		opt["left"].as<double>(),
		opt["top"].as<double>(),
		opt["right"].as<double>(),
		opt["bottom"].as<double>()
	};
	if (rect.left > rect.right || rect.top > rect.bottom)
	{
		throw std::runtime_error("Invalid rect");
	}
	return rect;
}

int main(int argc, const char ** argv)
{
	cxxopts::Options options_parser("tiles_to_image");
	options_parser.add_options()
		("empty_tiles", "", cxxopts::value<std::string>())
		("tiles", "", cxxopts::value<std::string>())
		("output", "", cxxopts::value<std::string>()->default_value("tiles.png"))
		("left", "", cxxopts::value<double>()->default_value("0.0"))
		("top", "", cxxopts::value<double>()->default_value("0.0"))
		("right", "", cxxopts::value<double>()->default_value("1.0"))
		("bottom", "", cxxopts::value<double>()->default_value("1.0"))
		("resolution", "", cxxopts::value<unsigned int>()->default_value("2000"));

	const auto opt = options_parser.parse(argc, argv);
	const auto rect = get_rect(opt);
	const auto resolution = opt["resolution"].as<unsigned int>();
	const auto output = opt["output"].as<std::string>();

	const auto map = load_map(
		opt["empty_tiles"].as<std::string>(),
		opt["tiles"].as<std::string>()
	);

	const auto image = generate_image(map, rect, resolution);
	save_image(image, output);
}
