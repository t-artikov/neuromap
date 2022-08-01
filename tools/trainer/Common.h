#pragma once
#include <fstream>
#include <vector>

#include <tiny-cuda-nn/gpu_memory.h>

template <class T>
std::vector<T> read_file_content(const std::string & path)
{
	std::ifstream f(path, std::ios_base::binary);
	f.seekg(0, std::ios::end);
	const size_t size = f.tellg();
	f.seekg(0, std::ios::beg);
	std::vector<T> result(size / sizeof(T));
	f.read(reinterpret_cast<char *>(result.data()), size);
	if (!f)
	{
		throw std::runtime_error("Failded to read file content");
	}
	return result;
}

template <class T>
tcnn::GPUMemory<T> to_gpu(const std::vector<T> & data)
{
	tcnn::GPUMemory<T> result(data.size());
	result.copy_from_host(data);
	return result;
}
