#pragma once
#include <stddef.h>

#include <tiny-cuda-nn/common.h>

template <class T>
class HostMemory
{
public:
	HostMemory(const HostMemory<T> & other) = delete;
	HostMemory(HostMemory<T> && other)
		: size_(other.size_)
		, data_(other.data_)
	{
		other.size_ = 0;
		other.data_ = nullptr;
	}

	HostMemory()
		: HostMemory(0)
	{
	}

	explicit HostMemory(size_t size)
		: size_(size)
	{
		if (size != 0)
		{
			CUDA_CHECK_THROW(cudaHostAlloc(reinterpret_cast<void **>(&data_), size * sizeof(T), 0));
		}
	}

	HostMemory<T> & operator=(const HostMemory<T> & other) = delete;

	HostMemory<T> & operator=(HostMemory<T> && other)
	{
		size_ = other.size_;
		data_ = other.data_;
		other.size_ = 0;
		other.data_ = nullptr;
	}

	~HostMemory()
	{
		if (data_)
		{
			cudaFreeHost(data_);
		}
	}

	bool empty() const
	{
		return size_ == 0;
	}

	size_t size() const
	{
		return size_;
	}

	T * data()
	{
		return data_;
	}

	const T * data() const
	{
		return data_;
	}

private:
	size_t size_;
	T * data_ = nullptr;
};
