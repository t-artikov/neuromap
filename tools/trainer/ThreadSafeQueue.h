#pragma once
#include <condition_variable>
#include <mutex>
#include <queue>

template <typename T>
class ThreadSafeQueue
{
public:
	void push(T v)
	{
		std::lock_guard<std::mutex> lock(mutex_);
		queue_.push(std::move(v));
		cv_.notify_all();
	}

	T pop()
	{
		std::unique_lock<std::mutex> lock(mutex_);
		cv_.wait(
			lock,
			[this]()
			{
				return !queue_.empty();
			});
		auto result = std::move(queue_.front());
		queue_.pop();
		return result;
	}

private:
	std::queue<T> queue_;
	std::mutex mutex_;
	std::condition_variable cv_;
};
