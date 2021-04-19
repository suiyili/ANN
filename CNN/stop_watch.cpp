#include "stop_watch.hpp"

template<typename type>
static size_t elapse(std::chrono::time_point<std::chrono::system_clock> start);

void stop_watch::start()
{
	start_ = std::chrono::system_clock::now();
}

size_t stop_watch::end_in_milliseconds()
{
	return elapse<std::chrono::milliseconds>(start_);
}

size_t stop_watch::end_in_seconds()
{
	return elapse<std::chrono::seconds>(start_);
}

size_t stop_watch::end_in_minutes()
{
	return elapse<std::chrono::minutes>(start_);
}

template<typename type>
inline size_t elapse(std::chrono::time_point<std::chrono::system_clock> start)
{
	auto elapse = std::chrono::system_clock::now() - start;
	return std::chrono::duration_cast<type>(elapse).count();
}
