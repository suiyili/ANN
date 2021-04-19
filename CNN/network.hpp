#pragma once

#include "net_layer_i.hpp"
#include "layer_factory_i.hpp"
#include "input_layer.hpp"
#include <list>

using namespace cnn::layer;
namespace cnn::net
{
	class network final
	{
	public:
		network(const tensor_shape& shape);
		~network() = default;

		void create_layer(layer_factory_i& factory);
		size_t predict(const value_array& values) noexcept;
		void train(const value_array& values, size_t label) noexcept;
		void learn(float learning_rate) noexcept;

	private:
		static void forward(std::list<std::unique_ptr<net_layer_i>>& layers) noexcept;
		input_layer source_;
		std::list<std::unique_ptr<net_layer_i>> layers_;
	};
}