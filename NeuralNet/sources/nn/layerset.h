#ifndef NN_LAYERSET_H
#define NN_LAYERSET_H

#include <vector>
#include <string>
#include <random>
#include "vec/function.h"




namespace nn {





	enum class InitType { He, Xavier, Std, Unify };






	class LayerSet {
	public:
		//vec::vector2d train_x;
		//vec::vector2d train_t;
		size_t data_size = 0;
		size_t label_size = 0;
		vec::vector3d weights;
		vec::vector2d bias;
		std::vector<int> node;
	public:
		LayerSet(const size_t data_size, const size_t label_size) : data_size(data_size), label_size(label_size) {}
		LayerSet() {}

		//void set_data(const vec::vector2d& train_x, const vec::vector2d& train_t);

		inline void set_node(const std::vector<int>& node) {
			this->node = node;
		}

		void initialize(const InitType type = InitType::He, const double val = 0);
	};







	/*
	void LayerSet::set_data(const vec::vector2d& train_x, const vec::vector2d& train_t)
	{
		//訓練データとラベルのサイズが合わない場合の例外処理
		if (train_x.size() != train_t.size()) {
			std::cout << "ERROR : layerset.h(" << __LINE__ << ") <LayerSet::set_data>" << std::endl;
			throw std::runtime_error("The size of data and label must be the same.");
		}
		//空のデータが空であった場合の例外処理
		if (train_x.size() == 0) {
			std::cout << "ERROR : layerset.h(" << __LINE__ << ") <LayerSet::set_data>" << std::endl;
			throw std::runtime_error("The data is empty.");
		}

		this->train_x = train_x;
		this->train_t = train_t;
	}
	*/





	void LayerSet::initialize(const InitType type, const double val)
	{
		//データがセットされていない場合の例外処理
		if (data_size == 0) {
			std::cout << "ERROR : layerset.h(" << __LINE__ << ") <LayerSet::initialize>" << std::endl;
			throw std::runtime_error("Data and label must be set the size to initialize.");
		}

		std::random_device rnd;
		std::mt19937 mt(rnd());	//乱数のシード値を取得

		weights.clear(); bias.clear();	//重みとバイアスをリセット

		std::vector<size_t> index;	//重みとバイアスのサイズを一時記憶する配列

		index.push_back(data_size);
		for (size_t i = 0; i < node.size(); i++) { index.push_back(node[i]); }
		index.push_back(label_size);

		//重みとバイアスのサイズを確保
		size_t layer_size = index.size() - 1;
		for (size_t i = 0; i < layer_size; i++) {
			weights.push_back(vec::vector2d(index[i], vec::vector1d(index[i + 1])));
			bias.push_back(vec::vector1d(index[i + 1], 0));
		}

		//重みの各要素を正規分布で初期化する
		size_t size = weights.size();
		for (size_t i = 0; i < size; i++)
		{
			size_t row = weights[i].size();

			double variance = val;	//標準偏差
			if (type == InitType::He) { variance = sqrt(2 / (double)index[i]); }
			else if (type == InitType::Xavier) { variance = sqrt(1 / (double)index[i]); }
			else if (type == InitType::Std) { variance = val; }

			if (type != InitType::Unify)
			{
				std::normal_distribution<> norm(0.0, variance);	//標準正規分布で乱数を生成

				for (size_t j = 0; j < row; j++) {
					size_t col = weights[i][j].size();
					for (size_t k = 0; k < col; k++) {
						weights[i][j][k] = norm(mt);
					}
				}
			}
			else if (type == InitType::Unify)
			{
				for (size_t j = 0; j < row; j++) {
					size_t col = weights[i][j].size();
					for (size_t k = 0; k < col; k++) {
						weights[i][j][k] = val;
					}
				}
			}
		}
	}




}

#endif //NN_LAYERSET_H