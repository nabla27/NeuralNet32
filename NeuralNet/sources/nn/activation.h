#ifndef NN_ACTIVATION_H
#define NN_ACTIVATION_H

#include "vec/function.h"
#include "vec/operator.h"





namespace nn {
	namespace ACTIVATION {




		class Affine {
		private:
			vec::vector3d Memory_X;
			int count = 0;
			vec::vector3d& weights;
			vec::vector2d& bias;
			vec::vector3d& dW;
			vec::vector2d& db;
		public:
			Affine(
				vec::vector3d& weights,
				vec::vector2d& bias,
				vec::vector3d& dW,
				vec::vector2d& db
			) : weights(weights), bias(bias), dW(dW), db(db) {}

			void forward(vec::vector2d& forward_propagate);

			void backward(vec::vector2d& backward_propagate);

			inline void reset() { Memory_X.clear(); count = 0; }
		};

		void Affine::forward(vec::vector2d& forward_propagate)
		{
			using namespace vec;

			Memory_X.push_back(forward_propagate);
			forward_propagate = vec::dot(forward_propagate, weights[count]) + bias[count];
			count++;
		}

		void Affine::backward(vec::vector2d& backward_propagate)
		{
			count--;

			dW[count] = vec::dot(vec::trans(Memory_X[count]), backward_propagate);

			vec::vector1d tmp_db(backward_propagate[0].size());
			size_t max_a = backward_propagate[0].size();
			size_t max_b = backward_propagate.size();
			for (size_t i = 0; i < max_a; ++i) {
				for (size_t j = 0; j < max_b; ++j) {
					tmp_db[i] += backward_propagate[j][i];
				}
			}
			db[count] = tmp_db;

			backward_propagate = vec::dot(backward_propagate, vec::trans(weights[count]));
		}












		class ReLU {
		private:
			vec::vector3d Memory_mask;
			int count = 0;
		public:
			void forward(vec::vector2d& forward_propagate);
			void backward(vec::vector2d& backward_propagate);
			inline void reset() { Memory_mask.clear(); count = 0; }
		};

		void ReLU::forward(vec::vector2d& forward_propagate)
		{
			Memory_mask.push_back(forward_propagate);

			size_t max_a = forward_propagate.size();
			size_t max_b = forward_propagate[0].size();
			for (size_t i = 0; i < max_a; ++i) {
				for (size_t j = 0; j < max_b; ++j) {
					if (forward_propagate[i][j] < 0) { forward_propagate[i][j] = 0; }
				}
			}

			count++;
		}

		void ReLU::backward(vec::vector2d& backward_propagate)
		{
			count--;

			size_t max_a = Memory_mask[count].size();
			size_t max_b = Memory_mask[count][0].size();
			for (size_t i = 0; i < max_a; ++i) {
				for (size_t j = 0; j < max_b; ++j) {
					if (Memory_mask[count][i][j] < 0) { backward_propagate[i][j] = 0; }
				}
			}
		}









		class tanhExp {
		private:
			vec::vector3d Memory_mask;
			int count = 0;
		public:
			void forward(vec::vector2d& forward_propagate);
			void backward(vec::vector2d& backward_propagate);
			inline void reset() { Memory_mask.clear(); count = 0; }
		};

		void tanhExp::forward(vec::vector2d& forward_propagate)
		{
			Memory_mask.push_back(forward_propagate);

			size_t max_a = forward_propagate.size();
			size_t max_b = forward_propagate[0].size();
			for (size_t i = 0; i < max_a; ++i) {
				for (size_t j = 0; j < max_b; ++j) {
					//オーバーフロー対策
					if (forward_propagate[i][j] > 3) {}
					else if (forward_propagate[i][j] < -25) { forward_propagate[i][j] = 0; }
					else { forward_propagate[i][j] *= tanh(exp(forward_propagate[i][j])); }
				}
			}

			count++;
		}

		void tanhExp::backward(vec::vector2d& backward_propagate)
		{
			count--;

			size_t max_a = Memory_mask[count].size();
			size_t max_b = Memory_mask[count][0].size();
			for (size_t i = 0; i < max_a; ++i) {
				for (size_t j = 0; j < max_b; ++j) {
					//オーバーフロー対策
					if (Memory_mask[count][i][j] > 3) {}
					else if (Memory_mask[count][i][j] < -25) { backward_propagate[i][j] = 0; }
					else {
						backward_propagate[i][j] *=
							tanh(exp(Memory_mask[count][i][j]))
							- Memory_mask[count][i][j] * exp(Memory_mask[count][i][j])
							* (tanh(exp(Memory_mask[count][i][j])) * tanh(exp(Memory_mask[count][i][j])) - 1);
					}
				}
			}
		}







		class Softmax {
		private:
			vec::vector3d Memory_soft;
			int count = 0;
		public:
			void forward(vec::vector2d& forward_propagate);
			void backward(vec::vector2d& backward_propagate);
			inline void reset() { Memory_soft.clear(); count = 0; }
		};

		void Softmax::forward(vec::vector2d& forward_propagate)
		{
			//順伝播の各行iの最大値をmax[i]に格納
			vec::vector1d max(forward_propagate.size());
			size_t max_a = forward_propagate.size();
			size_t max_b = forward_propagate[0].size();
			for (size_t i = 0; i < max_a; ++i) {
				for (size_t j = 0; j < max_b; ++j) {
					if (max[i] < forward_propagate[i][j]) { max[i] = forward_propagate[i][j]; }
				}
			}

			//ソフトマックス関数の出力
			double Max = 0.0;
			for (size_t i = 0; i < max_a; ++i)
			{
				Max = max[i];
				double Deno = 0.0;
				for (size_t j = 0; j < max_b; ++j) {
					Deno += exp(forward_propagate[i][j] - Max);
				}
				for (size_t j = 0; j < max_b; ++j) {
					forward_propagate[i][j] = exp(forward_propagate[i][j] - Max) / (Deno + 1e-7);
				}
			}

			Memory_soft.push_back(forward_propagate);

			count++;
		}

		void Softmax::backward(vec::vector2d& backward_propagate)
		{
			count--;

			size_t row = backward_propagate.size();
			size_t col = backward_propagate[0].size();
			for (size_t i = 0; i < row; ++i) {
				for (size_t j = 0; j < col; ++j) {
					backward_propagate[i][j] = (Memory_soft[count][i][j] - backward_propagate[i][j]) / (double)row;
				}
			}
		}








		class DropOut {
		private:
			vec::vector3d mask_drop;
			int count = 0;
		public:
			void forward(vec::vector2d& forward_propagate, const double ratio, const bool train_flag);
			inline void backward(vec::vector2d& backward_propagate);
			inline void reset() { mask_drop.clear(); count = 0; }
		};

		void DropOut::forward(vec::vector2d& forward_propagate, const double ratio, const bool train_flag)
		{
			if (train_flag) {
				using namespace vec;

				size_t row = forward_propagate.size();
				size_t col = forward_propagate[0].size();
				vec::vector2d tmp_mask(row, vec::vector1d(col, 0));

				std::random_device rnd;	//非決定的な乱数生成器
				std::mt19937 mt(rnd());
				std::uniform_real_distribution<> rand(0, 1);	//0から1の一様な乱数

				for (size_t i = 0; i < row; ++i) {
					for (size_t j = 0; j < col; ++j) {
						if (rand(mt) > ratio) { tmp_mask[i][j] = 1; }
					}
				}
				//VEC::show(tmp_mask, "tmp_mask");

				mask_drop.push_back(tmp_mask);
				forward_propagate *= tmp_mask;

				count++;
			}
			else {
				using namespace vec;
				forward_propagate = forward_propagate * (1 - ratio);
			}
		}

		inline void DropOut::backward(vec::vector2d& backward_propagate)
		{
			using namespace vec;

			count--;

			backward_propagate *= mask_drop[count];
		}







	} //ACTIVATION
} //nn

#endif //NN_ACTIVATION_H