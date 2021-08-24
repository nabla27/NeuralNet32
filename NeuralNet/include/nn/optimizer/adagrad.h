/*  LICENSE
	Copyright (c) 2021, nabla All rights reserved.
	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/
#ifndef NN_OPTIMIZER_ADAGRAD_H
#define NN_OPTIMIZER_ADAGRAD_H

#include <vector>
#include "vec/operator.h"
#include "vec/function.h"




namespace nn {
	namespace OPTIMIZER {





		class AdaGrad {
		private:
			vec::vector3d& weights;
			vec::vector2d& bias;
			vec::vector3d& dW;
			vec::vector2d& db;
		private:
			vec::vector3d Speed_W;
			vec::vector2d Speed_b;
			const double delta = 1e-7;
		public:
			AdaGrad(
				vec::vector3d& weights,
				vec::vector2d& bias,
				vec::vector3d& dW,
				vec::vector2d& db
			) : weights(weights), bias(bias), dW(dW), db(db) {}

			void _Init_() {
				vec::fitsize(Speed_W, weights, 1e-7);
				vec::fitsize(Speed_b, bias, 1e-7);
			}

			void update(
				const double lr = 0.01
			);
		};


		void AdaGrad::update(
			const double lr
		)
		{
			using namespace vec;
			const size_t numLayer = dW.size() - 1;

			for (size_t i = 0; i <= numLayer; ++i) {
				Speed_W[i] += dW[i] * dW[i];
				Speed_b[i] += db[i] * db[i];
				weights[i] -= lr * (1 / vec::sqrt(Speed_W[i] + delta)) * dW[i];
				bias[i] -= lr * (1 / vec::sqrt(Speed_b[i] + delta)) * db[i];
			}
		}







	} //OPTIMIZER
} //nn




#endif