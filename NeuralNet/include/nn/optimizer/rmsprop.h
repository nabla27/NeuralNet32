/*  LICENSE
	Copyright (c) 2021, nabla All rights reserved.
	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/
#ifndef NN_OPTIMIZER_RMSPROP_H
#define NN_OPTIMIZER_RMSPROP_H

#include <vector>
#include "vec/operator.h"
#include "vec/function.h"




namespace nn {
	namespace OPTIMIZER {





		class RMSprop {
		private:
			vec::vector3d& weights;
			vec::vector2d& bias;
			vec::vector3d& dW;
			vec::vector2d& db;
		private:
			vec::vector3d Vw;
			vec::vector2d Vb;
			const double delta = 1e-7;
		public:
			RMSprop(
				vec::vector3d& weights,
				vec::vector2d& bias,
				vec::vector3d& dW,
				vec::vector2d& db
			) : weights(weights), bias(bias), dW(dW), db(db) {}

			void _Init_() {
				vec::fitsize(Vw, weights, 1e-7);
				vec::fitsize(Vb, bias, 1e-7);
			}

			void update(
				const double alpha = 0.99,
				const double beta = 0.01
			);
		};


		void RMSprop::update(
			const double alpha,
			const double beta
		)
		{
			using namespace vec;
			const size_t numLayer = dW.size() - 1;

			for (size_t i = 0; i <= numLayer; ++i) {
				Vw[i] = ((alpha * Vw[i]) + ((1 - alpha) * (dW[i] * dW[i])));
				Vb[i] = ((alpha * Vb[i]) + ((1 - alpha) * (db[i] * db[i])));
				weights[i] -= ((beta / vec::sqrt(Vw[i] + delta)) * dW[i]);
				bias[i] -= ((beta / vec::sqrt(Vb[i] + delta)) * db[i]);
			}
		}







	} //OPTIMIZER
} //nn




#endif