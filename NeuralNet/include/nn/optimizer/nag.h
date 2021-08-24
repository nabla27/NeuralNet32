/*  LICENSE
	Copyright (c) 2021, nabla All rights reserved.
	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/
#ifndef NN_OPTIMIZER_NAG_H
#define NN_OPTIMIZER_NAG_H

#include <vector>
#include "vec/operator.h"
#include "vec/function.h"




namespace nn {
	namespace OPTIMIZER {





		class NAG {
		private:
			vec::vector3d& weights;
			vec::vector2d& bias;
			vec::vector3d& dW;
			vec::vector2d& db;
		private:
			vec::vector3d Speed_W;
			vec::vector2d Speed_b;
		public:
			NAG(
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
				const double alpha = 0.9,
				const double beta = 0.01
			);
		};


		void NAG::update(
			const double alpha,
			const double beta
		)
		{
			using namespace vec;
			const size_t numLayer = dW.size() - 1;

			for (size_t i = 0; i <= numLayer; ++i) {
				Speed_W[i] = alpha * Speed_W[i] + dW[i];
				Speed_b[i] = alpha * Speed_b[i] + db[i];

				weights[i] -= beta * (alpha * Speed_W[i] + dW[i]);
				bias[i] -= beta * (alpha * Speed_b[i] + db[i]);
			}
		}







	} //OPTIMIZER
} //nn




#endif