/*  LICENSE
	Copyright (c) 2021, nabla All rights reserved.
	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/
#ifndef NN_OPTIMIZER_ADAMAX_H
#define NN_OPTIMIZER_ADAMAX_H

#include <vector>
#include "vec/operator.h"
#include "vec/function.h"




namespace nn {
	namespace OPTIMIZER {





		class AdaMax {
		private:
			vec::vector3d& weights;
			vec::vector2d& bias;
			vec::vector3d& dW;
			vec::vector2d& db;
		private:
			vec::vector3d Mw;
			vec::vector3d Uw;
			vec::vector3d absGw;
			vec::vector2d Mb;
			vec::vector2d Ub;
			vec::vector2d absGb;
			const double alpha = 1e-3;
			const double delta = 1e-7;
			unsigned step = 0;
		public:
			AdaMax(
				vec::vector3d& weights,
				vec::vector2d& bias,
				vec::vector3d& dW,
				vec::vector2d& db
			) : weights(weights), bias(bias), dW(dW), db(db) {}

			void _Init_() {
				vec::fitsize(Mw, weights, 1e-7);
				vec::fitsize(Uw, weights, 1e-7);
				vec::fitsize(absGw, weights, 1e-7);
				vec::fitsize(Mb, bias, 1e-7);
				vec::fitsize(Ub, bias, 1e-7);
				vec::fitsize(absGb, bias, 1e-7);
			}

			void update(
				const double beta1 = 0.9,
				const double beta2 = 0.999
			);
		};


		void AdaMax::update(
			const double beta1,
			const double beta2
		)
		{
			using namespace vec;
			const size_t numLayer = dW.size() - 1;

			step++;
			for (size_t i = 0; i <= numLayer; ++i) 
			{
				Mw[i] += (1 - beta1) * (dW[i] - Mw[i]);
				Mb[i] += (1 - beta1) * (db[i] - Mb[i]);

				Uw[i] = vec::max(beta2 * Uw[i], vec::abs(dW[i]));
				Ub[i] = vec::max(beta2 * Ub[i], vec::abs(db[i]));

				weights[i] -= alpha / (1 - pow(beta1, step)) * Mw[i] / (Uw[i] + delta);
				bias[i] -= alpha / (1 - pow(beta1, step)) * Mb[i] / (Ub[i] + delta);
			}
		}







	} //OPTIMIZER
} //nn




#endif