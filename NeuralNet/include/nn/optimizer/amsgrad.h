/*  LICENSE
	Copyright (c) 2021, nabla All rights reserved.
	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/
#ifndef NN_OPTIMIZER_AMSGRAD_H
#define NN_OPTIMIZER_AMSGRAD_H

#include <vector>
#include "vec/operator.h"
#include "vec/function.h"




namespace nn {
	namespace OPTIMIZER {





		class AMSGrad {
		private:
			vec::vector3d& weights;
			vec::vector2d& bias;
			vec::vector3d& dW;
			vec::vector2d& db;
		private:
			vec::vector3d Mw;
			vec::vector3d Vw;
			vec::vector3d HVw;
			vec::vector2d Mb;
			vec::vector2d Vb;
			vec::vector2d HVb;
			const double delta = 1e-7;
			unsigned step = 0;
		public:
			AMSGrad(
				vec::vector3d& weights,
				vec::vector2d& bias,
				vec::vector3d& dW,
				vec::vector2d& db
			) : weights(weights), bias(bias), dW(dW), db(db) {}

			void _Init_() {
				vec::fitsize(Mw, weights, 1e-7);
				vec::fitsize(Vw, weights, 1e-7);
				vec::fitsize(HVw, weights, 1e-7);
				vec::fitsize(Mb, bias, 1e-7);
				vec::fitsize(Vb, bias, 1e-7);
				vec::fitsize(HVb, bias, 1e-7);
			}

			void update(
				const double alpha = 1e-3,
				const double beta1 = 0.9,
				const double beta2 = 0.999
			);
		};


		void AMSGrad::update(
			const double alpha,
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

				Vw[i] += (1 - beta2) * (dW[i] * dW[i] - Vw[i]);
				Vb[i] += (1 - beta2) * (db[i] * db[i] - Vb[i]);

				HVw[i] = max(HVw[i], Vw[i]);
				HVb[i] = max(HVb[i], Vb[i]);

				weights[i] -= alpha / sqrt(step) * Mw[i] / vec::sqrt(HVw[i] + delta);
				bias[i] -= alpha / sqrt(step) * Mb[i] / vec::sqrt(HVb[i] + delta);
			}
		}







	} //OPTIMIZER
} //nn




#endif