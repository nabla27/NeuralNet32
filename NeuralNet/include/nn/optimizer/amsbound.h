/*  LICENSE
	Copyright (c) 2021, nabla All rights reserved.
	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/
#ifndef NN_OPTIMIZER_AMSBOUND_H
#define NN_OPTIMIZER_AMSBOUND_H

#include <vector>
#include "vec/operator.h"
#include "vec/function.h"




namespace nn {
	namespace OPTIMIZER {





		// https://openreview.net/pdf?id=Bkg3g2R9FX
		class AMSBound {
		private:
			vec::vector3d& weights;
			vec::vector2d& bias;
			vec::vector3d& dW;
			vec::vector2d& db;
		private:
			vec::vector3d Mw;
			vec::vector3d Vw;
			vec::vector3d Hw;
			vec::vector2d Mb;
			vec::vector2d Vb;
			vec::vector2d Hb;
			unsigned step = 0;
		public:
			AMSBound(
				vec::vector3d& weights,
				vec::vector2d& bias,
				vec::vector3d& dW,
				vec::vector2d& db
			) : weights(weights), bias(bias), dW(dW), db(db) {}

			void _Init_() {
				vec::fitsize(Mw, weights, 1e-7);
				vec::fitsize(Vw, weights, 1e-7);
				vec::fitsize(Hw, weights, 1e-7);
				vec::fitsize(Mb, bias, 1e-7);
				vec::fitsize(Vb, bias, 1e-7);
				vec::fitsize(Hb, bias, 1e-7);
			}

			void update(
				const double alpha = 1e-3,
				const double eta = 1e-1,
				const double beta1 = 0.9,
				const double beta2 = 0.999
			);
		};


		void AMSBound::update(
			const double alpha,
			const double eta,
			const double beta1,
			const double beta2
		)
		{
			using namespace vec;
			const size_t numLayer = dW.size() - 1;

			step++;
			for (size_t i = 0; i <= numLayer; ++i)
			{
				Mw[i] = beta1 * Mw[i] + (1 - beta1) * dW[i];
				Mb[i] = beta1 * Mb[i] + (1 - beta1) * db[i];

				Vw[i] = beta2 * Vw[i] + (1 - beta2) * dW[i] * dW[i];
				Vb[i] = beta2 * Vb[i] + (1 - beta2) * db[i] * db[i];

				Hw[i] = vec::max(Hw[i], Vw[i]);
				Hb[i] = vec::max(Hb[i], Vb[i]);

				double Emin = eta * (1 - 1 / ((1 - beta2) * step + 1));
				double Emax = eta * (1 + 1 / ((1 - beta2) * step));

				vec::vector2d Ew = vec::clip(alpha / vec::sqrt(Hw[i]), Emin, Emax);
				vec::vector1d Eb = vec::clip(alpha / vec::sqrt(Hb[i]), Emin, Emax);

				weights[i] -= Ew / sqrt(step) * Mw[i];
				bias[i] -= Eb / sqrt(step) * Mb[i];
			}
		}







	} //OPTIMIZER
} //nn




#endif