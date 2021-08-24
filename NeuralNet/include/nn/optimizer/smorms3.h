/*  LICENSE
	Copyright (c) 2021, nabla All rights reserved.
	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/
#ifndef NN_OPTIMIZER_SMORMS3_H
#define NN_OPTIMIZER_SMORMS3_H

#include <vector>
#include "vec/operator.h"
#include "vec/function.h"




namespace nn {
	namespace OPTIMIZER {





		// https://sifter.org/~simon/journal/20150420.html
		class SMORMS3 {
		private:
			vec::vector3d& weights;
			vec::vector2d& bias;
			vec::vector3d& dW;
			vec::vector2d& db;
		private:
			vec::vector3d Sw;
			vec::vector3d Mw;
			vec::vector3d Vw;
			vec::vector2d Sb;
			vec::vector2d Mb;
			vec::vector2d Vb;
			const double delta = 1e-7;
		public:
			SMORMS3(
				vec::vector3d& weights,
				vec::vector2d& bias,
				vec::vector3d& dW,
				vec::vector2d& db
			) : weights(weights), bias(bias), dW(dW), db(db) {}

			void _Init_() {
				vec::fitsize(Sw, weights, 1e-7);
				vec::fitsize(Mw, weights, 1e-7);
				vec::fitsize(Vw, weights, 1e-7);
				vec::fitsize(Sb, bias, 1e-7);
				vec::fitsize(Mb, bias, 1e-7);
				vec::fitsize(Vb, bias, 1e-7);
			}

			void update(
				const double eta = 1e-3
			);
		};


		void SMORMS3::update(
			const double eta
		)
		{
			using namespace vec;
			const size_t numLayer = dW.size() - 1;

			for (size_t i = 0; i <= numLayer; ++i)
			{
				vec::vector2d Rw = 1 / (Sw[i] + 1);
				vec::vector1d Rb = 1 / (Sb[i] + 1);

				Mw[i] = (1 - Rw) * Mw[i] + Rw * dW[i];
				Mb[i] = (1 - Rb) * Mb[i] + Rb * db[i];

				Vw[i] = (1 - Rw) * Vw[i] + Rw * dW[i] * dW[i];
				Vb[i] = (1 - Rb) * Vb[i] + Rb * db[i] * db[i];

				weights[i] -= vec::min(eta, (Mw[i] * Mw[i]) / (Vw[i] + delta)) / vec::sqrt(Vw[i] + delta) * dW[i];
				bias[i] -= vec::min(eta, (Mb[i] * Mb[i]) / (Vb[i] + delta)) / vec::sqrt(Vb[i] + delta) * db[i];

				Sw[i] = 1 + Sw[i] * (1 - (Mw[i] * Mw[i]) / (Vw[i] + delta));
				Sb[i] = 1 + Sb[i] * (1 - (Mb[i] * Mb[i]) / (Vb[i] + delta));
			}
		}







	} //OPTIMIZER
} //nn




#endif