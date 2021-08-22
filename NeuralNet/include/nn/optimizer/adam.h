#ifndef NN_OPTIMIZER_ADAM_H
#define NN_OPTIMIZER_ADAM_H

#include <vector>
#include "vec/operator.h"
#include "vec/function.h"




namespace nn {
	namespace OPTIMIZER {





		class Adam {
		private:
			vec::vector3d& weights;
			vec::vector2d& bias;
			vec::vector3d& dW;
			vec::vector2d& db;
		private:
			vec::vector3d Mw;
			vec::vector3d Vw;
			vec::vector2d Mb;
			vec::vector2d Vb;
			const double delta = 1e-7;
			const double alpha = 1e-3;
			unsigned step = 0;
		public:
			Adam(
				vec::vector3d& weights,
				vec::vector2d& bias,
				vec::vector3d& dW,
				vec::vector2d& db
			) : weights(weights), bias(bias), dW(dW), db(db) {}

			void _Init_() {
				vec::fitsize(Mw, weights, 1e-7);
				vec::fitsize(Vw, weights, 1e-7);
				vec::fitsize(Mb, bias, 1e-7);
				vec::fitsize(Vb, bias, 1e-7);
			}

			void update(
				const double beta1 = 0.9,
				const double beta2 = 0.999
			);
		};


		void Adam::update(
			const double beta1,
			const double beta2
		)
		{
			using namespace vec;
			const size_t numLayer = dW.size() - 1;

			step++;
			for (size_t i = 0; i <= numLayer; ++i) {
				Mw[i] = beta1 * Mw[i] + (1 - beta1) * dW[i];
				Mb[i] = beta1 * Mb[i] + (1 - beta1) * db[i];

				Vw[i] = beta2 * Vw[i] + (1 - beta2) * dW[i] * dW[i];
				Vb[i] = beta2 * Vb[i] + (1 - beta2) * db[i] * db[i];

				weights[i] -= alpha * sqrt(1 - pow(beta2, step)) / (1 - pow(beta1, step)) * Mw[i] / vec::sqrt(Vw[i] + delta);
				bias[i] -= alpha * sqrt(1 - pow(beta2, step)) / (1 - pow(beta1, step)) * Mb[i] / vec::sqrt(Vb[i] + delta);
			}
		}







	} //OPTIMIZER
} //nn




#endif