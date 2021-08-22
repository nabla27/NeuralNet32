#ifndef NN_OPTIMIZER_RMSPROPGRAVES_H
#define NN_OPTIMIZER_RMSPROPGRAVES_H

#include <vector>
#include "vec/operator.h"
#include "vec/function.h"




namespace nn {
	namespace OPTIMIZER {





		class RMSpropGraves {
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
		public:
			RMSpropGraves(
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
				const double eta = 1e-4,
				const double rho = 0.95
			);
		};


		void RMSpropGraves::update(
			const double eta,
			const double rho
		)
		{
			using namespace vec;
			const size_t numLayer = dW.size() - 1;

			for (size_t i = 0; i <= numLayer; ++i) 
			{
				Mw[i] = rho * Mw[i] + (1 - rho) * dW[i];
				Mb[i] = rho * Mb[i] + (1 - rho) * db[i];

				Vw[i] = rho * Vw[i] + (1 - rho) * dW[i] * dW[i];
				Vb[i] = rho * Vb[i] + (1 - rho) * db[i] * db[i];

				weights[i] -= eta * dW[i] / vec::sqrt(Vw[i] - Mw[i] * Mw[i] + delta);
				bias[i] -= eta * db[i] / vec::sqrt(Vb[i] - Mb[i] * Mb[i] + delta);
			}
		}







	} //OPTIMIZER
} //nn




#endif