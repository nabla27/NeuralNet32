#ifndef NN_OPTIMIZER_ADADELTA_H
#define NN_OPTIMIZER_ADADELTA_H

#include <vector>
#include "vec/operator.h"
#include "vec/function.h"




namespace nn {
	namespace OPTIMIZER {





		class AdaDelta {
		private:
			vec::vector3d& weights;
			vec::vector2d& bias;
			vec::vector3d& dW;
			vec::vector2d& db;
		private:
			vec::vector3d Vw;
			vec::vector3d Uw;
			vec::vector2d Vb;
			vec::vector2d Ub;
			const double delta = 1e-7;
		public:
			AdaDelta(
				vec::vector3d& weights,
				vec::vector2d& bias,
				vec::vector3d& dW,
				vec::vector2d& db
			) : weights(weights), bias(bias), dW(dW), db(db) {}

			void _Init_() {
				vec::fitsize(Vw, weights, 1e-7);
				vec::fitsize(Uw, weights, 1e-7);
				vec::fitsize(Vb, bias, 1e-7);
				vec::fitsize(Ub, bias, 1e-7);
			}

			void update(
				const double rho = 0.95
			);
		};


		void AdaDelta::update(
			const double rho
		)
		{
			using namespace vec;
			const size_t numLayer = dW.size() - 1;

			for (size_t i = 0; i <= numLayer; ++i)
			{
				Vw[i] = rho * Vw[i] + (1 - rho) * dW[i] * dW[i];
				Vb[i] = rho * Vb[i] + (1 - rho) * db[i] * db[i];

				vec::vector2d Dw = vec::sqrt((Uw[i] + delta) / (Vw[i] + delta)) * dW[i];
				vec::vector1d Db = vec::sqrt((Ub[i] + delta) / (Vb[i] + delta)) * db[i];

				Uw[i] = rho * Uw[i] + (1 - rho) * Dw * Dw;
				Ub[i] = rho * Ub[i] + (1 - rho) * Db * Db;

				weights[i] -= Dw;
				bias[i] -= Db;
			}
		}







	} //OPTIMIZER
} //nn




#endif