#ifndef NN_OPTIMIZER_MOMENTUM_H
#define NN_OPTIMIZER_MOMENTUM_H

#include <vector>
#include "vec/operator.h"
#include "vec/function.h"




namespace nn {
	namespace OPTIMIZER {





		class Momentum {
		private:
			vec::vector3d& weights;
			vec::vector2d& bias;
			vec::vector3d& dW;
			vec::vector2d& db;
		private:
			vec::vector3d Speed_W;
			vec::vector2d Speed_b;
		public:
			Momentum(
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
				const double lr = 0.01,
				const double mt = 0.9
			);
		};


		void Momentum::update(
			const double lr,
			const double mt
		)
		{
			using namespace vec;
			const size_t numLayer = dW.size() - 1;

			for (size_t i = 0; i <= numLayer; ++i) {
				Speed_W[i] = (Speed_W[i] * mt) - (dW[i] * lr);
				Speed_b[i] = (Speed_b[i] * mt) - (db[i] * lr);
				weights[i] += Speed_W[i];
				bias[i] += Speed_b[i];
			}
		}







	} //OPTIMIZER
} //nn




#endif