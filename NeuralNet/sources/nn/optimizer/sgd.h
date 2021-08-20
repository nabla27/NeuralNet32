#ifndef NN_OPTIMIZER_SGD_H
#define NN_OPTIMIZER_SGD_H

#include <vector>
#include "vec/operator.h"
#include "vec/function.h"




namespace nn {
	namespace OPTIMIZER {





		class SGD {
		private:
			vec::vector3d& weights;
			vec::vector2d& bias;
			vec::vector3d& dW;
			vec::vector2d& db;
		public:
			SGD(
				vec::vector3d& weights,
				vec::vector2d& bias,
				vec::vector3d& dW,
				vec::vector2d& db
			) : weights(weights), bias(bias), dW(dW), db(db) {}

			void _Init_() {}
			
			void update(
				const double lr = 0.1
			);
		};



		void SGD::update(
			const double lr
		)
		{
			using namespace vec;
			const size_t numLayer = dW.size() - 1;

			for (size_t i = 0; i <= numLayer; ++i) {
				weights[i] -= dW[i] * lr;
				bias[i] -= db[i] * lr;
			}
		}







	} //OPTIMIZER
} //nn




#endif