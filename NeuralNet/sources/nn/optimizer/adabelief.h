#ifndef NN_OPTIMIZER_ADABELIEF_H
#define NN_OPTIMIZER_ADABELIEF_H

#include <vector>
#include "vec/operator.h"
#include "vec/function.h"




namespace nn {
	namespace OPTIMIZER {





		class AdaBelief {
		private:
			vec::vector3d& weights;
			vec::vector2d& bias;
			vec::vector3d& dW;
			vec::vector2d& db;
		private:
			vec::vector3d Mw;
			vec::vector3d Sw;
			vec::vector2d Mb;
			vec::vector2d Sb;
			const double delta = 1e-7;
			unsigned step = 0;
		public:
			AdaBelief(
				vec::vector3d& weights,
				vec::vector2d& bias,
				vec::vector3d& dW,
				vec::vector2d& db
			) : weights(weights), bias(bias), dW(dW), db(db) {}

			void _Init_() {
				vec::fitsize(Mw, weights, 1e-7);
				vec::fitsize(Sw, weights, 1e-7);
				vec::fitsize(Mb, bias, 1e-7);
				vec::fitsize(Sb, bias, 1e-7);
			}

			void update(
				const double alpha = 1e-3,
				const double beta1 = 0.9,
				const double beta2 = 0.999
			);
		};


		void AdaBelief::update(
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

				Sw[i] += (1 - beta2) * ((dW[i] - Mw[i]) * (dW[i] - Mw[i]) - Sw[i]);
				Sb[i] += (1 - beta2) * ((db[i] - Mb[i]) * (db[i] - Mb[i]) - Sb[i]);

				weights[i] -= alpha * sqrt(1 - pow(beta2, step)) / (1 - pow(beta1, step)) * Mw[i] / vec::sqrt(Sw[i] + delta);
				bias[i] -= alpha * sqrt(1 - pow(beta2, step)) / (1 - pow(beta1, step)) * Mb[i] / vec::sqrt(Sb[i] + delta);
			}
		}







	} //OPTIMIZER
} //nn




#endif