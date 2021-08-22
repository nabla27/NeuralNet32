#ifndef NN_OPTIMIZER_NADAM_H
#define NN_OPTIMIZER_NADAM_H

#include <vector>
#include "vec/operator.h"
#include "vec/function.h"




namespace nn {
	namespace OPTIMIZER {





		class NAdam {
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
			unsigned step = 0;
		public:
			NAdam(
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
				const double alpha = 1e-3,
				const double mu = 0.975,
				const double nu = 0.999
			);
		};


		void NAdam::update(
			const double alpha,
			const double mu,
			const double nu
		)
		{
			using namespace vec;
			const size_t numLayer = dW.size() - 1;

			step++;
			for (size_t i = 0; i <= numLayer; ++i) 
			{
				Mw[i] += (1 - mu) * (dW[i] - Mw[i]);
				Mb[i] += (1 - mu) * (db[i] - Mb[i]);

				Vw[i] += (1 - nu) * (dW[i] * dW[i] - Vw[i]);
				Vb[i] += (1 - nu) * (db[i] * db[i] - Vb[i]);

				vec::vector2d HMw = mu / (1 - pow(mu, step + 1)) * Mw[i] + (1 - mu) / (1 - pow(mu, step)) * dW[i];
				vec::vector1d HMb = mu / (1 - pow(mu, step + 1)) * Mb[i] + (1 - mu) / (1 - pow(mu, step)) * db[i];

				vec::vector2d HVw = nu / (1 - pow(nu, step)) * Vw[i];
				vec::vector1d HVb = nu / (1 - pow(nu, step)) * Vb[i];

				weights[i] -= alpha * HMw / vec::sqrt(HVw + delta);
				bias[i] -= alpha * HMb / vec::sqrt(HVb + delta);
			}
		}







	} //OPTIMIZER
} //nn




#endif