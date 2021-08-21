#ifndef NN_NETWORK_H
#define NN_NETWORK_H

#include "activation.h"
#include "layerset.h"
#include "optimizer.h"





namespace nn {




	template <
		class OptType = OPTIMIZER::SGD,
		class ActType = ACTIVATION::ReLU, 
		class OutType = ACTIVATION::Softmax
	>
	class Network :
		protected virtual Affine,
		protected virtual DropOut,
		protected virtual ActType,
		protected virtual OutType,
		public virtual OptType
	{
	private:
		vec::vector3d weights;
		vec::vector2d bias;

		vec::vector3d dW;
		vec::vector2d db;

		vec::vector2d forward_propagate;
		vec::vector2d backward_propagate;

		size_t numLayer = 0;
		double loss = 0;
	public:
		vec::vector2d out;
	public:
		Network(const LayerSet& layerset);

		void forward(
			const vec::vector2d& batch_x,
			const double dropout_ratio = 0,
			const bool train_flag = true
		);

		void backward(
			const vec::vector2d& batch_t,
			const double dropout_ratio = 0
		);

		void reset() {
			Affine::reset();
			DropOut::reset();
			ActType::reset();
			OutType::reset();
		}

		inline double get_loss() const { return loss; }
		
		LayerSet get_layerset() const;
	};










	//コンストラクタ
	template <class OptType, class ActType, class OutType>
	Network<OptType, ActType, OutType>::Network(const LayerSet& layerset) :
		numLayer(layerset.weights.size() - 1),
		weights(layerset.weights),
		bias(layerset.bias),
		Affine(this->weights, this->bias, this->dW, this->db),
		OptType(this->weights, this->bias, this->dW, this->db)
	{
		vec::fitsize(dW, layerset.weights);
		vec::fitsize(db, layerset.bias);
		OptType::_Init_();
	}





	//順伝播
	template <class OptType, class ActType, class OutType>
	void Network<OptType, ActType, OutType>::forward
	(
		const vec::vector2d& batch_x,
		const double dropout_ratio,
		const bool train_flag
	)
	{
		forward_propagate = batch_x;

		Affine::forward(forward_propagate);

		for (size_t i = 0; i < numLayer; ++i)
		{
			if (dropout_ratio != 0) { DropOut::forward(forward_propagate, dropout_ratio, train_flag); }

			ActType::forward(forward_propagate);

			Affine::forward(forward_propagate);
		}

		OutType::forward(forward_propagate);

		out = forward_propagate;
	}





	//逆伝播
	template <class OptType, class ActType, class OutType>
	void Network<OptType, ActType, OutType>::backward
	(
		const vec::vector2d& batch_t,
		const double dropout_ratio
	)
	{
		backward_propagate = batch_t;

		OutType::backward(backward_propagate);

		/* entoropy_error */
		double tmp = 0.0;
		const size_t row = out.size();
		const size_t col = out[0].size();
		for (size_t i = 0; i < row; ++i) {
			for (size_t j = 0; j < col; ++j) {
				tmp += batch_t[i][j] * log(out[i][j] + 1e-7);
			}
		}
		loss = -tmp / batch_t.size();

		Affine::backward(backward_propagate);

		for (int i = (int)numLayer - 1; i >= 0; --i)
		{
			ActType::backward(backward_propagate);

			if (dropout_ratio != 0) { DropOut::backward(backward_propagate); }

			Affine::backward(backward_propagate);
		}
	}




	//LayerSet型のオブジェクトで、重みやバイアスの情報を返す
	template <class OptType, class ActType, class OutType>
	LayerSet Network<OptType, ActType, OutType>::get_layerset() const
	{
		LayerSet layerset;
		layerset.weights = this->weights;
		layerset.bias = this->bias;
		for (size_t i = 1; i < weights.size(); ++i) {
			layerset.node.push_back(static_cast<int>(weights[i].size()));
		}

		return layerset;
	}












} //nn











#endif //NN_NETWORK_H
