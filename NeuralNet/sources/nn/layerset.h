#ifndef NN_LAYERSET_H
#define NN_LAYERSET_H

#include <vector>
#include <string>
#include <random>
#include "vec/function.h"
#include "util/exchanding.h"




namespace nn {





	enum class InitType 
	{ 
		He,      //�O�w�̃j���[��������n�Ƃ��āA�W���΍�sqrt{2/n}�̃K�E�X���z�ŏ�����
		Xavier,  //�O�w�̃j���[��������n�Ƃ��āA�W���΍�sqrt{1/n}�̃K�E�X���z�ŏ�����
		Std,     //�����Ŏw�肵���W���΍��̃K�E�X���z�ŏ�����
		Unify    //�����Ŏw�肵���l�ňꗥ�ɏ�����
	};






	class LayerSet {
	public:
		size_t data_size = 0;
		size_t label_size = 0;
		vec::vector3d weights;
		vec::vector2d bias;
		std::vector<int> node;
	public:
		LayerSet(const size_t data_size, const size_t label_size) : data_size(data_size), label_size(label_size) {}
		LayerSet() {}

		inline void set_node(const std::vector<int>& node) { this->node = node; }

		void initialize(const InitType type = InitType::He, const double val = 0);
	};








	void LayerSet::initialize(const InitType type, const double val)
	{
		//�f�[�^���Z�b�g����Ă��Ȃ��ꍇ�̗�O����
		if (data_size == 0) { exchandling::empty_data(__FILE__, __LINE__, "LayerSet::initialize"); }

		weights.clear(); bias.clear();  //�d�݂ƃo�C�A�X�����Z�b�g

		std::vector<size_t> index;      //�d�݂ƃo�C�A�X�̃T�C�Y���ꎞ�L������z��

		index.push_back(data_size);
		for (size_t i = 0; i < node.size(); i++) { index.push_back(node[i]); }
		index.push_back(label_size);

		//�d�݂ƃo�C�A�X�̃T�C�Y���m��
		const size_t layer_size = index.size() - 1;
		for (size_t i = 0; i < layer_size; i++) {
			weights.push_back(vec::vector2d(index[i], vec::vector1d(index[i + 1])));
			bias.push_back(vec::vector1d(index[i + 1], 0));
		}

		//�d�݂̊e�v�f������������
		const size_t size = weights.size();
		for (size_t i = 0; i < size; i++)
		{
			const size_t row = weights[i].size();

			//�W���΍��̎w��
			double variance = val;
			switch ((int)type) {
			case 0: variance = sqrt(2 / (double)index[i]); break;
			case 1: variance = sqrt(1 / (double)index[i]); break;
			case 2: variance = val; break;
			default: break;
			}

			if (type != InitType::Unify) { vec::initgauss(weights[i], 0.0, variance); }
			else if (type == InitType::Unify) { vec::initequal(weights, val); }
		}
	}




}

#endif //NN_LAYERSET_H