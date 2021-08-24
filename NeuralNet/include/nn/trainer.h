/*  LICENSE
	Copyright (c) 2021, nabla All rights reserved.
	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/
#ifndef NN_TRAINER_H
#define NN_TRAINER_H

#include <time.h>
#include <conio.h>
#include "network.h"
#include "io/iotxt.h"
#include "io/ioxml.h"
#include "util/timer.h"
#include "util/dir.h"
#include "util/cast.h"

/* �R���p�C���̃o�[�W�������m�F */
#ifndef HAS_CPLUS_17
#define HAS_CPLUS_17 0
#endif

#ifdef _MSVC_LANG
#if (_MSVC_LANG >= 201703L)	 //c++17 or later

#undef HAS_CPLUS_17
#define HAS_CPLUS_17 1

#endif
#endif

#ifdef __cplusplus
#if (__cplusplus >= 201703L) //c++17 or later

#undef HAS_CPLUS_17
#define HAS_CPLUS_17 1

#endif
#endif

#if HAS_CPLUS_17
#include <filesystem>
#else
#include <sys/stat.h>
#endif






namespace nn {





	//�P���̐ݒ�
	struct TrainCustom 
	{
		unsigned learning_step = 10000;     //�w�K��
		unsigned batch_size = 0;            //�o�b�`�T�C�Y
		double dropout_ratio = 0;           //�h���b�v�A�E�g�̊���
		unsigned acc_span = 100;            //���x���v�Z����p�x
		unsigned acc_type = 0;              //���x�v�Z�̃^�C�v

#ifdef HAS_BOOST_HEADER
		unsigned xml_span = 0;              //xml�t�@�C����
		float xmlout_inf = 0.97f;           //xml�t�@�C�����o�͂��鐸�x�̉���
#endif
	} default_custom;











	template <
		class OptType = OPTIMIZER::SGD,
		class ActType = ACTIVATION::ReLU,
		class OutType = ACTIVATION::Softmax
	>
	class Trainer {
	private:
		vec::vector2d train_x;
		vec::vector2d train_t;
		vec::vector2d test_x;
		vec::vector2d test_t;
		std::string output_path = "./train";
	public:
		Network<OptType, ActType, OutType> network;
	public:
		Trainer(const LayerSet& layerset) :
			network(layerset) {}

		void set_TrainData(
			const vec::vector2d& train_x,
			const vec::vector2d& train_t
		);

		void set_TestData(
			const vec::vector2d& test_x,
			const vec::vector2d& test_t
		);

		void set_OutputPath(const std::string output_path);

		void train(const TrainCustom custom = default_custom);

		float accuracy_t(
			const vec::vector2d& test_x,
			const vec::vector2d& test_t,
			const double threshold = 0.8
		);

		float accuracy_m(
			const vec::vector2d& test_x,
			const vec::vector2d& test_t
		);
	};











	//�P���f�[�^�̐ݒ�
	template <class OptType, class ActType, class OutType>
	inline void Trainer<OptType, ActType, OutType>::set_TrainData(
		const vec::vector2d& train_x,
		const vec::vector2d& train_t
	)
	{
		//�P���f�[�^�Ƃ��̃��x���̃T�C�Y������Ȃ��ꍇ�̗�O����
		if (train_x.size() != train_t.size()) { exchandling::mismatch_data_size(__FILE__, __LINE__, "Trainer::trian_data"); }
		//�f�[�^����ł���ꍇ�̗�O����
		if (train_x.size() == 0) { exchandling::empty_data(__FILE__, __LINE__, "Trainer::train_data"); }
		//�P���f�[�^�̃T�C�Y�����͑w�̃j���[�������ƈقȂ鎞�̗�O����
		if (train_x[0].size() != network.get_input_size()) { exchandling::invalid_data_size(__FILE__, __LINE__, "Trainer::train_data"); }
		//���x���̃T�C�Y���o�͑w�̃j���[�������ƈقȂ鎞�̗�O����
		if (train_t[0].size() != network.get_output_size()) { exchandling::invalid_data_size(__FILE__, __LINE__, "Trainer::train_data"); }

		this->train_x = train_x;
		this->train_t = train_t;
	}






	//�e�X�g�p�f�[�^�̐ݒ�
	template <class OptType, class ActType, class OutType>
	inline void Trainer<OptType, ActType, OutType>::set_TestData(
		const vec::vector2d& test_x,
		const vec::vector2d& test_t
	)
	{
		//�e�X�g�f�[�^�Ƃ��̃��x���̃T�C�Y������Ȃ��ꍇ�̗�O����
		if (test_x.size() != test_t.size()) { exchandling::mismatch_data_size(__FILE__, __LINE__, "Trainer::test_data"); }
		//�f�[�^����ł���ꍇ�̗�O����
		if (test_x.size() == 0) { exchandling::empty_data(__FILE__, __LINE__, "Trainer::test_data"); }
		//�e�X�g�f�[�^�̃T�C�Y�����͑w�̃j���[�������ƈقȂ鎞�̗�O����
		if (test_x[0].size() != network.get_input_size()) { exchandling::invalid_data_size(__FILE__, __LINE__, "Trainer::test_data"); }
		//���x���̃T�C�Y���o�͑w�̃j���[�������ƈقȂ鎞�̗�O����
		if (test_t[0].size() != network.get_output_size()) { exchandling::invalid_data_size(__FILE__, __LINE__, "Trainer::test_data"); }

		this->test_x = test_x;
		this->test_t = test_t;
	}






	//�o�͂���t�@�C���̃p�X�̐ݒ�
	template <class OptType, class ActType, class OutType>
	inline void Trainer<OptType, ActType, OutType>::set_OutputPath(
		const std::string output_path
	)
	{
#if HAS_CPLUS_17
		if (!std::filesystem::is_directory(get_parentdir(output_path))) { 
			std::filesystem::create_directory(get_parentdir(output_path));
			std::cout << "created a directory ... " << get_parentdir(output_path) << std::endl;
		}
		this->output_path = output_path;
#else
		struct stat statBuf;
		const char* directory_name = get_parentdir(output_path).c_str();
		
		if (stat(directory_name, &statBuf) != 0) {
			//�f�B���N�g�������݂��Ȃ��ꍇ�̗�O����
			exchandling::not_exist_path(__FILE__, __LINE__, "Trainer::set_OutPath");
		}
		else { this->output_path = output_path; }
#endif
	}







	//�P��
	template <class OptType, class ActType, class OutType>
	void Trainer<OptType, ActType, OutType>::train(const TrainCustom custom)
	{
		/* �f�[�^���Z�b�g����Ă��Ȃ������ꍇ�̗�O���� */
		if (train_x.size() == 0) { exchandling::empty_data(__FILE__, __LINE__, "Trainer::train"); }

		/* �w�肵���o�b�`�T�C�Y���f�[�^�T�C�Y�����傫�������ꍇ�̗�O���� */
		if (train_x.size() < custom.batch_size) { exchandling::invalid_batch_size(__FILE__, __LINE__, "Trainer::train"); }

		bool break_flag = false;                      //�P�����I������t���O
		float train_acc = 0;                          //�P���f�[�^�̐��x
		float test_acc = 0;                           //�e�X�g�f�[�^�̐��x
		vec::vector2d batch_x(custom.batch_size);     //�P���f�[�^�̃o�b�`
		vec::vector2d batch_t(custom.batch_size);     //���x���̃o�b�`
		io::Txtout txtout(output_path + "_log.txt");  //�덷�Ɛ��x�̃e�L�X�g�o��

		/* �w�K���Ԃ̌v���J�n */
		const clock_t start = clock();

		/* �o�b�`�f�[�^�̐ݒ� */
		if (custom.batch_size == 0) {
			batch_x = train_x;
			batch_t = train_t;
		}

		/* �w�K�̊J�n */
		for (unsigned step = 1; step <= custom.learning_step; ++step)
		{
			//�o�b�`����
			if (custom.batch_size != 0)
			{
				if ((step - 1) % (train_x.size() / custom.batch_size) == 0) {
					vec::shuffle(train_x, train_t);
				}

				const size_t min_index = ((step - 1) % (train_x.size() / custom.batch_size)) * custom.batch_size;
				const size_t max_index = min_index + custom.batch_size - 1;
				size_t batch_index = 0;
				for (size_t index = min_index; index <= max_index; ++index) {
					batch_x[batch_index] = train_x[index];
					batch_t[batch_index] = train_t[index];
					batch_index++;
				}
			}

			//�`�d�Ɗw�K
			network.forward(batch_x, custom.dropout_ratio, true);
			network.backward(batch_t, custom.dropout_ratio);
			network.update();
			network.reset();

			//���x�̌v�Z
			if (custom.acc_span != 0 && (step - 1) % custom.acc_span == 0 && step != 1)
			{
				switch (custom.acc_type) {
				case 0:
					train_acc = accuracy_m(train_x, train_t);
					test_acc = accuracy_m(test_x, test_t);
					break;
				case 1:
					train_acc = accuracy_t(train_x, train_t);
					test_acc = accuracy_t(test_x, test_t);
					break;
				}
#if HAS_BOOST_HEADER
				//�����x�̎���xml�o��
				if (test_acc > custom.xmlout_inf) {
					io::xml_writer(
						network.get_layerset(),
						output_path + "_" + std::to_string(step) + "_" + std::to_string(int(test_acc * 10000)) + ".xml"
					);
				}
#endif
			}

			//�W���o��
			std::cout <<
				"step: " << step << "  " <<
				"loss: " << network.get_loss() << "  " << 
				"train-acc: " << train_acc << "  " << 
				"test-acc: " << test_acc << "  " << std::endl;

			//�e�L�X�g�o��
			txtout.write(
				step,
				network.get_loss(),
				train_acc,
				test_acc
			);

#if HAS_BOOST_HEADER
			//xml�o��
			if (custom.xml_span != 0 && (step - 1) % custom.xml_span == 0 && step != 1) {
				io::xml_writer(
					network.get_layerset(),
					output_path + "_" + std::to_string(step) + ".xml"
				);
			}
#endif

			//�L�[���͏���
			if (_kbhit())
			{
				switch (_getch()) {
				case 't': show_et(start, clock()); break;  //�o�ߎ��Ԃ̊m�F
				case 'f': break_flag = true; break;        //�P���̏I��
				case 's': (void)_getch(); break;           //�ꎞ��~
				default: break;
				}
			}

			//�P���̏I��
			if (break_flag) { break; }
		}


#if HAS_BOOST_HEADER
		/* �ŏI�p�����[�^�̏o�� */
		io::xml_writer(
			network.get_layerset(),
			output_path + "_final.xml"
		);
#endif

		/* �P���̐ݒ���o�� */
		io::Txtout outinfo(output_path + "_info.txt");
		outinfo.write("Optimizer:        ", typename_to_str<OptType>());
		outinfo.write("Hidden_Activation:", typename_to_str<ActType>());
		outinfo.write("Output_Activation:", typename_to_str<OutType>());
		outinfo.write("batch-size:   ", custom.batch_size);
		outinfo.write("dropout-ratio:", custom.dropout_ratio);

		/* �w�K���Ԃ��o�ߓ������o�� */
		show_et(start, clock());

	}







	//�o�͑w�ŁA�w�肵��臒l�𒴂���j���[�����ƃ��x����1���r���A���x���v�Z
	template <class OptType, class ActType, class OutType>
	float Trainer<OptType, ActType, OutType>::accuracy_t(
		const vec::vector2d& test_x,
		const vec::vector2d& test_t,
		const double threshold
	)
	{
		if (test_x.size() == 0) { return 0; }

		Network<OptType, ActType, OutType> train_acc(network.get_layerset());
		train_acc.forward(test_x, 0, false);

		const size_t row = train_acc.out.size();
		const size_t col = train_acc.out[0].size();
		size_t count_col, count_row = 0;
		for (size_t i = 0; i < row; ++i) {
			count_col = 0;
			for (size_t j = 0; j < col; ++j) {
				if (train_acc.out[i][j] > threshold && test_t[i][j] == 1) { count_col++; }
				else if (train_acc.out[i][j] <= threshold && test_t[i][j] == 0) { count_col++; }
			}
			if (count_col == col) { count_row++; }
		}

		return (float)count_row / (float)row;
	}








	//�o�͑w�ň�ԑ傫�Ȓl�����j���[�����ƃ��x��1���r���A���x���v�Z
	template <class OptType, class ActType, class OutType>
	float Trainer<OptType, ActType, OutType>::accuracy_m(
		const vec::vector2d& test_x,
		const vec::vector2d& test_t
	)
	{
		if (test_x.size() == 0) { return 0; }

		Network<OptType, ActType, OutType> train_acc(network.get_layerset());
		train_acc.forward(test_x, 0, false);

		size_t max_index = 0;
		std::vector<size_t> max_index_array(test_t.size());

		const size_t row = test_t.size();
		const size_t col = test_t[0].size();
		for (size_t i = 0; i < row; ++i) {
			for (size_t j = 0; j < col; ++j) {
				if (train_acc.out[i][max_index] < train_acc.out[i][j]) { max_index = j; }
			}
			max_index_array[i] = max_index;
		}

		int count = 0;
		for (size_t i = 0; i < row; ++i) {
			if (test_t[i][max_index_array[i]] == 1) { count++; }
		}

		return (float)count / (float)row;
	}







	




}


#endif //NN_TRAINER_H