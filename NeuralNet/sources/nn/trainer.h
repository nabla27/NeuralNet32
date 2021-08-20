#ifndef NN_TRAINER_H
#define NN_TRAINER_H

#include <time.h>
#include <conio.h>
#include "network.h"
#include "optimizer.h"
#include "filing/iotxt.h"
#include "filing/ioxml.h"





namespace nn {



	//�P���̐ݒ�
	struct TrainCustom 
	{
		unsigned learning_step = 10000;     //�w�K��
		unsigned batch_size = 0;            //�o�b�`�T�C�Y
		std::string file_path = "./train";  //�t�@�C�����o�͂���p�X�Ɗ�{��
		double dropout_ratio = 0;           //�h���b�v�A�E�g�̊���
		unsigned acc_span = 100;            //���x���v�Z����p�x
		unsigned acc_type = 0;              //���x�v�Z�̃^�C�v

#ifdef HAS_BOOST_HEADER
		unsigned xml_span = 0;              //xml�t�@�C����
		float xmlout_inf = 0.97f;           //xml�t�@�C�����o�͂��鐸�x�̉���
#endif
	} default_custom;







	//�w�肵�����Ԃ̍���(�~���b)���o�ߓ����ɕϊ����A�W���o�͂���
	void show_time(clock_t start, clock_t end)
	{
		clock_t diff = (end - start) / CLOCKS_PER_SEC;
		clock_t minutes = diff / 60;
		clock_t hours = minutes / 60;
		clock_t days = hours / 24;
		std::cout << "time: "
			<< days << "days "
			<< hours - days * 24 << "hours "
			<< minutes - hours * 60 << "minutes "
			<< diff - minutes * 60 << "seconds "
			<< std::endl;
		(void)std::getchar();
	}






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
	public:
		Network<OptType, ActType, OutType> network;
	public:
		Trainer(const LayerSet& layerset) :
			network(layerset) {}

		void train_data(
			const vec::vector2d& train_x,
			const vec::vector2d& train_t
		);

		void test_data(
			const vec::vector2d& test_x,
			const vec::vector2d& test_t
		);

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
	void Trainer<OptType, ActType, OutType>::train_data(
		const vec::vector2d& train_x,
		const vec::vector2d& train_t
	)
	{
		//�P���f�[�^�ƃ��x���̃T�C�Y������Ȃ��ꍇ�̗�O����
		if (train_x.size() != train_t.size()) {
			std::cout << "ERROR : trainer.h(" << __LINE__ << ") <Trainer::train_data>" << std::endl;
			throw std::runtime_error("The size of data and label for training must be the same.");
		}
		//�f�[�^����ł���ꍇ�̗�O����
		if (train_x.size() == 0) {
			std::cout << "ERROR : trainer.h(" << __LINE__ << ") <Trainer::train_data>" << std::endl;
			throw std::runtime_error("The data for training is empty.");
		}
		this->train_x = train_x;
		this->train_t = train_t;
	}






	//�e�X�g�p�f�[�^�̐ݒ�
	template <class OptType, class ActType, class OutType>
	void Trainer<OptType, ActType, OutType>::test_data(
		const vec::vector2d& test_x,
		const vec::vector2d& test_t
	)
	{
		//�P���f�[�^�ƃ��x���̃T�C�Y������Ȃ��ꍇ�̗�O����
		if (test_x.size() != test_t.size()) {
			std::cout << "ERROR : trainer.h(" << __LINE__ << ") <Trainer::test_data>" << std::endl;
			throw std::runtime_error("The size of data and label for testing must be the same.");
		}
		//�f�[�^����ł���ꍇ�̗�O����
		if (test_x.size() == 0) {
			std::cout << "ERROR : trainer.h(" << __LINE__ << ") <Trainer::test_data>" << std::endl;
			throw std::runtime_error("The data for testing is empty.");
		}
		this->test_x = test_x;
		this->test_t = test_t;
	}






	//�P��
	template <class OptType, class ActType, class OutType>
	void Trainer<OptType, ActType, OutType>::train(const TrainCustom custom)
	{
		/* �f�[�^���Z�b�g����Ă��Ȃ������ꍇ�̗�O���� */
		if (train_x.size() == 0) {
			std::cout << "ERROR : trainer.h(" << __LINE__ << ") <Trainer::train>" << std::endl;
			throw std::runtime_error("The data for trainig is empty.");
		}

		/* �w�肵���o�b�`�T�C�Y���f�[�^�T�C�Y�����傫�������ꍇ�̗�O���� */
		if (train_x.size() < custom.batch_size) {
			std::cout << "ERROR : trainer.h(" << __LINE__ << ") <Trainer::train>" << std::endl;
			throw std::runtime_error("The batch size must be less than data size.");
		}

		bool train_breaker = false;
		float train_acc = 0;  //�P���f�[�^�̐��x
		float test_acc = 0;   //�e�X�g�f�[�^�̐��x
		vec::vector2d batch_x(custom.batch_size);  //�P���f�[�^�̃o�b�`
		vec::vector2d batch_t(custom.batch_size);  //���x���̃o�b�`

		filing::IOtxt txtout(custom.file_path + "_log.txt");  //�덷�Ɛ��x�̃e�L�X�g�o��
#if HAS_BOOST_HEADER
		filing::IOxml xmlout;                                 //�d�݂�o�C�A�X��xml�o��
#endif

		/* �w�K���Ԃ̌v���J�n */
		clock_t start = clock();

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
				if (step % (train_x.size() / custom.batch_size) == 0) {
					vec::shuffle(train_x, train_t);
				}

				size_t min_index = ((step - 1) % (train_x.size() / custom.batch_size)) * custom.batch_size;
				size_t max_index = min_index + custom.batch_size - 1;
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
			if (custom.acc_span != 0 && step % custom.acc_span == 0)
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
			if (custom.xml_span != 0 && step % custom.xml_span == 0) {
				xmlout.xml_writer(
					network.get_layerset(),
					custom.file_path + "_" + std::to_string(step) + ".xml"
				);
			}
			//�����x�̎���xml�o��
			if (test_acc > custom.xmlout_inf) {
				xmlout.xml_writer(
					network.get_layerset(),
					custom.file_path + "_" + std::to_string(step) + "_" + std::to_string(int(test_acc * 1000)) + ".xml"
				);
			}
#endif

			//�L�[���͏���
			if (_kbhit())
			{
				switch (_getch()) {
				case 't': show_time(start, clock()); break;  //�o�ߎ��Ԃ̊m�F
				case 'f': train_breaker = true; break;       //�P���̏I��
				case 's': (void)getchar(); break;            //�ꎞ��~
				default: break;
				}
			}

			//�P���̏I��
			if (train_breaker) { break; }
		}


#if HAS_BOOST_HEADER
		/* �ŏI�p�����[�^�̏o�� */
		xmlout.xml_writer(
			network.get_layerset(),
			custom.file_path + "_final.xml"
		);
#endif

		/* �w�K���Ԃ��o�ߓ������o�� */
		show_time(start, clock());

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

		Network train_acc(network.get_layerset());
		train_acc.forward(test_x, 0, false);

		size_t row = train_acc.out.size();
		size_t col = train_acc.out[0].size();
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

		Network train_acc(network.get_layerset());
		train_acc.forward(test_x, 0, false);

		size_t max_index = 0;
		std::vector<size_t> max_index_array(test_t.size());

		size_t row = test_t.size();
		size_t col = test_t[0].size();
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