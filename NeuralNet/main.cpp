/*  LICENSE
	Copyright (c) 2021, nabla All rights reserved.
	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/
#include "neuralnet32_nn.h"
#include "io/ioimg.h"
#if HAS_OPENCV_HEADER && HAS_CPLUS_17
#pragma comment(lib, "opencv_world452.lib")
#endif


int main()
{
	try{

#if HAS_OPENCV_HEADER && HAS_CPLUS_17 //MNIST�f�[�^

		/* �摜�̓ǂݎ�� */
		io::ReadImg read_train_img;
		io::ReadImg read_test_img;
		read_train_img.to_vector("E:/MNIST/IMG/train_x", 1);
		read_test_img.to_vector("E:/MNIST/IMG/test_x", 1);
		vec::vector2d train_x = read_train_img.get_x();     //training data
		vec::vector2d train_t = read_train_img.get_t();     //label of training data
		vec::vector2d test_x = read_test_img.get_x();       //testing data
		vec::vector2d test_t = read_test_img.get_t();       //label of testing data

		//�O����
		{ using namespace vec;

			for (size_t i = 0; i < train_x.size(); ++i) 
			{
				vector2d _train_x = reshape_to<vector2d>(train_x[i], read_train_img.get_row(), read_train_img.get_col());
				_train_x = padding(_train_x, 2, 0);
				_train_x = pooling_average(_train_x, 5, 1);
				train_x[i] = reshape_to<vector1d>(_train_x);
				if (i < test_x.size()) 
				{
					vector2d _test_x = reshape_to<vector2d>(test_x[i], read_test_img.get_row(), read_test_img.get_col());
					_test_x = padding(_test_x, 2, 0);
					_test_x = pooling_average(_test_x, 5, 1);
					test_x[i] = reshape_to<vector1d>(_test_x);
				}
			}

			train_x = train_x / 255;
			test_x = test_x / 255;
		}

#else //XOR�Q�[�g
		vec::vector2d train_x =
		{
			{0,0},
			{0,1},
			{1,0},
			{1,1},
		};
		vec::vector2d train_t =
		{
			{1,0},
			{0,1},
			{0,1},
			{1,0},
		};
		vec::vector2d test_x = train_x;
		vec::vector2d test_t = train_t;

#endif //HAS_OPENCV_HEADER && HAS_CPLUS_17












		/* �m�[�h���̎w��A�d�݁E�o�C�A�X�̏����� */
		nn::LayerSet layerset(train_x[0].size(), train_t[0].size());
		layerset.set_node({ 1024, 512, 256 });
		layerset.initialize(nn::InitType::He);



		/* �w�K�̏ڍאݒ� */
		nn::TrainCustom custom;
		custom.acc_span = 200;         //Outputting accuracy span.
		custom.batch_size = 100;       //Batch size. if you set 0, it do not batch learning.
		custom.dropout_ratio = 0.5;    //Dropout ratio. if you set0, it do not dropout.
		custom.learning_step = 70000;  //The number of leaning step (not epoch).
		custom.xmlout_inf = 0.98f;     //Lower limit of accuracy to output parameters such as weights, bias as xml files.
		custom.xml_span = 0;           //The span of outputting xml file to save parameters.



		/* �w�K */
		nn::Trainer <
			nn::OPTIMIZER::AdaBelief,   //Optimizer
			nn::ACTIVATION::tanhExp,    //Hidden Layer Activation
			nn::ACTIVATION::Softmax     //Output Layer Activation
			> trainer(layerset);
		trainer.set_TrainData(train_x, train_t);
		trainer.set_TestData(test_x, test_t);
		trainer.set_OutputPath("E:/MNIST/data/08250028/mnist");
		trainer.train(custom);




	}
	catch (const std::runtime_error& error) {
		std::cout << error.what() << std::endl;
	}


	//Trainer��TxtOut�I�u�W�F�N�g��XmlOut�I�u�W�F�N�g����������
	//ReadImg�N���X��template��


	return 0;
}

