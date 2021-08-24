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

#if HAS_OPENCV_HEADER && HAS_CPLUS_17 //MNISTデータ



		/* 画像の読み取り */
		io::ReadImg read_train_img;
		io::ReadImg read_test_img;
		read_train_img.to_vector("E:/MNIST/IMG/train_x", 1);
		read_test_img.to_vector("E:/MNIST/IMG/test_x", 1);
		vec::vector2d train_x = read_train_img.get_x();     //training data
		vec::vector2d train_t = read_train_img.get_t();     //label of training data
		vec::vector2d test_x = read_test_img.get_x();       //testing data
		vec::vector2d test_t = read_test_img.get_t();       //label of testing data

		//前処理
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

#else //XORゲート
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












		/* ノード数の指定、重み・バイアスの初期化 */
		nn::LayerSet layerset(train_x[0].size(), train_t[0].size());
		layerset.set_node({ 100 });
		layerset.initialize(nn::InitType::He);



		/* 学習の詳細設定 */
		nn::TrainCustom custom;
		custom.acc_span = 200;
		custom.batch_size = 100;
		custom.dropout_ratio = 0.1;
		custom.learning_step = 50000;
		custom.xmlout_inf = 0.98f;
		custom.xml_span = 0;



		/* 学習 */
		nn::Trainer <
			nn::OPTIMIZER::AdaBelief,
			nn::ACTIVATION::tanhExp,
			nn::ACTIVATION::Softmax
			> trainer(layerset);
		trainer.set_TrainData(train_x, train_t);
		trainer.set_TestData(test_x, test_t);
		trainer.set_OutputPath("E:/MNIST/data/08242316/mnist");
		trainer.train(custom);




	}
	catch (const std::runtime_error& error) {
		std::cout << error.what() << std::endl;
	}



	return 0;
}

