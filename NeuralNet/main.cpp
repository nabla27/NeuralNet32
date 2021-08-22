#include "neuralnet32_nn.h"
#include "io/ioimg.h"
#if HAS_OPENCV_HEADER && HAS_CPLUS_17
#pragma comment(lib, "opencv_world452.lib")
#endif


int main()
{

#if HAS_OPENCV_HEADER && HAS_CPLUS_17 //MNISTデータ

	/* 画像の読み取り */
	io::ReadImg read_train_img;
	io::ReadImg read_test_img;
	read_train_img.to_vector("E:/MNIST/IMG/train_x", 1);
	read_test_img.to_vector("E:/MNIST/IMG/test_x", 1);
	vec::vector2d train_x = read_train_img.get_x();
	vec::vector2d train_t = read_train_img.get_t();
	vec::vector2d test_x = read_test_img.get_x();
	vec::vector2d test_t = read_test_img.get_t();
	{   //正規化
		using namespace vec;
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

#endif


	try 
	{

		/* ノード数の指定、重み・バイアスの初期化 */
		nn::LayerSet layerset(train_x[0].size(), train_t[0].size());
		layerset.set_node({ 100 });
		layerset.initialize();

		/* 学習の詳細設定 */
		nn::TrainCustom custom;
		custom.acc_span = 200;
		custom.batch_size = 200;
		custom.dropout_ratio = 0.5;
		custom.learning_step = 5;
		custom.xmlout_inf = 0.98f;
		custom.xml_span = 0;

		/* 学習の開始 */
		nn::Trainer <
			nn::OPTIMIZER::AdaBelief,
			nn::ACTIVATION::tanhExp,
			nn::ACTIVATION::Softmax
			> trainer(layerset);
		trainer.set_TrainData(train_x, train_t);
		trainer.set_TestData(test_x, test_t);
		trainer.set_OutputPath("E:/MNIST/data/test/mnist");
		trainer.train(custom);




	}
	catch (const std::runtime_error& error) {
		std::cout << error.what() << std::endl;
	}





	return 0;
}

