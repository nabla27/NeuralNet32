#define USE_OPENCV
#include "neuralnet32_nn.h"
#pragma comment(lib, "opencv_world452.lib")








int main()
{
#if 1
	/* 画像の読み取り */
	reading::Img1ch read_train_img;
	reading::Img1ch read_test_img;
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

	vec::vector2d A = {
		{1,1},
		{2,2},
		{3,3},
		{4,4}
	};
	vec::vector2d B =
	{
		{1,1},
		{2,2},
		{3,3},
		{4,4}
	};
	vec::shuffle(A, B);
	vec::show(A, "A");
	vec::show(B, "B");
	(void)std::getchar();


#elif 0
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

	/* ノード数の指定、重み・バイアスの初期化 */
	nn::LayerSet layerset(train_x[0].size(), train_t[0].size());
	layerset.set_node({ 100 });
	layerset.initialize();

#if 0

	/* 出力用フォルダーの作成 */
	std::string path = "E:/MNIST/data/08200651";
	std::filesystem::create_directory(path);

	/* 学習設定 */
	nn::TrainCustom custom;
	custom.acc_span = 200;
	custom.batch_size = 100;
	custom.dropout_ratio = 0.5;
	custom.file_path = path + "/mnist";
	custom.learning_step = 50000;
	custom.xmlout_inf = 0.98f;
	custom.xml_span = 0;

	/* データのセットと訓練の開始 */
	nn::Trainer<
		nn::OPTIMIZER::SGD,
		nn::ACTIVATION::tanhExp
	> 
		trainer(layerset);

	try {
		trainer.train_data(train_x, train_t);
		trainer.test_data(test_x, test_t);     //bug
		trainer.train(custom);
	}
	catch (const std::runtime_error& error) {
		std::cout << "runtime-error: " << error.what() << std::endl;
	}
	catch (const boost::wrapexcept<boost::property_tree::xml_parser::xml_parser_error>& error) {
		std::cout << "boost-error: " << error.what() << std::endl;
	}

#endif


	nn::Trainer
		<
		nn::OPTIMIZER::AdaBelief,
		nn::ACTIVATION::tanhExp,
		nn::ACTIVATION::Softmax
		> 
		trainer(layerset);

	trainer.train_data(train_x, train_t);
	trainer.test_data(test_x, test_t);

	std::string path = "E:/MNIST/data/08212338";
	std::filesystem::create_directory(path);

	nn::default_custom.acc_span = 200;
	nn::default_custom.batch_size = 100;
	nn::default_custom.dropout_ratio = 0.5;
	nn::default_custom.file_path = path + "/mnist";
	nn::default_custom.learning_step = 50000;
	nn::default_custom.xmlout_inf = 0.98;
	nn::default_custom.xml_span = 0;

	trainer.train(nn::default_custom);


	




	return 0;
}