#ifndef NN_TRAINER_H
#define NN_TRAINER_H

#include <time.h>
#include <conio.h>
#include "network.h"
#include "optimizer.h"
#include "filing/iotxt.h"
#include "filing/ioxml.h"





namespace nn {



	//訓練の設定
	struct TrainCustom 
	{
		unsigned learning_step = 10000;     //学習回数
		unsigned batch_size = 0;            //バッチサイズ
		std::string file_path = "./train";  //ファイルを出力するパスと基本名
		double dropout_ratio = 0;           //ドロップアウトの割合
		unsigned acc_span = 100;            //精度を計算する頻度
		unsigned acc_type = 0;              //精度計算のタイプ

#ifdef HAS_BOOST_HEADER
		unsigned xml_span = 0;              //xmlファイルの
		float xmlout_inf = 0.97f;           //xmlファイルを出力する精度の下限
#endif
	} default_custom;







	//指定した時間の差分(ミリ秒)を経過日時に変換し、標準出力する
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











	//訓練データの設定
	template <class OptType, class ActType, class OutType>
	void Trainer<OptType, ActType, OutType>::train_data(
		const vec::vector2d& train_x,
		const vec::vector2d& train_t
	)
	{
		//訓練データとラベルのサイズが合わない場合の例外処理
		if (train_x.size() != train_t.size()) {
			std::cout << "ERROR : trainer.h(" << __LINE__ << ") <Trainer::train_data>" << std::endl;
			throw std::runtime_error("The size of data and label for training must be the same.");
		}
		//データが空である場合の例外処理
		if (train_x.size() == 0) {
			std::cout << "ERROR : trainer.h(" << __LINE__ << ") <Trainer::train_data>" << std::endl;
			throw std::runtime_error("The data for training is empty.");
		}
		this->train_x = train_x;
		this->train_t = train_t;
	}






	//テスト用データの設定
	template <class OptType, class ActType, class OutType>
	void Trainer<OptType, ActType, OutType>::test_data(
		const vec::vector2d& test_x,
		const vec::vector2d& test_t
	)
	{
		//訓練データとラベルのサイズが合わない場合の例外処理
		if (test_x.size() != test_t.size()) {
			std::cout << "ERROR : trainer.h(" << __LINE__ << ") <Trainer::test_data>" << std::endl;
			throw std::runtime_error("The size of data and label for testing must be the same.");
		}
		//データが空である場合の例外処理
		if (test_x.size() == 0) {
			std::cout << "ERROR : trainer.h(" << __LINE__ << ") <Trainer::test_data>" << std::endl;
			throw std::runtime_error("The data for testing is empty.");
		}
		this->test_x = test_x;
		this->test_t = test_t;
	}






	//訓練
	template <class OptType, class ActType, class OutType>
	void Trainer<OptType, ActType, OutType>::train(const TrainCustom custom)
	{
		/* データがセットされていなかった場合の例外処理 */
		if (train_x.size() == 0) {
			std::cout << "ERROR : trainer.h(" << __LINE__ << ") <Trainer::train>" << std::endl;
			throw std::runtime_error("The data for trainig is empty.");
		}

		/* 指定したバッチサイズがデータサイズよりも大きかった場合の例外処理 */
		if (train_x.size() < custom.batch_size) {
			std::cout << "ERROR : trainer.h(" << __LINE__ << ") <Trainer::train>" << std::endl;
			throw std::runtime_error("The batch size must be less than data size.");
		}

		bool train_breaker = false;
		float train_acc = 0;  //訓練データの精度
		float test_acc = 0;   //テストデータの精度
		vec::vector2d batch_x(custom.batch_size);  //訓練データのバッチ
		vec::vector2d batch_t(custom.batch_size);  //ラベルのバッチ

		filing::IOtxt txtout(custom.file_path + "_log.txt");  //誤差と精度のテキスト出力
#if HAS_BOOST_HEADER
		filing::IOxml xmlout;                                 //重みやバイアスのxml出力
#endif

		/* 学習時間の計測開始 */
		clock_t start = clock();

		/* バッチデータの設定 */
		if (custom.batch_size == 0) {
			batch_x = train_x;
			batch_t = train_t;
		}

		/* 学習の開始 */
		for (unsigned step = 1; step <= custom.learning_step; ++step)
		{
			//バッチ処理
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

			//伝播と学習
			network.forward(batch_x, custom.dropout_ratio, true);
			network.backward(batch_t, custom.dropout_ratio);
			network.update();
			network.reset();

			//精度の計算
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

			//標準出力
			std::cout <<
				"step: " << step << "  " <<
				"loss: " << network.get_loss() << "  " << 
				"train-acc: " << train_acc << "  " << 
				"test-acc: " << test_acc << "  " << std::endl;

			//テキスト出力
			txtout.write(
				step,
				network.get_loss(),
				train_acc,
				test_acc
			);

#if HAS_BOOST_HEADER
			//xml出力
			if (custom.xml_span != 0 && step % custom.xml_span == 0) {
				xmlout.xml_writer(
					network.get_layerset(),
					custom.file_path + "_" + std::to_string(step) + ".xml"
				);
			}
			//高精度の時のxml出力
			if (test_acc > custom.xmlout_inf) {
				xmlout.xml_writer(
					network.get_layerset(),
					custom.file_path + "_" + std::to_string(step) + "_" + std::to_string(int(test_acc * 1000)) + ".xml"
				);
			}
#endif

			//キー入力処理
			if (_kbhit())
			{
				switch (_getch()) {
				case 't': show_time(start, clock()); break;  //経過時間の確認
				case 'f': train_breaker = true; break;       //訓練の終了
				case 's': (void)getchar(); break;            //一時停止
				default: break;
				}
			}

			//訓練の終了
			if (train_breaker) { break; }
		}


#if HAS_BOOST_HEADER
		/* 最終パラメータの出力 */
		xmlout.xml_writer(
			network.get_layerset(),
			custom.file_path + "_final.xml"
		);
#endif

		/* 学習時間を経過日時を出力 */
		show_time(start, clock());

	}







	//出力層で、指定した閾値を超えるニューロンとラベルの1を比較し、精度を計算
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








	//出力層で一番大きな値をもつニューロンとラベル1を比較し、精度を計算
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