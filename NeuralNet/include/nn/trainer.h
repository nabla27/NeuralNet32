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

/* コンパイラのバージョンを確認 */
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





	//訓練の設定
	struct TrainCustom 
	{
		unsigned learning_step = 10000;     //学習回数
		unsigned batch_size = 0;            //バッチサイズ
		double dropout_ratio = 0;           //ドロップアウトの割合
		unsigned acc_span = 100;            //精度を計算する頻度
		unsigned acc_type = 0;              //精度計算のタイプ

#ifdef HAS_BOOST_HEADER
		unsigned xml_span = 0;              //xmlファイルの
		float xmlout_inf = 0.97f;           //xmlファイルを出力する精度の下限
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











	//訓練データの設定
	template <class OptType, class ActType, class OutType>
	inline void Trainer<OptType, ActType, OutType>::set_TrainData(
		const vec::vector2d& train_x,
		const vec::vector2d& train_t
	)
	{
		//訓練データとそのラベルのサイズが合わない場合の例外処理
		if (train_x.size() != train_t.size()) { exchandling::mismatch_data_size(__FILE__, __LINE__, "Trainer::trian_data"); }
		//データが空である場合の例外処理
		if (train_x.size() == 0) { exchandling::empty_data(__FILE__, __LINE__, "Trainer::train_data"); }
		//訓練データのサイズが入力層のニューロン数と異なる時の例外処理
		if (train_x[0].size() != network.get_input_size()) { exchandling::invalid_data_size(__FILE__, __LINE__, "Trainer::train_data"); }
		//ラベルのサイズが出力層のニューロン数と異なる時の例外処理
		if (train_t[0].size() != network.get_output_size()) { exchandling::invalid_data_size(__FILE__, __LINE__, "Trainer::train_data"); }

		this->train_x = train_x;
		this->train_t = train_t;
	}






	//テスト用データの設定
	template <class OptType, class ActType, class OutType>
	inline void Trainer<OptType, ActType, OutType>::set_TestData(
		const vec::vector2d& test_x,
		const vec::vector2d& test_t
	)
	{
		//テストデータとそのラベルのサイズが合わない場合の例外処理
		if (test_x.size() != test_t.size()) { exchandling::mismatch_data_size(__FILE__, __LINE__, "Trainer::test_data"); }
		//データが空である場合の例外処理
		if (test_x.size() == 0) { exchandling::empty_data(__FILE__, __LINE__, "Trainer::test_data"); }
		//テストデータのサイズが入力層のニューロン数と異なる時の例外処理
		if (test_x[0].size() != network.get_input_size()) { exchandling::invalid_data_size(__FILE__, __LINE__, "Trainer::test_data"); }
		//ラベルのサイズが出力層のニューロン数と異なる時の例外処理
		if (test_t[0].size() != network.get_output_size()) { exchandling::invalid_data_size(__FILE__, __LINE__, "Trainer::test_data"); }

		this->test_x = test_x;
		this->test_t = test_t;
	}






	//出力するファイルのパスの設定
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
			//ディレクトリが存在しない場合の例外処理
			exchandling::not_exist_path(__FILE__, __LINE__, "Trainer::set_OutPath");
		}
		else { this->output_path = output_path; }
#endif
	}







	//訓練
	template <class OptType, class ActType, class OutType>
	void Trainer<OptType, ActType, OutType>::train(const TrainCustom custom)
	{
		/* データがセットされていなかった場合の例外処理 */
		if (train_x.size() == 0) { exchandling::empty_data(__FILE__, __LINE__, "Trainer::train"); }

		/* 指定したバッチサイズがデータサイズよりも大きかった場合の例外処理 */
		if (train_x.size() < custom.batch_size) { exchandling::invalid_batch_size(__FILE__, __LINE__, "Trainer::train"); }

		bool break_flag = false;                      //訓練を終了するフラグ
		float train_acc = 0;                          //訓練データの精度
		float test_acc = 0;                           //テストデータの精度
		vec::vector2d batch_x(custom.batch_size);     //訓練データのバッチ
		vec::vector2d batch_t(custom.batch_size);     //ラベルのバッチ
		io::Txtout txtout(output_path + "_log.txt");  //誤差と精度のテキスト出力

		/* 学習時間の計測開始 */
		const clock_t start = clock();

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

			//伝播と学習
			network.forward(batch_x, custom.dropout_ratio, true);
			network.backward(batch_t, custom.dropout_ratio);
			network.update();
			network.reset();

			//精度の計算
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
				//高精度の時のxml出力
				if (test_acc > custom.xmlout_inf) {
					io::xml_writer(
						network.get_layerset(),
						output_path + "_" + std::to_string(step) + "_" + std::to_string(int(test_acc * 10000)) + ".xml"
					);
				}
#endif
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
			if (custom.xml_span != 0 && (step - 1) % custom.xml_span == 0 && step != 1) {
				io::xml_writer(
					network.get_layerset(),
					output_path + "_" + std::to_string(step) + ".xml"
				);
			}
#endif

			//キー入力処理
			if (_kbhit())
			{
				switch (_getch()) {
				case 't': show_et(start, clock()); break;  //経過時間の確認
				case 'f': break_flag = true; break;        //訓練の終了
				case 's': (void)_getch(); break;           //一時停止
				default: break;
				}
			}

			//訓練の終了
			if (break_flag) { break; }
		}


#if HAS_BOOST_HEADER
		/* 最終パラメータの出力 */
		io::xml_writer(
			network.get_layerset(),
			output_path + "_final.xml"
		);
#endif

		/* 訓練の設定を出力 */
		io::Txtout outinfo(output_path + "_info.txt");
		outinfo.write("Optimizer:        ", typename_to_str<OptType>());
		outinfo.write("Hidden_Activation:", typename_to_str<ActType>());
		outinfo.write("Output_Activation:", typename_to_str<OutType>());
		outinfo.write("batch-size:   ", custom.batch_size);
		outinfo.write("dropout-ratio:", custom.dropout_ratio);

		/* 学習時間を経過日時を出力 */
		show_et(start, clock());

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








	//出力層で一番大きな値をもつニューロンとラベル1を比較し、精度を計算
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