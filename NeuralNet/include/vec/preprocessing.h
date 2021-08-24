/*  LICENSE
	Copyright (c) 2021, nabla All rights reserved.
	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/
#ifndef VEC_PREPROCESSING_H
#define VEC_PROPROCESSING_H

#include "function.h"
#include "operator.h"
#include "util/exchanding.h"




namespace vec {


	template <class T1, class T2>
	T1 reshape_to(const T2& mat, const size_t row = 0, const size_t col = 0);





	//渡された1次元vector(double)型を指定した行・列で2次元vector(double)型にする
	template <>
	vector2d reshape_to(const vector1d& mat, const size_t row, const size_t col)
	{
		//指定した行数または列数が渡されたmatと合わない
		if (mat.size() != row * col) { exchandling::invalid_data_size(__FILE__, __LINE__, "reshape_to"); }
	
		vector2d reshaped(row, vector1d(col));
		
		size_t index = 0;
		for (size_t i = 0; i < row; ++i) {
			for (size_t j = 0; j < col; ++i) {
				reshaped[i][j] = mat[index];
				index++;
			}
		}

		return reshaped;
	}




	//渡された2次元vector(double)を1次元vector(double)に変換
	template <>
	vector1d reshape_to(const vector2d& mat, const size_t row, const size_t col)
	{
		const size_t mat_row = mat.size();
		const size_t mat_col = mat[0].size();
		vector1d reshaped(mat_row * mat_col);

		size_t index = 0;
		for (size_t i = 0; i < mat_row; ++i) {
			for (size_t j = 0; j < mat_col; ++j) {
				reshaped[index] = mat[i][j];
			}
		}

		return reshaped;
	}




	//渡された2次元vector(double)を指定した行・列の2次元vector(double)に変換
	template <>
	vector2d reshape_to(const vector2d& mat, const size_t row, const size_t col)
	{
		//指定した行数または列数が渡されたmatと合わない
		if (mat[0].size() * mat.size() != row * col) { exchandling::invalid_data_size(__FILE__, __LINE__, "reshape_to"); }

		const size_t mat_row = mat.size();
		const size_t mat_col = mat[0].size();
		vector2d reshaped(row, vector1d(col));
		
		size_t index_r = 0, index_c = 0;
		for (size_t i = 0; i < mat_row; ++i) {
			for (size_t j = 0; j < mat_col; ++j) 
			{
				reshaped[index_r][index_c] = mat[i][j];

				if (index_c == col) { index_r++; index_c = 0; }
			}
		}

		return reshaped;
	}

















	//データの周囲に固定のデータを埋める(パディング)
	vector2d padding(
		const vector2d& mat, 
		const size_t width = 1,   //パディングする幅
		const double val = 0      //paddingで拡張された部分の値
	)
	{
		const size_t row = mat.size();     //行数
		const size_t col = mat[0].size();  //列数
		vector2d padded(row + 2 * width, vector1d(col + 2 * width, val));  //パディング後の行列

		for (size_t i = 0; i < row; ++i) {
			for (size_t j = 0; j < col; ++j) {
				padded[i + width][j + width] = mat[i][j];
			}
		}

		return padded;
	}





	//Max Pooling
	vector2d pooling_max(
		const vector2d& mat,   //poolingするvector(double)型の行列
		const size_t width,    //ウィンドウの上下幅
		const size_t stride    //ウィンドウを移動させる歩幅
	)
	{
		const size_t Mr = mat.size();      //引数matの行数
		const size_t Mc = mat[0].size();   //引数matの列数
		const size_t Pr = (Mr - width) / stride + 1;   //pooling後の行数
		const size_t Pc = (Mc - width) / stride + 1;   //Pooling後の列数
		vector2d pooling(Pr, vector1d(Pc));

		//i,jはpoolingの添え字に対応
		for (size_t i = 0; i < Pr; ++i) {
			for (size_t j = 0; j < Pc; ++j) 
			{
				//ウィンドウ内(Mi,Mj)の最大値を捜索
				size_t max_index_row = i * stride, max_index_col = j * stride;   //ウィンドウ内の最大値のインデックスを記憶しておく変数
				for (size_t Mi = i * stride; Mi < i * stride + width; Mi++) {
					for (size_t Mj = j * stride; Mj < j * stride + width; Mj++) {
						if (mat[Mi][Mj] > mat[max_index_row][max_index_col]) { max_index_row = Mi; max_index_col = Mj; }
					}
				}

				//最大値を代入
				pooling[i][j] = mat[max_index_row][max_index_col];
			}
		}

		return pooling;
	}





	//Average Pooling
	vector2d pooling_average(
		const vector2d& mat,
		const size_t width,   //ウィンドウの上下幅
		const size_t stride   //ウィンドウを移動させる歩幅
	)
	{
		const size_t Mr = mat.size();      //引数matの行数
		const size_t Mc = mat[0].size();   //引数matの列数
		const size_t Pr = (Mr - width) / stride + 1;   //pooling後の行数
		const size_t Pc = (Mc - width) / stride + 1;   //Pooling後の列数
		vector2d pooling(Pr, vector1d(Pc));

		//i,jはpoolingの添え字に対応
		for (size_t i = 0; i < Pr; ++i) {
			for (size_t j = 0; j < Pc; ++j)
			{
				//ウィンドウ内(Mi,Mj)の値を足し合わせる
				double sum = 0;   //ウィンドウ内の値を集計
				for (size_t Mi = i * stride; Mi < i * stride + width; Mi++) {
					for (size_t Mj = j * stride; Mj < j * stride + width; Mj++) {
						sum += mat[Mi][Mj];
					}
				}

				//平均値を算出
				pooling[i][j] = sum / (static_cast<double>(width) * static_cast<double>(width));
			}
		}

		return pooling;
	}














} //namespace vec



#endif //VEC_PREPROCESSING_H