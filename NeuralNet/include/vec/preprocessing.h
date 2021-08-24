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
	vector2d padding(const vector2d& mat, const size_t wide = 1, const double val = 0)
	{
		const size_t row = mat.size();     //行数
		const size_t col = mat[0].size();  //列数
		vector2d padded(row + 2 * wide, vector1d(col + 2 * wide, val));  //パディング後の行列

		for (size_t i = 0; i < row; ++i) {
			for (size_t j = 0; j < col; ++j) {
				padded[i + wide][j + wide] = mat[i][j];
			}
		}

		return padded;
	}














} //namespace vec



#endif //VEC_PREPROCESSING_H