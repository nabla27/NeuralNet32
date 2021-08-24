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





	//�n���ꂽ1����vector(double)�^���w�肵���s�E���2����vector(double)�^�ɂ���
	template <>
	vector2d reshape_to(const vector1d& mat, const size_t row, const size_t col)
	{
		//�w�肵���s���܂��͗񐔂��n���ꂽmat�ƍ���Ȃ�
		if (mat.size() != row * col) { exchandling::invalid_data_size(__FILE__, __LINE__, "reshape_to"); }
	
		vector2d reshaped_mat(row, vector1d(col));
		
		size_t index = 0;
		for (size_t i = 0; i < row; ++i) {
			for (size_t j = 0; j < col; ++i) {
				reshaped_mat[i][j] = mat[index];
				index++;
			}
		}

		return reshaped_mat;
	}




	//�n���ꂽ2����vector(double)��1����vector(double)�ɕϊ�
	template <>
	vector1d reshape_to(const vector2d& mat, const size_t row, const size_t col)
	{
		const size_t mat_row = mat.size();
		const size_t mat_col = mat[0].size();
		vector1d reshaped_mat(mat_row * mat_col);

		size_t index = 0;
		for (size_t i = 0; i < mat_row; ++i) {
			for (size_t j = 0; j < mat_col; ++j) {
				reshaped_mat[index] = mat[i][j];
			}
		}

		return reshaped_mat;
	}




	//�n���ꂽ2����vector(double)���w�肵���s�E���2����vector(double)�ɕϊ�
	template <>
	vector2d reshape_to(const vector2d& mat, const size_t row, const size_t col)
	{
		//�w�肵���s���܂��͗񐔂��n���ꂽmat�ƍ���Ȃ�
		if (mat[0].size() * mat.size() != row * col) { exchandling::invalid_data_size(__FILE__, __LINE__, "reshape_to"); }

		const size_t mat_row = mat.size();
		const size_t mat_col = mat[0].size();
		vector2d reshaped_mat(row, vector1d(col));
		
		size_t index_r = 0, index_c = 0;
		for (size_t i = 0; i < mat_row; ++i) {
			for (size_t j = 0; j < mat_col; ++j) 
			{
				reshaped_mat[index_r][index_c] = mat[i][j];

				if (index_c == col) { index_r++; index_c = 0; }
			}
		}

		return reshaped_mat;
	}













} //namespace vec



#endif //VEC_PREPROCESSING_H