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




	//�n���ꂽ2����vector(double)��1����vector(double)�ɕϊ�
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




	//�n���ꂽ2����vector(double)���w�肵���s�E���2����vector(double)�ɕϊ�
	template <>
	vector2d reshape_to(const vector2d& mat, const size_t row, const size_t col)
	{
		//�w�肵���s���܂��͗񐔂��n���ꂽmat�ƍ���Ȃ�
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

















	//�f�[�^�̎��͂ɌŒ�̃f�[�^�𖄂߂�(�p�f�B���O)
	vector2d padding(
		const vector2d& mat, 
		const size_t width = 1,   //�p�f�B���O���镝
		const double val = 0      //padding�Ŋg�����ꂽ�����̒l
	)
	{
		const size_t row = mat.size();     //�s��
		const size_t col = mat[0].size();  //��
		vector2d padded(row + 2 * width, vector1d(col + 2 * width, val));  //�p�f�B���O��̍s��

		for (size_t i = 0; i < row; ++i) {
			for (size_t j = 0; j < col; ++j) {
				padded[i + width][j + width] = mat[i][j];
			}
		}

		return padded;
	}





	//Max Pooling
	vector2d pooling_max(
		const vector2d& mat,   //pooling����vector(double)�^�̍s��
		const size_t width,    //�E�B���h�E�̏㉺��
		const size_t stride    //�E�B���h�E���ړ����������
	)
	{
		const size_t Mr = mat.size();      //����mat�̍s��
		const size_t Mc = mat[0].size();   //����mat�̗�
		const size_t Pr = (Mr - width) / stride + 1;   //pooling��̍s��
		const size_t Pc = (Mc - width) / stride + 1;   //Pooling��̗�
		vector2d pooling(Pr, vector1d(Pc));

		//i,j��pooling�̓Y�����ɑΉ�
		for (size_t i = 0; i < Pr; ++i) {
			for (size_t j = 0; j < Pc; ++j) 
			{
				//�E�B���h�E��(Mi,Mj)�̍ő�l��{��
				size_t max_index_row = i * stride, max_index_col = j * stride;   //�E�B���h�E���̍ő�l�̃C���f�b�N�X���L�����Ă����ϐ�
				for (size_t Mi = i * stride; Mi < i * stride + width; Mi++) {
					for (size_t Mj = j * stride; Mj < j * stride + width; Mj++) {
						if (mat[Mi][Mj] > mat[max_index_row][max_index_col]) { max_index_row = Mi; max_index_col = Mj; }
					}
				}

				//�ő�l����
				pooling[i][j] = mat[max_index_row][max_index_col];
			}
		}

		return pooling;
	}





	//Average Pooling
	vector2d pooling_average(
		const vector2d& mat,
		const size_t width,   //�E�B���h�E�̏㉺��
		const size_t stride   //�E�B���h�E���ړ����������
	)
	{
		const size_t Mr = mat.size();      //����mat�̍s��
		const size_t Mc = mat[0].size();   //����mat�̗�
		const size_t Pr = (Mr - width) / stride + 1;   //pooling��̍s��
		const size_t Pc = (Mc - width) / stride + 1;   //Pooling��̗�
		vector2d pooling(Pr, vector1d(Pc));

		//i,j��pooling�̓Y�����ɑΉ�
		for (size_t i = 0; i < Pr; ++i) {
			for (size_t j = 0; j < Pc; ++j)
			{
				//�E�B���h�E��(Mi,Mj)�̒l�𑫂����킹��
				double sum = 0;   //�E�B���h�E���̒l���W�v
				for (size_t Mi = i * stride; Mi < i * stride + width; Mi++) {
					for (size_t Mj = j * stride; Mj < j * stride + width; Mj++) {
						sum += mat[Mi][Mj];
					}
				}

				//���ϒl���Z�o
				pooling[i][j] = sum / (static_cast<double>(width) * static_cast<double>(width));
			}
		}

		return pooling;
	}














} //namespace vec



#endif //VEC_PREPROCESSING_H