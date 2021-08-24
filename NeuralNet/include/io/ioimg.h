/*  This source code use opencv library.
* 
	Copyright 2021 nabla

	Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

	1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

	2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

	3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
*/

/*  LICENSE
	Copyright (c) 2021, nabla All rights reserved.
	Use of this source code is governed by a MIT license that can be found
	in the LICENSE file.
*/

#ifndef IO_IOIMG_H
#define IO_IOIMG_H



/* opencv���C�u�������p�X�ɑ��݂��邩 */
#ifndef HAS_OPENCV_HEADER
#define HAS_OPENCV_HEADER 0
#endif



/* c++17�ȏ�̃o�[�W�����ł��邩 */
#ifndef HAS_CPLUS_17
#define HAS_CPLUS_17 0
#endif



/* opencv�̕K�v�ȃ��C�u���������݂��邩 */
#ifdef __has_include
#if __has_include(<opencv2/opencv.hpp>)

#undef HAS_OPENCV_HEADER
#define HAS_OPENCV_HEADER 1

#endif
#endif



/* �R���p�C���̃o�[�W�������m�F */
#ifdef _MSVC_LANG
#if (_MSVC_LANG >= 201703L)	//c++17 or later

#undef HAS_CPLUS_17
#define HAS_CPLUS_17 1

#endif
#endif

#ifdef __cplusplus
#if (__cplusplus >= 201703L)	//c++17 or later

#undef HAS_CPLUS_17
#define HAS_CPLUS_17 1

#endif
#endif






#if HAS_CPLUS_17
#if HAS_OPENCV_HEADER

#include <vector>
#include <filesystem>	//c++17 or later
#include <typeinfo>
#include <opencv2/opencv.hpp>
#include "util/exchanding.h"
#include "vec/function.h"





namespace io {






	class ReadImg {
	private:
		vec::vector2d img_x;
		vec::vector2d img_t;
		size_t row = 0;
		size_t col = 0;
	public:
		void to_vector(
			const std::string path,
			const double scale = 0.5,
			const bool disp = false
		);

		inline vec::vector2d get_x() const { return img_x; }
		inline vec::vector2d get_t() const { return img_t; }
		inline size_t get_row() const { return row; }
		inline size_t get_col() const { return col; }
	};













	/**************************************************
		path ----- folder1 ----- img1
			  |             |--- img2
			  |             |--- img3
			  |
			  |--- folder2 ----- img4
			  |             |--- img5
			  |             |--- img6
			  |
			  |--- folder3 ----- img7
			  |             |--- img8
			  |             |--- img9
			  |
			  |---
	****************************************************
	* �w�肳�ꂽ�p�X���Ƀt�H���_�[���Ƃɉ摜�����ނ��ꂽ
	* ���t�f�[�^����vector�z��Ƃ��āA�P���f�[�^�Ɛ�����
	* �x���𐶐�����B
	* **************************************************/

	void ReadImg::to_vector(
		const std::string path,
		const double scale,
		const bool disp
	)
	{
		namespace fs = std::filesystem;

		std::cout << "---------- loading images ----------" << std::endl;

		/* �w�肵���p�X���f�B���N�g���ł��邩�A���̃f�B���N�g�����Ƀt�H���_�[���������邩 */
		size_t num_type = 0;    //�摜���ސ�(�w��p�X���̃f�B���N�g����)
		if (fs::is_directory(path)) {
			for (const auto& p : fs::recursive_directory_iterator(path)) {
				if (fs::is_directory(p.path())) { num_type++; }
			}
		}
		else {
			std::cout << "cannot open : " << path << std::endl;
			return;
		}

		cv::Mat img, dst;
		std::vector<double> img_vec;    //1�摜�����i�[����
		int index = 0;                  //���ڂ̉摜��
		int type = 0;                   //�摜�̕��ޔԍ�
		int width, height;              //�摜�̃T�C�Y

		/* �摜��vector�z��ɕϊ� */
		for (const auto& dir_name : fs::recursive_directory_iterator(path))
		{
			if (fs::is_directory(dir_name.path()))
			{
				std::cout << "open folder : " << dir_name.path().string() << std::endl;
				std::string folder = dir_name.path().string();    //�摜�̓������t�H���_�[��

				for (const auto& file : fs::recursive_directory_iterator(folder))
				{
					img = cv::imread(file.path().string());       //�摜�f�[�^

					if (img.empty()) {
						std::cout << "cannot load : " << file.path().string() << std::endl;
					}
					else
					{
						if (disp) { std::cout << "loading ... " << file.path().string() << std::endl; }

						cv::cvtColor(img, dst, cv::COLOR_RGB2GRAY);    //�摜img���O���C�X�P�[��(����)�ɕϊ�

						if (index == 0) {    //�ꖇ�ڂ̉摜�����ɃX�P�[���ϊ���̃T�C�Y�����߂�

							width = (int)(dst.cols * scale);
							height = (int)(dst.rows * scale);
						}
						if (dst.cols != width || dst.rows != height) {    //�摜�̃T�C�Y�𑵂���
							cv::resize(dst, dst, cv::Size(width, height));
						}

						//���摜�̃T�C�Y���L��
						this->row = dst.rows;
						this->col = dst.cols;

						dst = dst.reshape(0, 1);    //1�����z��ɕϊ�
						dst.copyTo(img_vec);        //cv::mat�^����vector�^�ɕϊ�
						img_x.push_back(img_vec);
						img_t.push_back(vec::vector1d(num_type, 0));
						img_t[index][type] = 1;
						index++;
					}
				}
				type++;
			}
		}
		std::cout << "---------- done ----------" << std::endl;
	}










	/* 1�����z����w�肵���T�C�Y�ŉ摜�ɕϊ����A�\������ */
	template <class T = unsigned char>
	cv::Mat to_img(
		const vec::vector1d& mat,
		const int row,
		const int col
	)
	{
		//�����Ŏw�肳�ꂽ�T�C�Y��vector�z��̃T�C�Y����v���Ȃ���O����
		if (mat.size() != (size_t)row * (size_t)col) { exchandling::mismatch_data_size(__FILE__, __LINE__, "to_img"); }

		const int dimensions = 2;                   //������
		const int sizes[dimensions] = { row, col }; //�摜�T�C�Y

		//Mat�̃^�C�v�ϊ�
		int image_type = 0;
		if (typeid(T) == typeid(unsigned char)) { image_type = 0; }
		else if (typeid(T) == typeid(signed char)) { image_type = 1; }
		else if (typeid(T) == typeid(unsigned short)) { image_type = 2; }
		else if (typeid(T) == typeid(short)) { image_type = 3; }
		else if (typeid(T) == typeid(int)) { image_type = 4; }
		else if (typeid(T) == typeid(float)) { image_type = 5; }
		else if (typeid(T) == typeid(double)) { image_type = 6; }

		cv::Mat image(dimensions, sizes, image_type);

		size_t index = 0;
		for (int i = 0; i < row; ++i) {
			for (int j = 0; j < col; ++j) {
				image.at<T>(i, j) = static_cast<T>(mat[index]);
				index++;
			}
		}

		return image;
	}












}




#endif //HAS_OPENCV_HEADER

#endif //HAS_CPLUS_17

#endif //IO_IOIMG_H

