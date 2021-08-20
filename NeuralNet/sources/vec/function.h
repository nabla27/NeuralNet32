#ifndef VECTOR_FUNCTION_H
#define VECTOR_FUNCTION_H

#include <vector>
#include <random>
#include <iostream>
#include <algorithm>



namespace vec {

	using vector1d = std::vector<double>;
	using vector2d = std::vector<vector1d>;
	using vector3d = std::vector<vector2d>;



	vector2d dot(const vector2d& matA, const vector2d& matB)
	{
		size_t row_a = matA.size();
		size_t col_b = matB[0].size();
		size_t col_a = matA[0].size();
		vector2d mul(row_a, vector1d(col_b));

		for (size_t i = 0; i < row_a; ++i) {
			for (size_t j = 0; j < col_b; ++j) {
				for (size_t k = 0; k < col_a; ++k) {
					mul[i][j] += matA[i][k] * matB[k][j];
				}
			}
		}
		return mul;
	}

	vector2d trans(const vector2d& mat)
	{
		size_t row = mat.size();
		size_t col = mat[0].size();
		vector2d ts(col, vector1d(row));

		for (size_t i = 0; i < row; ++i) {
			for (size_t j = 0; j < col; ++j) {
				ts[j][i] = mat[i][j];
			}
		}
		return ts;
	}

	vector2d sqrt(const vector2d& mat)
	{
		size_t row = mat.size();
		size_t col = mat[0].size();
		vector2d tmp(row, vector1d(col));

		for (size_t i = 0; i < row; ++i) {
			for (size_t j = 0; j < col; ++j) {
				tmp[i][j] = std::sqrt(mat[i][j]);
			}
		}
		return tmp;
	}

	vector1d sqrt(const vector1d& mat)
	{
		size_t size = mat.size();
		vector1d tmp(size);

		for (size_t i = 0; i < size; ++i) {
			tmp[i] = std::sqrt(mat[i]);
		}
		return tmp;
	}

	vector2d min(const double val, const vector2d& mat)
	{
		size_t row = mat.size();
		size_t col = mat[0].size();
		vector2d tmp(row, vector1d(col));

		for (size_t i = 0; i < row; ++i) {
			for (size_t j = 0; j < col; ++j) {
				tmp[i][j] = (val <= mat[i][j]) ? val : mat[i][j];
			}
		}
		return tmp;
	}

	vector1d min(const double val, const vector1d& mat)
	{
		size_t size = mat.size();
		vector1d tmp(size);

		for (size_t i = 0; i < size; ++i) {
			tmp[i] = (val <= mat[i]) ? val : mat[i];
		}
		return tmp;
	}

	vector2d abs(const vector2d& mat)
	{
		size_t row = mat.size();
		size_t col = mat[0].size();
		vector2d tmp(row, vector1d(col));

		for (size_t i = 0; i < row; ++i) {
			for (size_t j = 0; j < col; ++j) {
				tmp[i][j] = std::abs(mat[i][j]);
			}
		}
		return tmp;
	}

	vector1d abs(const vector1d& mat)
	{
		size_t size = mat.size();
		vector1d tmp(size);

		for (size_t i = 0; i < size; ++i) {
			tmp[i] = std::abs(mat[i]);
		}
		return tmp;
	}

	vector2d max(const vector2d& matA, const vector2d& matB)
	{
		size_t row = matA.size();
		size_t col = matA[0].size();
		vector2d tmp(row, vector1d(col));

		for (size_t i = 0; i < row; ++i) {
			for (size_t j = 0; j < col; ++j) {
				tmp[i][j] = (matA[i][j] >= matB[i][j]) ? matA[i][j] : matB[i][j];
			}
		}
		return tmp;
	}

	vector1d max(const vector1d& matA, const vector1d& matB)
	{
		size_t size = matA.size();
		vector1d tmp(size);

		for (size_t i = 0; i < size; ++i) {
			tmp[i] = (matA[i] >= matB[i]) ? matA[i] : matB[i];
		}
		return tmp;
	}

	vector2d clip(const vector2d& mat, const double min, const double max)
	{
		size_t row = mat.size();
		size_t col = mat[0].size();
		vector2d tmp(row, vector1d(col));

		for (size_t i = 0; i < row; ++i) {
			for (size_t j = 0; j < col; ++j) {
				if (mat[i][j] < min) { tmp[i][j] = min; }
				else if (mat[i][j] > max) { tmp[i][j] = max; }
				else { tmp[i][j] = mat[i][j]; }
			}
		}
		return tmp;
	}

	vector1d clip(const vector1d& mat, const double min, const double max)
	{
		size_t size = mat.size();
		vector1d tmp(size);

		for (size_t i = 0; i < size; ++i) {
			if (mat[i] < min) { tmp[i] = min; }
			else if (mat[i] > max) { tmp[i] = max; }
			else { tmp[i] = mat[i]; }
		}
		return tmp;
	}

	vector2d randomer(const vector2d& mat, const double& val1 = 0.0, const double& val2 = 1.0)
	{
		size_t row = mat.size();
		size_t col = mat[0].size();
		vector2d tmp(row, vector1d(col));

		std::random_device rnd;
		std::default_random_engine mt(rnd());
		std::normal_distribution<> dist(val1, val2);

		for (size_t i = 0; i < row; ++i) {
			for (size_t j = 0; j < col; ++j) {
				tmp[i][j] = dist(mt);
			}
		}
		return tmp;
	}

	vector1d randomer(const vector1d& mat, const double& val1 = 0.0, const double& val2 = 1.0)
	{
		size_t size = mat.size();
		vector1d tmp(size);

		std::random_device rnd;
		std::mt19937 mt(rnd());
		std::normal_distribution<> dist(val1, val2);

		for (size_t i = 0; i < size; ++i) {
			tmp[i] = dist(mt);
		}
		return tmp;
	}

	vector2d exp(const vector2d& mat)
	{
		size_t row = mat.size();
		size_t col = mat[0].size();
		vector2d tmp(row, vector1d(col));

		for (size_t i = 0; i < row; ++i) {
			for (size_t j = 0; j < col; ++j) {
				tmp[i][j] = std::exp(mat[i][j]);
			}
		}
		return tmp;
	}

	vector1d exp(const vector1d& mat)
	{
		size_t size = mat.size();
		vector1d tmp(size);

		for (size_t i = 0; i < size; ++i) {
			tmp[i] = std::exp(mat[i]);
		}
		return tmp;
	}

	void initequal(vector3d& mat, const double& val)
	{
		size_t max_a = mat.size();
		for (size_t i = 0; i < max_a; ++i) {
			size_t max_b = mat[i].size();
			for (size_t j = 0; j < max_b; ++j) {
				size_t max_c = mat[i][j].size();
				for (size_t k = 0; k < max_c; ++k) {
					mat[i][j][k] = val;
				}
			}
		}
	}

	void initequal(vector2d& mat, const double val)
	{
		size_t row = mat.size();
		for (size_t i = 0; i < row; ++i) {
			size_t col = mat[i].size();
			for (size_t j = 0; j < col; ++j) {
				mat[i][j] = val;
			}
		}
	}


	void shuffle(vector2d& mat)
	{
		std::random_device get_rand_dev;
		std::mt19937 get_rand_mt(get_rand_dev());
		std::shuffle(mat.begin(), mat.end(), get_rand_mt);
	}

	void shuffle(vector2d& matA, vector2d& matB)
	{
		size_t row = matA.size();

		std::random_device get_rand_dev;	//非決定的にシードを決定
		std::mt19937 get_rand_mt(get_rand_dev());	//シードから乱数の生成
		std::uniform_int_distribution<size_t> rnd(0, row - 1);	//0からrow-1の間の一様乱数を取得

		for (size_t i = 0; i < row; ++i)
		{
			size_t rnd_index = rnd(get_rand_mt);

			vector1d vec_a = matA[i];
			matA[i] = matA[rnd_index];
			matA[rnd_index] = vec_a;

			vector1d vec_b = matB[i];
			matB[i] = matB[rnd_index];
			matB[rnd_index] = vec_b;
		}
	}







	void element_checker(const vector2d& mat)
	{
		for (size_t i = 0; i < mat.size(); i++) {
			for (size_t j = 0; j < mat[i].size(); j++) {
				if (mat[i][j] == 0) { std::cout << "(" << i << "," << j << ") : zero == " << mat[i][j] << "\n"; }
				if (mat[i][j] >= std::numeric_limits<double>::max()) { std::cout << "(" << i << "," << j << ") : max == " << mat[i][j] << "\n"; }
				if (mat[i][j] <= -std::numeric_limits<double>::max()) { std::cout << "(" << i << "," << j << ") : min == " << mat[i][j] << "\n"; }
			}
		}
	}

	void hasNun(const vector3d& mat, const char str[] = "", bool stop = false)
	{
		std::cout << str << std::endl;
		bool flag = false;

		size_t size = mat.size();
		for (size_t i = 0; i < size; ++i) {
			size_t row = mat[i].size();
			for (size_t j = 0; j < row; ++j) {
				size_t col = mat[i][j].size();
				for (size_t k = 0; k < col; ++k) {
					if (isnan(mat[i][j][k])) {
						flag = true;
						if (stop) { throw std::runtime_error("is nun."); }
					}
				}
			}
		}
		if (flag) { std::cout << "is nan."; }
		else { std::cout << "is not nan." << std::endl; }
	}

	void hasNun(const vector2d& mat, const char str[] = "", bool stop = false)
	{
		std::cout << str << std::endl;
		bool flag = false;

		size_t row = mat.size();
		for (size_t i = 0; i < row; ++i) {
			size_t col = mat[i].size();
			for (size_t j = 0; j < col; ++j) {
				if (isnan(mat[i][j])) {
					flag = true;
					if (stop) { throw std::runtime_error("is nun."); }
				}
			}
		}
		if (flag) { std::cout << "is nan." << std::endl; }
		else { std::cout << "is not nan." << std::endl; }
	}

	void hasNun(const vector1d& mat, const char str[] = "", bool stop = false)
	{
		std::cout << str << std::endl;
		bool flag = false;

		size_t col = mat.size();
		for (size_t i = 0; i < col; ++i) {
			if (isnan(mat[i])) {
				flag = true;
				if (stop) { throw std::runtime_error("is nun."); }
			}
		}
		if (flag) { std::cout << "is nan." << std::endl; }
		else { std::cout << "is not nan." << std::endl; }
	}

	void fitsize(vector1d& mat, const vector1d& fit, double val = 0.0)
	{
		mat = vector1d(fit.size(), val);
	}

	void fitsize(vector2d& mat, const vector2d& fit, double val = 0.0)
	{
		size_t row = fit.size();
		mat = vector2d(row);

		for (size_t i = 0; i < row; ++i) {
			mat[i] = vector1d(fit[i].size(), val);
		}
	}

	void fitsize(vector3d& mat, const vector3d& fit, double val = 0.0)
	{
		size_t dim = fit.size();
		mat = vector3d(dim);

		for (size_t i = 0; i < dim; ++i) 
		{
			size_t row = fit[i].size();
			mat[i] = vector2d(row);
			
			for (size_t j = 0; j < row; ++j) {
				mat[i][j] = vector1d(fit[i][j].size(), val);
			}
		}
	}











	void show(const vector3d& mat, const char str[] = "")
	{
		std::cout << "===============" << str << "===============\n";
		for (size_t i = 0; i < mat.size(); i++) {
			for (size_t j = 0; j < mat[i].size(); j++) {
				for (size_t k = 0; k < mat[i][j].size(); k++) {
					std::cout << "  " << mat[i][j][k];
				}
				std::cout << "\n";
			}
			std::cout << "------------------------------\n";
		}
	}
	void show(const vector2d& mat, const char str[] = "")
	{
		std::cout << "===============" << str << "===============\n";
		for (size_t i = 0; i < mat.size(); i++) {
			for (size_t j = 0; j < mat[i].size(); j++) {
				std::cout << "  " << mat[i][j];
			}
			std::cout << "\n";
		}
	}
	void show(const vector1d& mat, const char str[] = "")
	{
		std::cout << "===============" << str << "===============\n";
		for (size_t i = 0; i < mat.size(); i++) {
			std::cout << "  " << mat[i];
		}
		std::cout << "\n";
	}

	template <class T>
	void show(std::vector<T>& mat, const char str[] = "")
	{
		std::cout << "<<<" << str << ">>>" << std::endl;
		for (size_t i = 0; i < mat.size(); i++) {
			std::cout << "  " << mat[i];
		}
		std::cout << "\n>>><<<\n" << std::endl;
	}

	template <class T>
	void show(std::vector<std::vector<T> >& mat, const char str[] = "")
	{
		std::cout << "<<<" << str << ">>>" << std::endl;
		for (size_t i = 0; i < mat.size(); i++) {
			for (size_t j = 0; j < mat[i].size(); j++) {
				std::cout << "  " << mat[i][j];
			}
			std::cout << "\n";
		}
		std::cout << ">>><<<\n" << std::endl;
	}

	template <class T>
	void show(std::vector<std::vector<std::vector<T> > >& mat, const char str[] = "")
	{
		std::cout << "<<<" << str << ">>>" << std::endl;
		for (size_t i = 0; i < mat.size(); i++) {
			for (size_t j = 0; j < mat[i].size(); j++) {
				for (size_t k = 0; k < mat[i][j].size(); k++) {
					std::cout << "  " << mat[i][j][k];
				}
				std::cout << "\n";
			}
			std::cout << "------------------------------\n";
		}
		std::cout << ">>><<<\n" << std::endl;
	}

}


#endif //VECTOR_FUNCTION_H
