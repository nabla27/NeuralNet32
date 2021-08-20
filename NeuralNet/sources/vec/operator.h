#ifndef VECTOR_OPERATOR_H
#define VECTOR_OPERATOR_H

#include <vector>
#include <random>
#include <iostream>


namespace vec {

	using vector1d = std::vector<double>;
	using vector2d = std::vector<vector1d>;
	using vector3d = std::vector<vector2d>;



	vector2d operator+(const vector2d& matA, const vector2d& matB) {
		vector2d ad(matA.size(), vector1d(matA[0].size()));
		size_t max_a = matA.size();
		size_t max_b = matA[0].size();
		for (size_t i = 0; i < max_a; i++) {
			for (size_t j = 0; j < max_b; j++) {
				ad[i][j] = matA[i][j] + matB[i][j];
			}
		}
		return ad;
	}
	vector2d operator+(const vector2d& matA, const vector1d& matB) {
		vector2d ad(matA.size(), vector1d(matA[0].size()));
		size_t max_a = matA.size();
		size_t max_b = matA[0].size();
		for (size_t i = 0; i < max_a; i++) {
			for (size_t j = 0; j < max_b; j++) {
				ad[i][j] = matA[i][j] + matB[j];
			}
		}
		return ad;
	}
	vector2d operator+(const vector1d& matA, const vector2d& matB) {
		vector2d ad(matB.size(), vector1d(matB[0].size()));
		size_t max_a = matB.size();
		size_t max_b = matB[0].size();
		for (size_t i = 0; i < max_a; i++) {
			for (size_t j = 0; j < max_b; j++) {
				ad[i][j] = matB[i][j] + matA[j];
			}
		}
		return ad;
	}
	vector2d operator+(const vector2d& matA, const double val) {
		vector2d ad(matA.size(), vector1d(matA[0].size()));
		size_t max_a = matA.size();
		size_t max_b = matA[0].size();
		for (size_t i = 0; i < max_a; i++) {
			for (size_t j = 0; j < max_b; j++) {
				ad[i][j] = matA[i][j] + val;
			}
		}
		return ad;
	}
	vector2d operator+(const double val, const vector2d& matA) {
		vector2d ad(matA.size(), vector1d(matA[0].size()));
		size_t max_a = matA.size();
		size_t max_b = matA[0].size();
		for (size_t i = 0; i < max_a; i++) {
			for (size_t j = 0; j < max_b; j++) {
				ad[i][j] = matA[i][j] + val;
			}
		}
		return ad;
	}
	vector1d operator+(const vector1d& matA, const vector1d& matB) {
		vector1d tmp(matA.size());
		size_t max = matA.size();
		for (size_t i = 0; i < max; i++) {
			tmp[i] = matA[i] + matB[i];
		}
		return tmp;
	}
	vector1d operator+(const vector1d& matA, const double val) {
		vector1d tmp(matA.size());
		size_t max = matA.size();
		for (size_t i = 0; i < max; i++) {
			tmp[i] = matA[i] + val;
		}
		return tmp;
	}
	vector1d operator+(const double val, const vector1d& matA) {
		vector1d tmp(matA.size());
		size_t max = matA.size();
		for (size_t i = 0; i < max; i++) {
			tmp[i] = matA[i] + val;
		}
		return tmp;
	}


	vector2d operator-(const vector2d& matA, const vector2d& matB) {
		vector2d tmp(matA.size(), vector1d(matA[0].size()));
		size_t max_a = matA.size();
		size_t max_b = matA[0].size();
		for (size_t i = 0; i < max_a; i++) {
			for (size_t j = 0; j < max_b; j++) {
				tmp[i][j] = matA[i][j] - matB[i][j];
			}
		}
		return tmp;
	}
	vector2d operator-(const vector2d& matA, const vector1d& matB) {
		vector2d tmp(matA.size(), vector1d(matA[0].size()));
		size_t max_a = matA.size();
		size_t max_b = matA[0].size();
		for (size_t i = 0; i < max_a; i++) {
			for (size_t j = 0; j < max_b; j++) {
				tmp[i][j] = matA[i][j] - matB[j];
			}
		}
		return tmp;
	}

	vector2d operator-(const vector1d& matA, const vector2d& matB) {
		vector2d tmp(matB.size(), vector1d(matB[0].size()));
		size_t max_a = matB.size();
		size_t max_b = matB[0].size();
		for (size_t i = 0; i < max_a; i++) {
			for (size_t j = 0; j < max_b; j++) {
				tmp[i][j] = matA[j] - matB[i][j];
			}
		}
		return tmp;
	}
	vector2d operator-(const vector2d& matA, const double val) {
		vector2d tmp(matA.size(), vector1d(matA[0].size()));
		size_t max_a = matA.size();
		size_t max_b = matA[0].size();
		for (size_t i = 0; i < max_a; i++) {
			for (size_t j = 0; j < max_b; j++) {
				tmp[i][j] = matA[i][j] - val;
			}
		}
		return tmp;
	}
	vector2d operator-(const double val, const vector2d& matA) {
		vector2d tmp(matA.size(), vector1d(matA[0].size()));
		size_t max_a = matA.size();
		size_t max_b = matA[0].size();
		for (size_t i = 0; i < max_a; i++) {
			for (size_t j = 0; j < max_b; j++) {
				tmp[i][j] = val - matA[i][j];
			}
		}
		return tmp;
	}
	vector1d operator-(const vector1d& matA, const vector1d& matB) {
		vector1d tmp(matA.size());
		size_t max = matA.size();
		for (size_t i = 0; i < max; i++) {
			tmp[i] = matA[i] - matB[i];
		}
		return tmp;
	}
	vector1d operator-(const vector1d& matA, const double val) {
		vector1d tmp(matA.size());
		size_t max = matA.size();
		for (size_t i = 0; i < max; i++) {
			tmp[i] = matA[i] - val;
		}
		return tmp;
	}
	vector1d operator-(const double val, const vector1d& matA) {
		vector1d tmp(matA.size());
		size_t max = matA.size();
		for (size_t i = 0; i < max; i++) {
			tmp[i] = val - matA[i];
		}
		return tmp;
	}




	vector2d operator*(const vector2d& matA, const vector2d& matB) {
		vector2d tmp(matA.size(), vector1d(matA[0].size()));
		size_t max_a = matA.size();
		size_t max_b = matA[0].size();
		for (size_t i = 0; i < max_a; i++) {
			for (size_t j = 0; j < max_b; j++) {
				tmp[i][j] = matA[i][j] * matB[i][j];
			}
		}
		return tmp;
	}
	vector2d operator*(const vector2d& matA, const vector1d& matB) {
		vector2d tmp(matA.size(), vector1d(matA[0].size()));
		size_t max_a = matA.size();
		size_t max_b = matA[0].size();
		for (size_t i = 0; i < max_a; i++) {
			for (size_t j = 0; j < max_b; j++) {
				tmp[i][j] = matA[i][j] * matB[j];
			}
		}
		return tmp;
	}
	vector2d operator*(const vector1d& matA, const vector2d& matB) {
		vector2d tmp(matB.size(), vector1d(matB[0].size()));
		size_t max_a = matB.size();
		size_t max_b = matB[0].size();
		for (size_t i = 0; i < max_a; i++) {
			for (size_t j = 0; j < max_b; j++) {
				tmp[i][j] = matB[i][j] * matA[j];
			}
		}
		return tmp;
	}
	vector2d operator*(const vector2d& matA, const double val) {
		vector2d tmp(matA.size(), vector1d(matA[0].size()));
		size_t max_a = matA.size();
		size_t max_b = matA[0].size();
		for (size_t i = 0; i < max_a; i++) {
			for (size_t j = 0; j < max_b; j++) {
				tmp[i][j] = matA[i][j] * val;
			}
		}
		return tmp;
	}
	vector2d operator*(const double val, const vector2d& matA) {
		vector2d tmp(matA.size(), vector1d(matA[0].size()));
		size_t max_a = matA.size();
		size_t max_b = matA[0].size();
		for (size_t i = 0; i < max_a; i++) {
			for (size_t j = 0; j < max_b; j++) {
				tmp[i][j] = matA[i][j] * val;
			}
		}
		return tmp;
	}
	vector1d operator*(const vector1d& matA, const vector1d& matB) {
		vector1d tmp(matA.size());
		size_t max = matA.size();
		for (size_t i = 0; i < max; i++) {
			tmp[i] = matA[i] * matB[i];
		}
		return tmp;
	}
	vector1d operator*(const vector1d& matA, const double val) {
		vector1d tmp(matA.size());
		size_t max = matA.size();
		for (size_t i = 0; i < max; i++) {
			tmp[i] = matA[i] * val;
		}
		return tmp;
	}
	vector1d operator*(const double val, const vector1d& matA) {
		vector1d tmp(matA.size());
		size_t max = matA.size();
		for (size_t i = 0; i < max; i++) {
			tmp[i] = matA[i] * val;
		}
		return tmp;
	}



	vector2d operator/(const vector2d& matA, const vector2d& matB) {
		vector2d tmp(matA.size(), vector1d(matA[0].size()));
		size_t max_a = matA.size();
		size_t max_b = matA[0].size();
		for (size_t i = 0; i < max_a; i++) {
			for (size_t j = 0; j < max_b; j++) {
				tmp[i][j] = matA[i][j] / matB[i][j];
			}
		}
		return tmp;
	}
	vector2d operator/(const vector2d& matA, const vector1d& matB) {
		vector2d tmp(matA.size(), vector1d(matA[0].size()));
		size_t max_a = matA.size();
		size_t max_b = matA[0].size();
		for (size_t i = 0; i < max_a; i++) {
			for (size_t j = 0; j < max_b; j++) {
				tmp[i][j] = matA[i][j] / matB[j];
			}
		}
		return tmp;
	}
	vector2d operator/(const vector1d& matA, const vector2d& matB) {
		vector2d tmp(matB.size(), vector1d(matB[0].size()));
		size_t max_a = matB.size();
		size_t max_b = matB[0].size();
		for (size_t i = 0; i < max_a; i++) {
			for (size_t j = 0; j < max_b; j++) {
				tmp[i][j] = matA[j] / matB[i][j];
			}
		}
		return tmp;
	}
	vector2d operator/(const vector2d& matA, const double val) {
		vector2d tmp(matA.size(), vector1d(matA[0].size()));
		size_t max_a = matA.size();
		size_t max_b = matA[0].size();
		for (size_t i = 0; i < max_a; i++) {
			for (size_t j = 0; j < max_b; j++) {
				tmp[i][j] = matA[i][j] / val;
			}
		}
		return tmp;
	}
	vector2d operator/(const double val, const vector2d& matA) {
		vector2d tmp(matA.size(), vector1d(matA[0].size()));
		size_t max_a = matA.size();
		size_t max_b = matA[0].size();
		for (size_t i = 0; i < max_a; i++) {
			for (size_t j = 0; j < max_b; j++) {
				tmp[i][j] = val / matA[i][j];
			}
		}
		return tmp;
	}
	vector1d operator/(const vector1d& matA, const vector1d& matB) {
		vector1d tmp(matA.size());
		size_t max = matA.size();
		for (size_t i = 0; i < max; i++) {
			tmp[i] = matA[i] / matB[i];
		}
		return tmp;
	}
	vector1d operator/(const vector1d& matA, const double val) {
		vector1d tmp(matA.size());
		size_t max = matA.size();
		for (size_t i = 0; i < max; i++) {
			tmp[i] = matA[i] / val;
		}
		return tmp;
	}
	vector1d operator/(const double val, const vector1d& matA) {
		vector1d tmp(matA.size());
		size_t max = matA.size();
		for (size_t i = 0; i < max; i++) {
			tmp[i] = val / matA[i];
		}
		return tmp;
	}


	void operator+=(vector2d& matA, const vector2d& matB) {
		matA = matA + matB;
	}
	void operator+=(vector1d& matA, const vector1d& matB) {
		matA = matA + matB;
	}

	void operator-=(vector2d& matA, const vector2d& matB) {
		matA = matA - matB;
	}
	void operator-=(vector1d& matA, const vector1d& matB) {
		matA = matA - matB;
	}

	void operator*=(vector2d& matA, const vector2d& matB) {
		matA = matA * matB;
	}
	void operator*=(vector1d& matA, const vector1d& matB) {
		matA = matA * matB;
	}

	void operator/=(vector2d& matA, const vector2d& matB) {
		matA = matA / matB;
	}
	void operator/=(vector1d& matA, const vector1d& matB) {
		matA = matA / matB;
	}


}

#endif //VECTOR_OPERATOR_H
