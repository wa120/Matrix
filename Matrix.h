#pragma once
#include <iostream>


class Matrix
{
public:
	int width;
	int height;
	int area;
	int channel;
	int tensor;
	int tensor_area;
	int total;
	float *data;

	Matrix(const int& width, const int& height);
	Matrix(const int& width, const int& height, const int& channel);
	Matrix(const int& width, const int& height, const int& channel, const int& tensor);

	void createGaussianKernel(const float& sigma);
	void createCosineWindow();

	Matrix correlation();
	Matrix correlation(const Matrix& m);
	Matrix gaussianCorrelation(const Matrix& m, float sigma);

	Matrix magnitude();
	Matrix phase();
	Matrix conjunction();

	float calcSunSquareNorm();
	void runingAvage(Matrix& m, const float& factor);

	void dft();
	void idft();

	void show();

	Matrix operator +(const Matrix& m);
	Matrix operator -(const Matrix& m);
	Matrix operator *(const Matrix& m);
	Matrix operator /(const Matrix& m);

	Matrix operator +(const float& m);
	Matrix operator -(const float& m);
	Matrix operator *(const float& m);
	Matrix operator /(const float& m);

	
	~Matrix();

private:
	Matrix mat_operator_mat(void(*op)(const float& m_ptr, const float& ptr, float& result_ptr), const float* m_ptr);
	Matrix mat_operator_num(void(*op)(const float& m_ptr, const float& ptr, float& result_ptr), const float& num);
};
