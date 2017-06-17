#pragma once
#include <iostream>
#include <opencv2\opencv.hpp>

class Matrix
{
public:
	int width;
	int height;
	int area;
	int channel;
	int channel_area;
	int tensor;
	int total;
	float *data;

	Matrix();
	Matrix(const cv::Mat& m);
	Matrix(const Matrix& m);
	Matrix(const int& width, const int& height);
	Matrix(const int& width, const int& height, const int& channel);
	Matrix(const int& width, const int& height, const int& channel, const int& tensor);

	Matrix createGaussianKernel(const float& sigma);
	Matrix createCosineWindow();

	Matrix correlation();
	Matrix correlation(const Matrix& m);
	Matrix gaussianCorrelation(const Matrix& m, float sigma);
	Matrix reduceTensor();

	void gradient(Matrix& m, Matrix& o);
	Matrix fhog();
	Matrix magnitude();
	Matrix phase();

	float calcSumSquareNorm() const;
	void runingAvage(Matrix& m, const float& factor);

	Matrix fft();
	Matrix ifft();

	Matrix clone();
	Matrix clone(const int& x, const int& y, const int& w,const int& h, const bool& paddingAlignment);

	Matrix softMax(const float& compare_val,const float& data);
	Matrix exp();
	void show();

	Matrix operator +(const Matrix& m);
	Matrix operator -(const Matrix& m);
	Matrix operator *(const Matrix& m);
	Matrix operator /(const Matrix& m);
	Matrix operator =(const Matrix& m);
	Matrix operator ~() const;
	

	Matrix operator +(const float& m);
	Matrix operator -(const float& m);
	Matrix operator *(const float& m);
	Matrix operator /(const float& m);

	
	~Matrix();

	int align_area;
	int align_channel_area;
	int align_total;

private:

};
