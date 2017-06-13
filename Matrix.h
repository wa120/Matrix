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
	Matrix reduceTensor();

	Matrix magnitude();
	Matrix phase();

	float calcSumSquareNorm() const;
	void runingAvage(Matrix& m, const float& factor);

	void fft();
	void ifft();

	void show();

	Matrix operator +(const Matrix& m);
	Matrix operator -(const Matrix& m);
	Matrix operator *(const Matrix& m);
	Matrix operator /(const Matrix& m);
	Matrix operator =(const Matrix& m);
	Matrix operator ~();
	

	Matrix operator +(const float& m);
	Matrix operator -(const float& m);
	Matrix operator *(const float& m);
	Matrix operator /(const float& m);

	
	~Matrix();

private:
	
};
