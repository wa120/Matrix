#include "Matrix.h"
#include <opencv2\opencv.hpp>
#include <math.h>

using namespace std;
using namespace cv;

float* circshift(const cv::Mat &patch, int x_rot, int y_rot);

Matrix::Matrix(const int& width, const int& height)
	:width(width),height(height), channel(1),tensor(1),
	 area(width*height), total(area),tensor_area(area),
	 data((float *)malloc(sizeof(float)*total))
{
}
Matrix::Matrix(const int& width, const int& height, const int& channel)
	: width(width), height(height), channel(channel), tensor(1), 
	  area(width*height), total(area*channel), tensor_area(area),
	  data((float *)malloc(sizeof(float)*total))
{
}
Matrix::Matrix(const int& width, const int& height, const int& channel, const int& tensor)
	: width(width), height(height),  channel(channel), tensor(tensor),
	  area(width*height),total(area*channel*tensor), tensor_area(area*tensor),
	  data((float *)malloc(sizeof(float)*total))
{
}

void Matrix::dft()
{
}
void Matrix::idft()
{
}
Matrix Matrix::correlation()
{
}
Matrix Matrix::correlation(const Matrix& m)
{
}
Matrix Matrix::gaussianCorrelation(const Matrix& m, float sigma)
{
}
void Matrix::createCosineWindow()
{
	cv::Mat m1(1, height, CV_32FC1), m2(width, 1, CV_32FC1);
	double N_inv = 1. / (static_cast<double>(height) - 1.);
	for (int i = 0; i < height; ++i)
		m1.at<float>(i) = 0.5*(1. - std::cos(2. * CV_PI * static_cast<double>(i) * N_inv));
	N_inv = 1. / (static_cast<double>(width) - 1.);
	for (int i = 0; i < width; ++i)
		m2.at<float>(i) = 0.5*(1. - std::cos(2. * CV_PI * static_cast<double>(i) * N_inv));
	cv::Mat ret = m2*m1;
	data=(float *)ret.data;
}
void Matrix::createGaussianKernel(const float& sigma)
{
	cv::Mat labels(width, height, CV_32FC1);
	int range_y[2] = { -width / 2, width - width / 2 };
	int range_x[2] = { -height / 2, height - height / 2 };

	double sigma_s = sigma*sigma;

	for (int y = range_y[0], j = 0; y < range_y[1]; ++y, ++j) {
		float * row_ptr = labels.ptr<float>(j);
		double y_s = y*y;
		for (int x = range_x[0], i = 0; x < range_x[1]; ++x, ++i) {
			row_ptr[i] = std::exp(-0.5 * (y_s + x*x) / sigma_s);
		}
	}

	//rotate so that 1 is at top-left corner (see KCF paper for explanation)
	data = circshift(labels, range_x[0], range_y[0]);
	//sanity check, 1 at top left corner
	assert(data >= 1.f - 1e-10f);
	
}
Matrix Matrix::magnitude()
{
	Matrix result(width, height, 1, tensor);
	float *result_ptr = result.data;
	const float *ptr = data;

	if (channel == 2)
	{
		for (int j = 0; j < tensor_area; j++)
		{
			*result_ptr  = *ptr**ptr;	ptr++;
			*result_ptr += *ptr**ptr;   ptr++;
			*result_ptr = sqrt(*result_ptr);
			result_ptr++;
		}
	}
	return result;
}
Matrix Matrix::phase()
{
	Matrix result(width, height, 1, tensor);
	float *result_ptr = result.data;
	const float *ptr = data;

	if (channel == 2)
	{
		for (int j = 0; j < tensor_area; j++)
		{
			*result_ptr = *ptr;					ptr++;
			*result_ptr = *ptr / *result_ptr;   ptr++;
			*result_ptr = atan(*result_ptr);
			result_ptr++;
		}
	}
	return result;
}
Matrix Matrix::conjunction()
{
	Matrix result(width, height, 2, tensor);
	float *result_ptr = result.data;
	const float *ptr = data;

	if (channel == 2)
	{
		for (int i = 0; i < tensor_area; i++)
		{
			*result_ptr =  *ptr;	ptr++;	result_ptr++;
			*result_ptr = -*ptr;	ptr++;	result_ptr++;
		}
	}
	return result;
}
void Matrix::runingAvage(Matrix& m, const float& factor)
{
	*this = *this*(1 - factor) + m * factor;
}
void Matrix::show()
{

}
Matrix Matrix::operator +(const Matrix& m)
{
	return mat_operator_mat([](const float& m_ptr, const float& ptr, float& result_ptr) {result_ptr = m_ptr + ptr; }, m.data);		
}
Matrix Matrix::operator -(const Matrix& m)
{
	return mat_operator_mat([](const float& m_ptr, const float& ptr, float& result_ptr) {result_ptr = m_ptr - ptr; }, m.data);
}
Matrix Matrix::operator *(const Matrix& m)
{
	return mat_operator_mat([](const float& m_ptr, const float& ptr, float& result_ptr) {result_ptr = m_ptr * ptr; }, m.data);
}
Matrix Matrix::operator /(const Matrix& m)
{
	return mat_operator_mat([](const float& m_ptr, const float& ptr, float& result_ptr) {result_ptr = m_ptr / ptr; }, m.data);
}

Matrix Matrix::operator +(const float& num)
{
	return mat_operator_num([](const float& num, const float& ptr, float& result_ptr) {result_ptr = num + ptr; }, num);
}
Matrix Matrix::operator -(const float& num)
{
	return mat_operator_num([](const float& num, const float& ptr, float& result_ptr) {result_ptr = num - ptr; }, num);
}
Matrix Matrix::operator *(const float& num)
{
	return mat_operator_num([](const float& num, const float& ptr, float& result_ptr) {result_ptr = num * ptr; }, num);
}
Matrix Matrix::operator /(const float& num)
{
	return mat_operator_num([](const float& num, const float& ptr, float& result_ptr) {result_ptr = num / ptr; }, num);
}
inline Matrix Matrix::mat_operator_mat(void (*op)(const float& m_ptr, const float& ptr,float& result_ptr),const float* m_ptr)
{
	Matrix result(width, height, channel, tensor);
	float *result_ptr = result.data;
	const float *ptr = data;

	for (int i = 0; i < total; i++)
	{
		op(*m_ptr, *ptr, *result_ptr);
		ptr++;	m_ptr++;	result_ptr++;
	}
	return result;
}
inline Matrix Matrix::mat_operator_num(void(*op)(const float& m_ptr, const float& ptr, float& result_ptr), const float& num)
{
	Matrix result(width, height, channel, tensor);
	float *result_ptr = result.data;
	const float *ptr = data;

	for (int i = 0; i < total; i++)
	{
		op(num, *ptr, *result_ptr);
		ptr++;		result_ptr++;
	}
	return result;
}
float Matrix::calcSunSquareNorm()
{
	float sum_sqr_norm = 0;
	const float *ptr = data;
	for (int i = 0; i<total; i++)
		sum_sqr_norm += *ptr**ptr; ptr++;

	return sum_sqr_norm / static_cast<float>(area);
}


inline
float* circshift(const cv::Mat &patch, int x_rot, int y_rot)
{
	cv::Mat rot_patch(patch.size(), CV_32FC1);
	cv::Mat tmp_x_rot(patch.size(), CV_32FC1);

	//circular rotate x-axis
	if (x_rot < 0) {
		//move part that does not rotate over the edge
		cv::Range orig_range(-x_rot, patch.cols);
		cv::Range rot_range(0, patch.cols - (-x_rot));
		patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));

		//rotated part
		orig_range = cv::Range(0, -x_rot);
		rot_range = cv::Range(patch.cols - (-x_rot), patch.cols);
		patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));
	}
	else if (x_rot > 0) {
		//move part that does not rotate over the edge
		cv::Range orig_range(0, patch.cols - x_rot);
		cv::Range rot_range(x_rot, patch.cols);
		patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));

		//rotated part
		orig_range = cv::Range(patch.cols - x_rot, patch.cols);
		rot_range = cv::Range(0, x_rot);
		patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));
	}
	else {    //zero rotation
			  //move part that does not rotate over the edge
		cv::Range orig_range(0, patch.cols);
		cv::Range rot_range(0, patch.cols);
		patch(cv::Range::all(), orig_range).copyTo(tmp_x_rot(cv::Range::all(), rot_range));
	}

	//circular rotate y-axis
	if (y_rot < 0) {
		//move part that does not rotate over the edge
		cv::Range orig_range(-y_rot, patch.rows);
		cv::Range rot_range(0, patch.rows - (-y_rot));
		tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));

		//rotated part
		orig_range = cv::Range(0, -y_rot);
		rot_range = cv::Range(patch.rows - (-y_rot), patch.rows);
		tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));
	}
	else if (y_rot > 0) {
		//move part that does not rotate over the edge
		cv::Range orig_range(0, patch.rows - y_rot);
		cv::Range rot_range(y_rot, patch.rows);
		tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));

		//rotated part
		orig_range = cv::Range(patch.rows - y_rot, patch.rows);
		rot_range = cv::Range(0, y_rot);
		tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));
	}
	else { //zero rotation
		   //move part that does not rotate over the edge
		cv::Range orig_range(0, patch.rows);
		cv::Range rot_range(0, patch.rows);
		tmp_x_rot(orig_range, cv::Range::all()).copyTo(rot_patch(rot_range, cv::Range::all()));
	}

	return (float *)rot_patch.data;
}
Matrix::~Matrix()
{
}
