#include "Matrix.h"
#include "piotr_fhog/sse.hpp"
#include <math.h>

using namespace std;
using namespace cv;

float* circshift(const cv::Mat &patch, int x_rot, int y_rot);
inline Matrix mat_operator_mat(const Matrix& m1, const Matrix& m2,
					void(*_op)(const __m128& ptr, const __m128& m_ptr, __m128& result_ptr),
					void(*op) (const  float& ptr,  const float& m_ptr,  float& result_ptr));
inline Matrix mat_operator_num(const Matrix& m1,  const float& num, 
					void(*_op)(const __m128& ptr, const  float& m_ptr, __m128& result_ptr),
					void(*op) (const  float& ptr,  const float& m_ptr,  float& result_ptr));
inline float* initAcosTable();
static float *acosTable;

Matrix::Matrix()
	:width(), height(), channel(), tensor(),
	area(), channel_area(), total(),
	align_total(),align_channel_area(),align_area(),
	data()
{}
Matrix::Matrix(const cv::Mat& m)
	: width(m.cols), height(m.rows), channel(m.channels()), tensor(1),
	area(width*height), channel_area(m.total()), total(channel_area),
	align_area(area-area%4), align_channel_area(channel_area-channel_area%4),align_total(align_channel_area),
	data((float *)m.data)
{
}
Matrix::Matrix(const Matrix& m)
	:width(m.width), height(m.height), channel(m.channel), tensor(m.tensor),
	area(m.area), channel_area(m.channel_area), total(m.total),
	align_area(area-area%4), align_channel_area(channel_area-channel_area%4), 
	align_total(total-total%4),
	data(m.data)
{
}

Matrix::Matrix(const int& width, const int& height)
	:width(width),height(height), channel(1),tensor(1),
	 area(width*height), channel_area(area), total(area),
	 align_area(area-area%4), align_channel_area(align_area),align_total(align_area),
	 data((float *)malloc(sizeof(float)*total))
{
}
Matrix::Matrix(const int& width, const int& height, const int& channel)
	: width(width), height(height), channel(channel), tensor(1), 
	  area(width*height), channel_area(area*channel), total(channel_area),
	  align_area(area-area%4), align_channel_area(channel_area-channel_area%4), 
	  align_total(align_channel_area),
	  data((float *)malloc(sizeof(float)*total))
{
}
Matrix::Matrix(const int& width, const int& height, const int& channel, const int& tensor)
	: width(width), height(height),  channel(channel), tensor(tensor),
	  area(width*height), channel_area(area*channel), total(channel_area*tensor),
	  align_area(area-area%4), align_channel_area(channel_area-channel_area%4), 
	  align_total(total-total%4),
	  data((float *)malloc(sizeof(float)*total))
{
}

Matrix Matrix::fft()
{
	
}
Matrix Matrix::ifft()
{

}
Matrix Matrix::correlation()
{
	return magnitude();
}
Matrix Matrix::correlation(const Matrix& m)
{
	return *this*(~m);
}
Matrix Matrix::gaussianCorrelation(const Matrix& m, float sigma)
{
	float this_square_norm = calcSumSquareNorm();
	float m_square_norm;
	Matrix sumMat;
	if (data == m.data)
	{
		sumMat = magnitude().ifft().reduceTensor();
		m_square_norm = this_square_norm;
	}
	else
	{
		sumMat = (*this*(~m)).ifft().reduceTensor();
		m_square_norm = m.calcSumSquareNorm();
	}
	
	float numel_inv = 1.f / (channel_area);
	Matrix result = (this_square_norm + m_square_norm - 2.0f * sumMat).softMax(0, 0).exp();
	return result;
}
Matrix Matrix::reduceTensor()
{
	Matrix result(width,height,channel,1);
	float *result_ptr = result.data;
	float **channels = (float **)malloc(channel * sizeof(float *));
	float **channels_ptr = channels;
	float *ptr=data;
	for (int i = 0; i < channel; i++)
	{
		*channels_ptr = data;
		data += area;
		channels_ptr++;
	}
	for (int i = 0; i < tensor; i++)
	{
		channels_ptr = channels;
		result_ptr = result.data;
		for (int j = 0; j < channel; j++)
		{
			__m128 *_channels_ptr = (__m128 *) *channels_ptr ;
			__m128 *_result_ptr = (__m128 *)result_ptr;
			int k = 0;
			for (; k < align_area; k+=4)
			{
				*_result_ptr = ADD(*_result_ptr, *_channels_ptr);
				_result_ptr++;	channels_ptr++;
			}
			*channels_ptr = (float *)_channels_ptr;
			result_ptr = (float *)_result_ptr;
			for (; k < area; k++)
			{
				*result_ptr += **channels_ptr;
				result_ptr++;	*channels_ptr++;
			}
			*channels_ptr += channel_area;
			channels_ptr++;
		}
	}
	delete channels;
	return result;
}
void Matrix::gradient(Matrix& m, Matrix& o)
{

}
Matrix Matrix::fhog()
{
}
Matrix Matrix::clone()
{
	Matrix result(width, height, channel, tensor);

	float *ptr = data;
	float *result_ptr = result.data;
	__m128 *_ptr = (__m128 *)ptr;
	__m128 *_result_ptr = (__m128 *)result_ptr;

	int i = 0;
	for (;i < align_total; i+=4)
	{
		*_result_ptr = *_ptr;
		_result_ptr++;	_ptr++;
	}
	ptr = (float *)_ptr;
	result_ptr = (float *)_result_ptr;
	for (;i < total; i++)
	{
		*result_ptr = *ptr;
		result_ptr++;	ptr++;
	}
	return result;
}

Matrix Matrix::clone(const int& x, const int& y, const int& w, const int& h,const bool& paddingAlignment=true)
{
	int padding_w = w;
	int align_w;
	int mode_w = w % 4;
	
	if (paddingAlignment)
	{
		padding_w += (x + w + mode_w < width)  ? 4 - mode_w : -mode_w;
		align_w = padding_w;
	}
	else
	{
		align_w = w - mode_w;
	}
	int diff_w = width - padding_w;
	int next_channel_step = (diff_w -x) + (height -y -h)*width+channel_area;
	Matrix result(padding_w, h, channel, tensor);
	float *ptr = data;
	float *result_ptr = result.data;
	float **channels = (float **)malloc(channel * sizeof(float *));
	float **channels_ptr = channels;
	for (int i = 0; i < channel; i++)
	{
		*channels_ptr = ptr;
		channels_ptr++;	ptr += area;
	}
	for (int i = 0; i < tensor; i++)
	{
		channels_ptr = channels;
		for (int j = 0; j < channel; j++)
		{
			*channels_ptr += (x - 1)*width + y;

			for (int k=0; k < h; k++)
			{
				__m128 *_channels_ptr = (__m128 *) *channels_ptr;
				__m128 *_result_ptr = (__m128 *) result_ptr;
				int l = 0;
				for (; l < align_w; l+=4)
				{
					*_result_ptr = *_channels_ptr;
					_result_ptr++;	_channels_ptr++;
				}
				*channels_ptr = (float *)_channels_ptr;
				result_ptr = (float *)_result_ptr;
				for (;l < w; l++)
				{
					*result_ptr = **channels_ptr;
					result_ptr ++ ;	*channels_ptr++;
				}
				*channels_ptr += diff_w;
			}
			*channels_ptr += next_channel_step;
			channels_ptr++;			
		}
	}
	delete channels;
	return result;
}
Matrix Matrix::createCosineWindow()
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
	return *this;
}
Matrix Matrix::createGaussianKernel(const float& sigma)
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
	return *this;
}
Matrix Matrix::magnitude()
{
	Matrix result(width, height, 1, tensor);
	float *result_ptr = result.data;
	const float *ptr1 = data;
	const float *ptr2 = data + area;

	if (channel == 2)
	{
		for (int i = 0; i < tensor; i++)
		{
			__m128 *_ptr1 = (__m128 *)ptr1;
			__m128 *_ptr2 = (__m128 *)ptr2;
			__m128 *_result_ptr = (__m128 *)result_ptr;

			int j = 0;
			for (;j < align_area; j++)
			{
				*_result_ptr = SQRT(ADD(MUL(*_ptr1, *_ptr1), MUL(*_ptr2, *_ptr2)));
				_result_ptr++;	_ptr1++;	_ptr2++;
			}
			result_ptr = (float *)_result_ptr;
			ptr1 = (float *)_ptr1;
			ptr2 = (float *)_ptr2;
			for (; j < area; j++)
			{
				*result_ptr = sqrt((*ptr1**ptr1) + (*ptr2**ptr2));
				result_ptr++;	ptr1++;	ptr2++;
			}
			result_ptr++;
		}
	}
	return result;
}
Matrix Matrix::phase()
{
	
	acosTable = initAcosTable();
	Matrix result(width, height, 1, tensor);
	float *result_ptr = result.data;
	const float *ptr1 = data;
	const float *ptr2 = data + area;

	if (channel == 2)
	{
		for (int i = 0; i < tensor; i++)
		{
			__m128 *_ptr1 = (__m128 *)ptr1;
			__m128 *_ptr2 = (__m128 *)ptr2;
			__m128 *_result_ptr = (__m128 *)result_ptr;

			int j = 0;
			for (; j < align_area; j++)
			{
				*_result_ptr = ATAN(DIV(*_ptr2, *_ptr1));
				_result_ptr++;	_ptr1++;	_ptr2++;
			}
			result_ptr = (float *)_result_ptr;
			ptr1 = (float *)_ptr1;
			ptr2 = (float *)_ptr2;
			for (; j < area; j++)
			{
				*result_ptr = atan(*ptr2/ *ptr1);
				result_ptr++;	ptr1++;	ptr2++;
			}
			result_ptr++;
		}
	}
	return result;
}

void Matrix::runingAvage(Matrix& m, const float& factor)
{
	*this = *this*(1 - factor) + m * factor;
}
Matrix Matrix::softMax(const float& compare_val, const float& data)
{

}
Matrix Matrix::exp()
{
}
void Matrix::show()
{

}
Matrix Matrix::operator +(const Matrix& m)
{
	return mat_operator_mat(*this,m,
				[](const __m128& ptr, const __m128& m_ptr, __m128& result_ptr) {result_ptr = ADD(ptr, m_ptr); },
				[](const  float& ptr,  const float& m_ptr,  float& result_ptr) {result_ptr = ptr + m_ptr; }
							);		
}
Matrix Matrix::operator -(const Matrix& m)
{
	return mat_operator_mat(*this, m,
		[](const __m128& ptr, const __m128& m_ptr, __m128& result_ptr) {result_ptr = SUB(ptr, m_ptr); },
		[](const  float& ptr, const  float& m_ptr,  float& result_ptr) {result_ptr = ptr - m_ptr; }
							);
}
Matrix Matrix::operator *(const Matrix& m)
{
	return mat_operator_mat(*this, m,
		[](const __m128& ptr, const __m128& m_ptr, __m128& result_ptr) {result_ptr = MUL(ptr, m_ptr); },
		[](const  float& ptr, const  float& m_ptr, float& result_ptr) {result_ptr = ptr * m_ptr; }
	);
}
Matrix Matrix::operator /(const Matrix& m)
{
	return mat_operator_mat(*this, m,
		[](const __m128& ptr, const __m128& m_ptr, __m128& result_ptr) {result_ptr = DIV(ptr, m_ptr); },
		[](const  float& ptr, const  float& m_ptr, float& result_ptr) {result_ptr = ptr / m_ptr; }
	);
}
Matrix Matrix::operator ~()  const
{
	Matrix result(width, height, 2, tensor);
	float *result_ptr1 = result.data;
	float *result_ptr2 = result.data+area;

	const float *ptr1 = data;
	const float *ptr2 = data + area;

	if (channel == 2)
	{
		for (int i = 0; i < tensor; i++)
		{
			
			__m128 *_result_ptr1 = (__m128 *)result_ptr1;
			__m128 *_result_ptr2 = (__m128 *)result_ptr2;

			__m128 *_ptr1 = (__m128 *)ptr1;
			__m128 *_ptr2 = (__m128 *)ptr2;
			int j = 0;
			for (; j < align_area; j+=4)
			{
				*_result_ptr1 = *_ptr1;
				*_result_ptr2 = NEG(*_ptr2);
				_result_ptr1++;	_ptr1++;
				_result_ptr2++;	_ptr2++;
			}
			result_ptr1 = (float *)_result_ptr1;
			result_ptr2 = (float *)_result_ptr2;
			ptr1 = (float *)_ptr1;
			ptr2 = (float *)_ptr2;
			for (; j < area; j++)
			{
				*result_ptr1 =  *ptr1;
				*result_ptr2 = -*ptr2;
				result_ptr1++;	ptr1++;
				result_ptr2++;	ptr2++;
			}
			result_ptr1 += area;
			result_ptr2 += area;
			ptr1 += area;
			ptr2 += area;
		}
	}
	return result;
}
Matrix Matrix::operator =(const Matrix& m)
{
	if (data != m.data)
	{
		delete data;
		
		data = m.data;	width = m.width;	height = m.height;
		channel = m.channel;	tensor = m.tensor;
		area = m.area;	channel_area = m.channel_area;
		total = m.total;
		align_area = m.align_area;
		align_channel_area = m.align_channel_area;
		align_total = m.align_total;		
	}
}
Matrix Matrix::operator +(const float& num)
{
	return mat_operator_num(*this, num, 
			[](const __m128& ptr, const float& num, __m128& result_ptr) {result_ptr = ADD(ptr,num); },
			[](const  float& ptr, const float& num,  float& result_ptr) {result_ptr = num + ptr; }
		);
}
Matrix Matrix::operator -(const float& num)
{
	return mat_operator_num(*this, num,
		[](const __m128& ptr, const float& num, __m128& result_ptr) {result_ptr = SUB(ptr, num); },
		[](const  float& ptr, const float& num, float& result_ptr) {result_ptr = num - ptr; }
	);
}
Matrix Matrix::operator *(const float& num)
{
	return mat_operator_num(*this, num,
		[](const __m128& ptr, const float& num, __m128& result_ptr) {result_ptr = MUL(ptr, num); },
		[](const  float& ptr, const float& num, float& result_ptr) {result_ptr = num * ptr; }
	);
}
Matrix Matrix::operator /(const float& num)
{
	return mat_operator_num(*this, num,
		[](const __m128& ptr, const float& num, __m128& result_ptr) {result_ptr = DIV(ptr, num); },
		[](const  float& ptr, const float& num, float& result_ptr) {result_ptr = num / ptr; }
	);
}
inline 
Matrix mat_operator_mat(const Matrix& m1,const Matrix& m2,
					void(*_op)(const __m128& ptr,   const __m128& m_ptr, __m128& result_ptr),
					void (*op)(const  float& ptr, const    float& m_ptr,  float& result_ptr))
{
	Matrix result(m1.width, m1.height, m1.channel, m1.tensor);
	float *result_ptr = result.data;	
	const float *m1_ptr = m1.data;
	const float *m2_ptr = m2.data;
	if (m2.channel == 1 && m2.tensor == 1)
	{
		int tensor = m1.tensor;
		int align_channel_area = m1.align_channel_area;
		int channel_area = m1.channel_area;
		for (int i = 0; i < tensor; i++)
		{			
			__m128 *_m1_ptr = (__m128 *)m1_ptr;
			__m128 *_m2_ptr = (__m128 *)m2_ptr;
			__m128 *_result_ptr = (__m128 *)result_ptr;
			int j = 0;
			for (; j < align_channel_area; j+=4)
			{
				_op(*_m1_ptr, *_m2_ptr, *_result_ptr);
				_m1_ptr++; _m2_ptr++;	result_ptr++;
			}
			m1_ptr = (float *)_m1_ptr;
			m2_ptr = (float *)_m2_ptr;
			result_ptr = (float *)_result_ptr;;
			for (; j < channel_area; j++)
			{
				op(*m1_ptr, *m2_ptr, *result_ptr);
				m1_ptr++; m2_ptr++;	result_ptr++;
			}
		}
	}
	else
	{	
		int align_total = m1.align_total;
		int total = m1.total;
		__m128 *_m1_ptr = (__m128 *)m1_ptr;
		__m128 *_m2_ptr = (__m128 *)m2_ptr;
		__m128 *_result_ptr = (__m128 *)result_ptr;
		int i = 0;
		for (; i < align_total; i+=4)
		{
			_op(*_m1_ptr, *_m2_ptr, *_result_ptr);
			_m1_ptr++;	_m2_ptr++;	_result_ptr++;
		}
		m1_ptr = (float *)_m1_ptr;
		m2_ptr = (float *)_m2_ptr;
		result_ptr = (float *)_result_ptr;
		for (; i < total; i++)
		{
			op(*m1_ptr, *m2_ptr, *result_ptr);
			m1_ptr++;	m2_ptr++;	result_ptr++;
		}
	}
	return result;
}
inline 
Matrix mat_operator_num(const Matrix& m1, const float& num,
				void(*_op)(const __m128& ptr, const  float& m_ptr, __m128& result_ptr),
				void(*op) (const  float& ptr, const  float& m_ptr,  float& result_ptr))
{
	Matrix result(m1.width, m1.height, m1.channel, m1.tensor);
	int align_total = m1.align_total;
	int total = m1.total;
	float *result_ptr = result.data;
	float *m1_ptr = m1.data;
	
	__m128 *_m1_ptr = (__m128 *)m1_ptr;
	__m128 *_result_ptr=(__m128 *)result_ptr;
	int i = 0;
	for (; i < align_total; i+=4)
	{
		_op( *_m1_ptr, num, *_result_ptr);
		_m1_ptr++;		_result_ptr++;
	}
	m1_ptr = (float *)_m1_ptr;
	result_ptr = (float *)_result_ptr;
	for (; i < total; i++)
	{
		op(*m1_ptr, num, *result_ptr);
		m1_ptr++;		result_ptr++;
	}
	return result;
}
float Matrix::calcSumSquareNorm() const
{
	float sum_sqr_norm = 0;
	const float *ptr = (const float *)data;
	for (int i = 0; i<total; i++)
	{
		sum_sqr_norm += *ptr**ptr; ptr++;
	}
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

inline
float* initAcosTable() 
{
	const int n = 10000, b = 10; 
	int i;
	static float a[n * 2 + b * 2]; 
	static bool init = false;
	float *a1 = a + n + b; if (init) 
		return a1;
	for (i = -n - b; i<-n; i++)   a1[i] = CV_PI;
	for (i = -n; i<n; i++)      a1[i] = float(acos(i / float(n)));
	for (i = n; i<n + b; i++)     a1[i] = 0;
	for (i = -n - b; i<n / 10; i++) if (a1[i] > CV_PI - 1e-6f) a1[i] = CV_PI - 1e-6f;
	init = true; return a1;
}
