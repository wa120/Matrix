#include "Matrix.h"
#include "piotr_fhog/sse.hpp"
#include <math.h>

#define PADDING_SIZE 8 //4
#define NO_ALGIN 0
#define PADDING_MEMORY 1
#define MOD_SIZE 2
#define ALIGN_METHOD  PADDING_MEMORY

#define SIMD_ALIGN(x) x + PADDING_SIZE - x % PADDING_SIZE
#define SIMD_MALLOC() (float *)malloc(align_total*sizeof(float))

using namespace std;
using namespace cv;

float* circshift(const cv::Mat &patch, int x_rot, int y_rot);
inline Matrix mat_operator_mat(const Matrix& m1, const Matrix& m2,
					void(*_op)(const __m128& ptr, const __m128& m_ptr, __m128& result_ptr),
					void(*op) (const  float& ptr,  const float& m_ptr,  float& result_ptr));
inline Matrix mat_operator_num(const Matrix& m1,  const  float& num,
					void(*_op)(const __m128& ptr, const __m128& m_ptr, __m128& result_ptr),
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
	align_area(SIMD_ALIGN(area)),
	align_channel_area(SIMD_ALIGN(channel_area)),
	align_total(align_channel_area),
	data(SIMD_MALLOC())
{
	if (m.channels() == 1)
		memcpy(data, m.data, total);
	else
		setRawImage(m.data);

}
Matrix::Matrix(const Matrix& m)
	:width(m.width), height(m.height), channel(m.channel), tensor(m.tensor),
	area(m.area), channel_area(m.channel_area), total(m.total),
	align_area(SIMD_ALIGN(area)),
	align_channel_area(SIMD_ALIGN(channel_area)),
	align_total(SIMD_ALIGN(total)),
	data(m.data)
{
}

Matrix::Matrix(const int& width, const int& height)
	:width(width),height(height), channel(1),tensor(1),
	 area(width*height), channel_area(area), total(area),
	 align_area(SIMD_ALIGN(area)),
	 align_channel_area(align_area),align_total(align_area),
	 data(SIMD_MALLOC())
{
}
Matrix::Matrix(const int& width, const int& height, const int& channel)
	: width(width), height(height), channel(channel), tensor(1), 
	  area(width*height), channel_area(area*channel), total(channel_area),
	  align_area(SIMD_ALIGN(area)),
	  align_channel_area(SIMD_ALIGN(channel_area)),
	  align_total(align_channel_area),
	  data(SIMD_MALLOC())
{
}
Matrix::Matrix(const int& width, const int& height, const int& channel, const int& tensor)
	: width(width), height(height),  channel(channel), tensor(tensor),
	  area(width*height), channel_area(area*channel), total(channel_area*tensor),
	  align_area(SIMD_ALIGN(area)),
	  align_channel_area(SIMD_ALIGN(channel_area)),
	  align_total(SIMD_ALIGN(total)),
	  data(SIMD_MALLOC())
{
}
Matrix Matrix::setRawImage(const uchar *raw_data)
{
	float *ptr = data;
	const uchar *raw_data_ptr = raw_data;
	const uchar **raw_channels = (uchar **)malloc(channel * sizeof(uchar *));
	const uchar **raw_channels_ptr = raw_channels;
	float **channels = (float **)malloc(channel * sizeof(float *));
	float **channels_ptr = channels;
	int temp = area*(channel - 1);
	for (int i = 0; i < channel; i++)
	{
		*channels_ptr = ptr;
		channels_ptr++;	ptr += area;
		*raw_channels_ptr = raw_data_ptr;
		raw_channels_ptr++;	raw_data_ptr++;
	}
	for (int i = 0; i < tensor; i++)
	{
		for (int j = 0; j < area; j++)
		{
			channels_ptr = channels;
			for (int k = 0; k < channel; k++)
			{
				**channels_ptr = (float)**raw_channels_ptr;
				*channels_ptr++;	*raw_channels_ptr++;
				channels_ptr++;
			}
		}
		channels_ptr = channels;
		for (int j = 0; j < channel; j++)
		{
			*channels_ptr += temp;
			channels_ptr++;
		}
	}
	delete raw_channels;
	delete channels;
}
Matrix Matrix::setRawImage(const float *raw_data)
{
	float *ptr = data;
	const float *raw_data_ptr = raw_data;
	const float **raw_channels = (float **) malloc(channel*sizeof(float *));
	const float **raw_channels_ptr = raw_channels;
	float **channels = (float **)malloc(channel * sizeof(float *));
	float **channels_ptr = channels;
	int temp = area*(channel - 1);
	for (int i = 0; i < channel; i++)
	{
		*channels_ptr = ptr;
		channels_ptr++;	ptr += area;
		*raw_channels_ptr = raw_data_ptr;
		raw_channels_ptr++;	raw_data_ptr++;
	}
	for (int i = 0; i < tensor; i++)
	{		
		for (int j = 0; j < area; j++)
		{
			channels_ptr = channels;
			for (int k = 0; k < channel; k++)
			{
				**channels_ptr = **raw_channels_ptr;
				*channels_ptr++;	*raw_channels_ptr++;
				channels_ptr++;
			}
		}
		channels_ptr = channels;
		for (int i = 0; i < channel; i++)
		{
			*channels_ptr += temp;
			channels_ptr++;
		}
	}
	delete raw_channels;
	delete channels;
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
			for (; k < align_area; k+= PADDING_SIZE)
			{
				*_result_ptr = ADD(*_result_ptr, *_channels_ptr);
				_result_ptr++;	channels_ptr++;
#if PADDING_SIZE == 8
				* _result_ptr = ADD(*_result_ptr, *_channels_ptr);
				_result_ptr++;	channels_ptr++;
#endif
			}
#if PADDING_SIZE == 8
#else
			*channels_ptr = (float *)_channels_ptr;
			result_ptr = (float *)_result_ptr;
			for (; k < area; k++)
			{
				*result_ptr += **channels_ptr;
				result_ptr++;	*channels_ptr++;
			}
#endif
			*channels_ptr += channel_area;
			channels_ptr++;
		}
	}
	delete channels;
	return result;
}
void Matrix::gradient(Matrix& m, Matrix& o,bool fullOrientation = true)
{

	float *acost = acosTable = initAcosTable() , acMult = 10000.0f;
	// allocate memory for storing one column of output (padded so h4%4==0)
	int align_width = width - (width % 4);
	int width_1 = width - 1;
	int Gx_area = (height - 2)*width;
	int align_Gx_area = Gx_area - Gx_area % 4;
	Matrix gx(width, height, channel, tensor);
	Matrix gy(width, height, channel, tensor);
	Matrix m = Matrix(width, height, channel, tensor);
	Matrix o = Matrix(width, height, channel, tensor);

	float *I = this->data;
	float *M = m.data,
		  *Gx = gx.data,
	      *Gy = gy.data,
		  *O  = o.data; 
	__m128 *_M = (__m128 *) M,
		   *_Gx = (__m128 *) Gx,
		   *_Gy = (__m128 *) Gy,
		   *_O  = (__m128 *) O;

	// compute gradient magnitude and orientation for each column
	for (int c = 0; c < tensor; c++)
	{
#define _GRAD_OP(_G,_W1,_W3,_R) *_G++ = MUL(SUB(*_W1++,*_W3++), _R)
#define  GRAD_OP( G, W1, W3, R)	* G++ = (*W1 - *W3)*R

		//-------------------Gx---------------------//
		// Gx = I * [-1 0 1]^T

		float *w1 = I, *w3 = I + width;
		__m128 *_w1, *_w3, *_Gx, _r;

		//**begin border**/
		_Gx = (__m128 *)Gx;	_w1 = (__m128*) w1; _w3 = (__m128 *)w3;
		_r = SET(1.f);
		int i = 0;
		for (; i < align_width; i += 4)	_GRAD_OP(_Gx, _w1, _w3, _r);

		Gx = (float *)_Gx;	w1 = (float *)_w1;	w3 = (float *)_w3;
		for (; i < width; i++)			 GRAD_OP(Gx, w1, w3, 1.f);
		//**end border**/

		//**begin body**//
		w1 = I;
		_Gx = (__m128 *)Gx;	_w1 = (__m128*) w1; _w3 = (__m128 *)w3;
		_r = SET(0.5f);
		i = 0;
		for (; i < align_Gx_area; i += PADDING_SIZE)
		{		
			_GRAD_OP(_Gx, _w1, _w3, _r);
#if PADDING_SIZE == 8
			_GRAD_OP(_Gx, _w1, _w3, _r);
#endif
		}
#if PADDING_SIZE == 8
#else
		Gx = (float *)_Gx;	w1 = (float *)_w1;	w3 = (float *)_w3;
		for (; i < Gx_area; i++)		 GRAD_OP(Gx, w1, w3, 0.5);
#endif
		//**end body**//

		//**begin border**//
		w3 = w3 - width;
		_Gx = (__m128 *)Gx;	_w1 = (__m128*) w1; _w3 = (__m128 *)w3;
		_r = SET(1.f);
		i = 0;
		for (; i < align_width; i += PADDING_SIZE)
		{
			_GRAD_OP(_Gx, _w1, _w3, _r);
#if PADDING_SIZE == 8
			_GRAD_OP(_Gx, _w1, _w3, _r);
#endif
		}
#if PADDING_SIZE == 8
#else
		Gx = (float *)_Gx;	w1 = (float *)_w1;	w3 = (float *)_w3;
		for (; i < width; i++)			 GRAD_OP(Gx, w1, w3, 1.f);
#endif
		//**end border**//

//-------------------Gy---------------------//

		for (int i = 0; i < height; i++)
		{
		
			// Gy = I * [-1 0 1]
			//**begin border**//
			w1 = I;	w3 = I+1;	GRAD_OP(Gy, w1, w3, 1.f);
			//**end border**//

			//**begin body**//
			_Gx = (__m128 *)Gx;	_w1 = (__m128*) w1; _w3 = (__m128 *)w3;
			_r = SET(0.5f);
			int j = 1;
			for (; j < align_width; i += PADDING_SIZE)
			{
				_GRAD_OP(_Gy, _w1, _w3, _r);
#if PADDING_SIZE == 8
				_GRAD_OP(_Gy, _w1, _w3, _r);
#endif
			}
#if PADDING_SIZE == 8
#else
			Gy = (float *)_Gy;	w1 = (float *)_w1;	w3 = (float *)_w3;
			for (; j < width_1; j++)		 GRAD_OP(Gy, w1, w3, 1.0);
#endif
			//**end body**//

			//**begin border**//
			w1++;				GRAD_OP(Gy, w1, w3, 1.f);
			//**end border**//
		}
#undef _GRAD_OP
#undef  GRAD_OP

/*--------------*/
		_Gx = (__m128*) gx.data,
		_Gy = (__m128*) gy.data;
		int i = 0;
		for (; i < align_area; i+=PADDING_SIZE)
		{
			__m128 _m = _MIN(RCPSQRT(ADD(MUL(*_Gx, *_Gx), MUL(*_Gy, *_Gy))), SET(1e10f));
			*_M = RCP(_m);
			*_O = XOR(MUL(MUL(*_Gx, _m), SET(10000.0f)), AND(*_Gy, SET(-0.f)));
			_M++;	_Gx++;	_Gy++;	_O++;
#if PADDING_SIZE == 8
			__m128 _m = _MIN(RCPSQRT(ADD(MUL(*_Gx, *_Gx), MUL(*_Gy, *_Gy))), SET(1e10f));
			*_M = RCP(_m);
			*_O = XOR(MUL(MUL(*_Gx, _m), SET(10000.0f)), AND(*_Gy, SET(-0.f)));
			_M++;	_Gx++;	_Gy++;	_O++;
#endif
		}
#if PADDING_SIZE == 8
#else
		M = (float *)_M;
		Gx = (float *)_Gx;
		Gy = (float *)_Gy;
		for (; i < area; i++)
		{
			*M = *Gx**Gx + *Gy**Gy;
			M++;	Gx++;	Gy++;
		}
#endif
		O = o.data;
		if (fullOrientation)
		{
			for (int i = 0; i < area; i++)
			{
				*O = acost[(int)*O];
			}
			i = 0;
			_O = (__m128 *)o.data;
			_Gy = (__m128 *)m.data;
			for (; i < align_area; i+= PADDING_SIZE)
			{
				*_O = ADD(*_O, AND(CMPLT(*_Gy, SET(0.f)), SET((float)CV_PI)));
				_O++;	_Gy++;
#if PADDING_SIZE == 8
				* _O = ADD(*_O, AND(CMPLT(*_Gy, SET(0.f)), SET((float)CV_PI)));
#endif
			}
		}
		else
		{
			for (int i = 0; i < area; i++)
			{
				*O = acost[(int)*O];
				*O++;
			}			
		}

	}

}
Matrix Matrix::fhog()
{
	// d image dimension -> gray image d = 1
	// h, w -> height, width of image
	// full -> ??
	// I -> input image, M, O -> mag, orientation OUTPUT
	int h = img.rows, w = img.cols, d = 1;
	bool full = true;
	if (height < 2 || width < 2)
	{
		cout << "I must be at least 2x2." << endl;
		return Matrix();
	}
	Matrix result(height, width);
	Matrix M,O;
	Matrix H(height / 4, width / 4, 9, 4);
	(transpose() / 255).gradient(M, O);

//	fhog(M, O, H, h, w, bin_size, n_orients, soft_bin, clip);
/********************/
	Matrix R1(H.width, H.height,9,2), R2(H.width, H.height,9);
	const int hb = h / binSize, wb = w / binSize, nb = hb*wb, nbo = nb*nOrients;
	float *N, *R1, *R2; int o, x;
	// compute unnormalized constrast sensitive histograms
	R1 = (float*)calloc(wb*hb*nOrients * 2, sizeof(float));
//	gradHist(M, O, R1, h, w, binSize, nOrients * 2, softBin, true);
/*------------------*/
	const int hb = h / bin, wb = w / bin, h0 = hb*bin, w0 = wb*bin, nb = wb*hb;
	const float s = (float)bin, sInv = 1 / s, sInv2 = 1 / s / s;
	float *H0, *H1; 
	int x, y; 
	int *O0 = (int *)malloc(height*sizeof(int)), 
		*O1 = (int *)malloc(height * sizeof(int));
	float *M0 = (float *)malloc(height * sizeof(float)),
 		  *M1 = (float *)malloc(height * sizeof(float));
	float xb, init;
	
	// main loop
	for (x = 0; x<width; x++) {
		// compute target orientation bins for entire column - very fast
//		gradQuantize(O + x*h, M + x*h, O0, O1, M0, M1, nb, h0, sInv2, nOrients, full, softBin >= 0);
/*............................*/
// assumes all *OUTPUT* matrices are 4-byte aligned
		int i, o0, o1; float o, od, m;
		__m128i _o0, _o1, *_O0, *_O1; 
		__m128 _o, _od, _m, *_M0, *_M1;
		// define useful constants
		const float oMult = (float)9 /  2 * CV_PI ;
		const int oMax = H.align_channel_area;
		const __m128 _norm = SET(norm), _oMult = SET(oMult), _nbf = SET((float)nb);
		const __m128i _oMax = SET(oMax), _nb = SET(nb);
		// perform the majority of the work with sse
		_O0 = (__m128i*) O0; _O1 = (__m128i*) O1; _M0 = (__m128*) M0; _M1 = (__m128*) M1;
		if (interpolate) for (i = 0; i <= n - 4; i += 4) {
			_o = MUL(LDu(O[i]), _oMult); _o0 = CVT(_o); _od = SUB(_o, CVT(_o0));
			_o0 = CVT(MUL(CVT(_o0), _nbf)); _o0 = AND(CMPGT(_oMax, _o0), _o0); *_O0++ = _o0;
			_o1 = ADD(_o0, _nb); _o1 = AND(CMPGT(_oMax, _o1), _o1); *_O1++ = _o1;
			_m = MUL(LDu(M[i]), _norm); *_M1 = MUL(_od, _m); *_M0++ = SUB(_m, *_M1); _M1++;
		}
		else for (i = 0; i <= n - 4; i += 4) {
			_o = MUL(LDu(O[i]), _oMult); _o0 = CVT(ADD(_o, SET(.5f)));
			_o0 = CVT(MUL(CVT(_o0), _nbf)); _o0 = AND(CMPGT(_oMax, _o0), _o0); *_O0++ = _o0;
			*_M0++ = MUL(LDu(M[i]), _norm); *_M1++ = SET(0.f); *_O1++ = SET(0);
		}
		// compute trailing locations without sse
		if (interpolate) for (; i<n; i++) {
			o = O[i] * oMult; o0 = (int)o; od = o - o0;
			o0 *= nb; if (o0 >= oMax) o0 = 0; O0[i] = o0;
			o1 = o0 + nb; if (o1 == oMax) o1 = 0; O1[i] = o1;
			m = M[i] * norm; M1[i] = od*m; M0[i] = m - M1[i];
		}
		else for (; i<n; i++) {
			o = O[i] * oMult; o0 = (int)(o + .5f);
			o0 *= nb; if (o0 >= oMax) o0 = 0; O0[i] = o0;
			M0[i] = M[i] * norm; M1[i] = 0; O1[i] = 0;
		}

/*............................*/

		if (softBin<0 && softBin % 2 == 0) {
			// no interpolation w.r.t. either orienation or spatial bin
			H1 = H + (x / bin)*hb;
#define GH H1[O0[y]]+=M0[y]; y++;
			if (bin == 1)      for (y = 0; y<h0;) { GH; H1++; }
			else if (bin == 2) for (y = 0; y<h0;) { GH; GH; H1++; }
			else if (bin == 3) for (y = 0; y<h0;) { GH; GH; GH; H1++; }
			else if (bin == 4) for (y = 0; y<h0;) { GH; GH; GH; GH; H1++; }
			else for (y = 0; y<h0;) { for (int y1 = 0; y1<bin; y1++) { GH; } H1++; }
#undef GH

		}
		else if (softBin % 2 == 0 || bin == 1) {
			// interpolate w.r.t. orientation only, not spatial bin
			H1 = H + (x / bin)*hb;
#define GH H1[O0[y]]+=M0[y]; H1[O1[y]]+=M1[y]; y++;
			if (bin == 1)      for (y = 0; y<h0;) { GH; H1++; }
			else if (bin == 2) for (y = 0; y<h0;) { GH; GH; H1++; }
			else if (bin == 3) for (y = 0; y<h0;) { GH; GH; GH; H1++; }
			else if (bin == 4) for (y = 0; y<h0;) { GH; GH; GH; GH; H1++; }
			else for (y = 0; y<h0;) { for (int y1 = 0; y1<bin; y1++) { GH; } H1++; }
#undef GH

		}
		else {
			// interpolate using trilinear interpolation
			float ms[4], xyd, yb, xd, yd; __m128 _m, _m0, _m1;
			bool hasLf, hasRt; int xb0, yb0;
			if (x == 0) { init = (0 + .5f)*sInv - 0.5f; xb = init; }
			hasLf = xb >= 0; xb0 = hasLf ? (int)xb : -1; hasRt = xb0 < wb - 1;
			xd = xb - xb0; xb += sInv; yb = init; y = 0;
			// macros for code conciseness
#define GHinit yd=yb-yb0; yb+=sInv; H0=H+xb0*hb+yb0; xyd=xd*yd; \
        ms[0]=1-xd-yd+xyd; ms[1]=yd-xyd; ms[2]=xd-xyd; ms[3]=xyd;
#define GH(H,ma,mb) H1=H; STRu(*H1,ADD(LDu(*H1),MUL(ma,mb)));
			// leading rows, no top bin
			for (; y<bin / 2; y++) {
				yb0 = -1; GHinit;
				if (hasLf) { H0[O0[y] + 1] += ms[1] * M0[y]; H0[O1[y] + 1] += ms[1] * M1[y]; }
				if (hasRt) { H0[O0[y] + hb + 1] += ms[3] * M0[y]; H0[O1[y] + hb + 1] += ms[3] * M1[y]; }
			}
			// main rows, has top and bottom bins, use SSE for minor speedup
			if (softBin<0) for (; ; y++) {
				yb0 = (int)yb; if (yb0 >= hb - 1) break; GHinit; _m0 = SET(M0[y]);
				if (hasLf) { _m = SET(0, 0, ms[1], ms[0]); GH(H0 + O0[y], _m, _m0); }
				if (hasRt) { _m = SET(0, 0, ms[3], ms[2]); GH(H0 + O0[y] + hb, _m, _m0); }
			}
			else for (; ; y++) {
				yb0 = (int)yb; if (yb0 >= hb - 1) break; GHinit;
				_m0 = SET(M0[y]); _m1 = SET(M1[y]);
				if (hasLf) {
					_m = SET(0, 0, ms[1], ms[0]);
					GH(H0 + O0[y], _m, _m0); GH(H0 + O1[y], _m, _m1);
				}
				if (hasRt) {
					_m = SET(0, 0, ms[3], ms[2]);
					GH(H0 + O0[y] + hb, _m, _m0); GH(H0 + O1[y] + hb, _m, _m1);
				}
			}
			// final rows, no bottom bin
			for (; y<h0; y++) {
				yb0 = (int)yb; GHinit;
				if (hasLf) { H0[O0[y]] += ms[0] * M0[y]; H0[O1[y]] += ms[0] * M1[y]; }
				if (hasRt) { H0[O0[y] + hb] += ms[2] * M0[y]; H0[O1[y] + hb] += ms[2] * M1[y]; }
			}
#undef GHinit
#undef GH
		}
	}
	alFree(O0); alFree(O1); alFree(M0); alFree(M1);
	// normalize boundary bins which only get 7/8 of weight of interior bins
	if (softBin % 2 != 0) for (int o = 0; o<nOrients; o++) {
		x = 0; for (y = 0; y<hb; y++) H[o*nb + x*hb + y] *= 8.f / 7.f;
		y = 0; for (x = 0; x<wb; x++) H[o*nb + x*hb + y] *= 8.f / 7.f;
		x = wb - 1; for (y = 0; y<hb; y++) H[o*nb + x*hb + y] *= 8.f / 7.f;
		y = hb - 1; for (x = 0; x<wb; x++) H[o*nb + x*hb + y] *= 8.f / 7.f;
	}
/*------------------*/
	// compute unnormalized contrast insensitive histograms
	R2 = (float*)calloc(wb*hb*nOrients, sizeof(float));
	for (o = 0; o<nOrients; o++) for (x = 0; x<nb; x++)
		R2[o*nb + x] = R1[o*nb + x] + R1[(o + nOrients)*nb + x];
	// compute block normalization values
	N = hogNormMatrix(R2, nOrients, hb, wb, binSize);
	// normalized histograms and texture channels
	hogChannels(H + nbo * 0, R1, N, hb, wb, nOrients * 2, clip, 1);
	hogChannels(H + nbo * 2, R2, N, hb, wb, nOrients * 1, clip, 1);
	hogChannels(H + nbo * 3, R1, N, hb, wb, nOrients * 2, clip, 2);
	free(N); free(R1); free(R2);
/********************/

	//convert, assuming row-by-row-by-channel storage
	std::vector<cv::Mat> res;
	int n_res_channels = (use_hog == 2) ? n_chns - 1 : n_chns;    //last channel all zeros for fhog
	res.reserve(n_res_channels);
	for (int i = 0; i < n_res_channels; ++i) {
		//output rows-by-rows
		//            cv::Mat desc(hb, wb, CV_32F, (H+hb*wb*i));

		//output cols-by-cols
		cv::Mat desc(hb, wb, CV_32F);
		for (int x = 0; x < wb; ++x) {
			for (int y = 0; y < hb; ++y) {
				desc.at<float>(y, x) = H[i*hb*wb + x*hb + y];
			}
		}

		res.push_back(desc.clone());
	}

	//clean
	delete[] I;
	delete[] M;
	delete[] O;
	delete[] H;

	return res;
}
Matrix Matrix::transpose()
{
	Matrix result(height, width, channel, tensor);
	int tensor_channel = tensor*channel;
	int shift_ptr = area - width;
	float *result_ptr = result.data;
	float *ptr = data;
	float **height_arr = (float **)malloc(height * sizeof(float *));
	float **height_arr_ptr = height_arr;
	for (int i = 0; i < height; i++)
	{
		*height_arr_ptr = ptr;
		height_arr_ptr++;	ptr += width;
	}
	for (int i = 0; i < tensor_channel; i++)
	{	
		for (int j = 0; j < width; j++)
		{
			height_arr_ptr = height_arr;
			for (int k = 0; k < height; k++)
			{
				*result_ptr = **height_arr_ptr;
				*height_arr_ptr++;
				height_arr_ptr++;	result_ptr++;				
			}	
		}
		height_arr_ptr = height_arr;
		for (int j = 0; j < height; j++)
		{
			*height_arr_ptr += shift_ptr;
			height_arr_ptr++;
		}
	}
	delete height_arr;
	return result;
}
Matrix Matrix::clone()
{
	return mat_operator_num(*this,0.f,
			[](const __m128& ptr, const  __m128& m_ptr, __m128& result_ptr) {result_ptr = ptr; },
			[](const  float& ptr, const  float& m_ptr, float& result_ptr) {result_ptr = ptr; }
		);	
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
				for (; l < align_w; l+=PADDING_SIZE)
				{
					*_result_ptr = *_channels_ptr;
					_result_ptr++;	_channels_ptr++;
#if PADDING_SIZE == 8
					* _result_ptr = *_channels_ptr;
					_result_ptr++;	_channels_ptr++;
#endif
				}
#if PADDING_SIZE == 8
#else
				*channels_ptr = (float *)_channels_ptr;
				result_ptr = (float *)_result_ptr;
				for (;l < w; l++)
				{
					*result_ptr = **channels_ptr;
					result_ptr ++ ;	*channels_ptr++;
				}
#endif
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
			for (;j < align_area; j+=PADDING_SIZE)
			{
				*_result_ptr = SQRT(ADD(MUL(*_ptr1, *_ptr1), MUL(*_ptr2, *_ptr2)));
				_result_ptr++;	_ptr1++;	_ptr2++;
#if PADDING_SIZE == 8
				* _result_ptr = SQRT(ADD(MUL(*_ptr1, *_ptr1), MUL(*_ptr2, *_ptr2)));
				_result_ptr++;	_ptr1++;	_ptr2++;
#endif
			}
#if PADDING_SIZE == 8
#else
			result_ptr = (float *)_result_ptr;
			ptr1 = (float *)_ptr1;
			ptr2 = (float *)_ptr2;
			for (; j < area; j++)
			{
				*result_ptr = sqrt((*ptr1**ptr1) + (*ptr2**ptr2));
				result_ptr++;	ptr1++;	ptr2++;
			}
#endif
	
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
			for (; j < align_area; j+=PADDING_SIZE)
			{
				*_result_ptr = ATAN(DIV(*_ptr2, *_ptr1));
				_result_ptr++;	_ptr1++;	_ptr2++;
#if PADDING_SIZE == 8
				* _result_ptr = ATAN(DIV(*_ptr2, *_ptr1));
				_result_ptr++;	_ptr1++;	_ptr2++;
#endif
			}
#if PADDING_SIZE == 8
#else
			result_ptr = (float *)_result_ptr;
			ptr1 = (float *)_ptr1;
			ptr2 = (float *)_ptr2;
			for (; j < area; j++)
			{
				*result_ptr = atan(*ptr2/ *ptr1);
				result_ptr++;	ptr1++;	ptr2++;
			}
#endif
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
			for (; j < align_area; j+=PADDING_SIZE)
			{
				*_result_ptr1 = *_ptr1;
				*_result_ptr2 = NEG(*_ptr2);
				_result_ptr1++;	_ptr1++;
				_result_ptr2++;	_ptr2++;
#if PADDING_SIZE == 8
				* _result_ptr1 = *_ptr1;
				*_result_ptr2 = NEG(*_ptr2);
				_result_ptr1++;	_ptr1++;
				_result_ptr2++;	_ptr2++;
#endif
			}
#if PADDING_SIZE == 8
#else
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
#endif
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
			[](const __m128& ptr, const __m128& num, __m128& result_ptr) {result_ptr = ADD(ptr,num); },
			[](const  float& ptr, const  float& num,  float& result_ptr) {result_ptr = num + ptr; }
		);
}
Matrix Matrix::operator -(const float& num)
{
	return mat_operator_num(*this, num,
		[](const __m128& ptr, const __m128& num, __m128& result_ptr) {result_ptr = SUB(ptr, num); },
		[](const  float& ptr, const  float& num, float& result_ptr) {result_ptr = num - ptr; }
	);
}
Matrix Matrix::operator *(const float& num)
{
	return mat_operator_num(*this, num,
		[](const __m128& ptr, const __m128& num, __m128& result_ptr) {result_ptr = MUL(ptr, num); },
		[](const  float& ptr, const  float& num, float& result_ptr) {result_ptr = num * ptr; }
	);
}
Matrix Matrix::operator /(const float& num)
{
	return mat_operator_num(*this, num,
		[](const __m128& ptr, const __m128& num, __m128& result_ptr) {result_ptr = DIV(ptr, num); },
		[](const  float& ptr, const  float& num, float& result_ptr) {result_ptr = num / ptr; }
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
			for (; j < align_channel_area; j+=PADDING_SIZE)
			{
				_op(*_m1_ptr, *_m2_ptr, *_result_ptr);
				_m1_ptr++; _m2_ptr++;	result_ptr++;
#if PADDING_SIZE == 8
				_op(*_m1_ptr, *_m2_ptr, *_result_ptr);
				_m1_ptr++; _m2_ptr++;	result_ptr++;
#endif
			}
#if PADDING_SIZE == 8
#else
			m1_ptr = (float *)_m1_ptr;
			m2_ptr = (float *)_m2_ptr;
			result_ptr = (float *)_result_ptr;;
			for (; j < channel_area; j++)
			{
				op(*m1_ptr, *m2_ptr, *result_ptr);
				m1_ptr++; m2_ptr++;	result_ptr++;
			}
#endif
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
		for (; i < align_total; i+=PADDING_SIZE)
		{
			_op(*_m1_ptr, *_m2_ptr, *_result_ptr);
			_m1_ptr++;	_m2_ptr++;	_result_ptr++;
#if PADDING_SIZE == 8
			_op(*_m1_ptr, *_m2_ptr, *_result_ptr);
			_m1_ptr++;	_m2_ptr++;	_result_ptr++;
#endif
		}
#if PADDING_SIZE == 8
#else
		m1_ptr = (float *)_m1_ptr;
		m2_ptr = (float *)_m2_ptr;
		result_ptr = (float *)_result_ptr;
		for (; i < total; i++)
		{
			op(*m1_ptr, *m2_ptr, *result_ptr);
			m1_ptr++;	m2_ptr++;	result_ptr++;
		}
#endif
	}
	return result;
}
inline 
Matrix mat_operator_num(const Matrix& m1, const float& num,
				void(*_op)(const __m128& ptr, const __m128& m_ptr, __m128& result_ptr),
				void(*op) (const  float& ptr, const  float& m_ptr,  float& result_ptr))
{
	Matrix result(m1.width, m1.height, m1.channel, m1.tensor);
	int align_total = m1.align_total;
	int total = m1.total;
	float *result_ptr = result.data;
	float *m1_ptr = m1.data;
	
	__m128 *_m1_ptr = (__m128 *)m1_ptr;
	__m128 *_result_ptr=(__m128 *)result_ptr;
	__m128 _num = SET(num);
	int i = 0;
	for (; i < align_total; i+=PADDING_SIZE)
	{
		_op( *_m1_ptr, _num, *_result_ptr);
		_m1_ptr++;		_result_ptr++;
#if PADDING_SIZE == 8
		_op(*_m1_ptr, _num, *_result_ptr);
		_m1_ptr++;		_result_ptr++;
#endif
	}
#if PADDING_SIZE == 8
#else
	m1_ptr = (float *)_m1_ptr;
	result_ptr = (float *)_result_ptr;
	for (; i < total; i++)
	{
		op(*m1_ptr, num, *result_ptr);
		m1_ptr++;		result_ptr++;
	}
#endif
	return result;
}
float Matrix::calcSumSquareNorm() const
{
	float result = 0;
	const float *ptr = (const float *)data;
	__m128 _result;
	__m128 *_ptr = (__m128 *) ptr;
	int i = 0;
	for (; i<align_total; i+=PADDING_SIZE)
	{
		_result = ADD(MUL(*_ptr,*_ptr),_result); 
		_ptr++;
#if PADDING_SIZE == 8
		_result = ADD(MUL(*_ptr, *_ptr), _result);
		_ptr++;
#endif
	}
	ptr = (float *)_ptr;
	float* result_ptr = (float *)&_result;
	
	for (int i = 0; i < 4; i++)
	{
		result += *result_ptr;	result_ptr++;
	}	
#if PADDING_SIZE == 8
#else
	for (; i < total; i++)
	{
		result += *ptr;
		ptr++;
	}
#endif
	return result / area;
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
