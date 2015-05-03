#ifndef MDTRAJ_VECTORIZE_H_
#define MDTRAJ_VECTORIZE_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2013-2015 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors: Robert T. McGibbon                                           *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "msvccompat.h"
#include <pmmintrin.h>
#include <cstdio>

// This file defines classes and functions to simplify vectorizing code with SSE.

#ifndef __SSE4_1__

/* Macros from http://sseplus.sourceforge.net/_s_s_e_plus__platform_8h-source.html */
#ifdef _MSC_VER
#define __CNST32I28I_( x ) \
    ((unsigned __int8)((x) & 0xFF)), ((unsigned __int8)(((x) >> 8) & 0xFF)), ((unsigned __int8)(((x) >> 16) & 0xFF)), ((unsigned __int8)(((x) >> 24) & 0xFF))

#define SSP_CONST_SETR_32I( a, b, c, d ) \
    { __CNST32I28I_((a)), __CNST32I28I_((b)), __CNST32I28I_((c)), __CNST32I28I_((d)) }

#define SSP_CONST_SET_32I( a, b, c, d ) \
    SSP_CONST_SETR_32I( (d), (c), (b), (a) )
#else
#define __CNST32TO64_( a, b ) \
        ( ((b)<<32) | ((a) & 0xFFFFFFFF) )

#define SSP_CONST_SETR_32I( a, b, c, d ) \
    { __CNST32TO64_( (unsigned long long)(a), (unsigned long long)(b) ), \
      __CNST32TO64_( (unsigned long long)(c), (unsigned long long)(d) ) }

#define SSP_CONST_SET_32I( a, b, c, d ) \
    SSP_CONST_SETR_32I( (d), (c), (b), (a) )
#endif

static INLINE __m128 _mm_hsum_ps(__m128 v) {
    v = _mm_hadd_ps(v, v);
    v = _mm_hadd_ps(v, v);
    return v;
}

static INLINE __m128i ssp_logical_bitwise_select_SSE2( __m128i a, __m128i b, __m128i mask )   // Bitwise (mask ? a : b)
{
    a = _mm_and_si128(a, mask);       // clear a where mask = 0
    b = _mm_andnot_si128(mask, b);    // clear b where mask = 1
    a = _mm_or_si128(a, b);           // a = a OR b
    return a;
}


static INLINE __m128i _mm_min_epi32(__m128i a, __m128i b) {
    __m128i mask  = _mm_cmplt_epi32( a, b );  // FFFFFFFF where a < b
    a = ssp_logical_bitwise_select_SSE2( a, b, mask );
    return a;
}

static INLINE __m128i _mm_max_epi32(__m128i a, __m128i b) {
    __m128i mask  = _mm_cmpgt_epi32( a, b );  // FFFFFFFF where a > b
    a = ssp_logical_bitwise_select_SSE2( a, b, mask );
    return a;
}

static INLINE __m128i _mm_abs_epi32(__m128i a) {
    // FFFF   where a < 0
    __m128i mask = _mm_cmplt_epi32( a, _mm_setzero_si128() );
    a    = _mm_xor_si128 ( a, mask );   // Invert where a < 0
    mask = _mm_srli_epi32( mask, 31 );  // 0001   where a < 0
    a = _mm_add_epi32( a, mask );       // Add 1  where a < 0
    return a;
}

static INLINE __m128 _mm_dp_ps( __m128 a, __m128 b, const int mask ) {
    /*
     Copyright (c) 2006-2008 Advanced Micro Devices, Inc. All Rights Reserved.
     This software is subject to the Apache v2.0 License.
    */

    /* Shift mask multiply moves 0,1,2,3 bits to left, becomes MSB */
    const static __m128i mulShiftImm_0123 = SSP_CONST_SET_32I(0x010000, 0x020000, 0x040000, 0x080000);
    /* Shift mask multiply moves 4,5,6,7 bits to left, becomes MSB */
    const static __m128i mulShiftImm_4567 = SSP_CONST_SET_32I(0x100000, 0x200000, 0x400000, 0x800000);

    /* Begin mask preparation */
    __m128i mHi, mLo;
    mLo = _mm_set1_epi32(mask);    /* Load the mask into register */
    mLo = _mm_slli_si128(mLo, 3);  /* Shift into reach of the 16 bit multiply */
    mHi = _mm_mullo_epi16(mLo, mulShiftImm_0123);  /* Shift the bits */
    mLo = _mm_mullo_epi16(mLo, mulShiftImm_4567);  /* Shift the bits */
    mHi = _mm_cmplt_epi32(mHi, _mm_setzero_si128()); /* FFFFFFFF if bit set, 00000000 if not set */
    mLo = _mm_cmplt_epi32(mLo, _mm_setzero_si128()); /* FFFFFFFF if bit set, 00000000 if not set */
    /* End mask preparation - Mask bits 0-3 in mLo, 4-7 in mHi */
    a = _mm_and_ps(a, *(__m128*)& mHi);   /* Clear input using the high bits of the mask */
    a = _mm_mul_ps(a, b);
    a = _mm_hsum_ps(a);                  /* Horizontally add the 4 values */
    a = _mm_and_ps(a, *(__m128*)& mLo);  /* Clear output using low bits of the mask */
    return a;
}

#endif


class ivec4;

/**
 * A four element vector of floats.
 */
class fvec4 {
public:
    __m128 val;

    fvec4() {}
    fvec4(float v) : val(_mm_set1_ps(v)) {}
    fvec4(float v1, float v2, float v3, float v4) : val(_mm_set_ps(v4, v3, v2, v1)) {}
    fvec4(__m128 v) : val(v) {}
    fvec4(const float* v) : val(_mm_loadu_ps(v)) {}
    operator __m128() const {
        return val;
    }
    float operator[](int i) const {
        float result[4];
        store(result);
        return result[i];
    }
    void store(float* v) const {
        _mm_storeu_ps(v, val);
    }
    void print(void) const {
        float result[4];
        store(result);
        printf("%f  %f  %f  %f\n", result[0], result[1], result[2], result[3]);
    }
    fvec4 operator+(const fvec4& other) const {
        return _mm_add_ps(val, other);
    }
    fvec4 operator-(const fvec4& other) const {
        return _mm_sub_ps(val, other);
    }
    fvec4 operator*(const fvec4& other) const {
        return _mm_mul_ps(val, other);
    }
    fvec4 operator/(const fvec4& other) const {
        return _mm_div_ps(val, other);
    }
    void operator+=(const fvec4& other) {
        val = _mm_add_ps(val, other);
    }
    void operator-=(const fvec4& other) {
        val = _mm_sub_ps(val, other);
    }
    void operator*=(const fvec4& other) {
        val = _mm_mul_ps(val, other);
    }
    void operator/=(const fvec4& other) {
        val = _mm_div_ps(val, other);
    }
    fvec4 operator-() const {
        return _mm_sub_ps(_mm_set1_ps(0.0f), val);
    }
    fvec4 operator&(const fvec4& other) const {
        return _mm_and_ps(val, other);
    }
    fvec4 operator|(const fvec4& other) const {
        return _mm_or_ps(val, other);
    }
    fvec4 operator==(const fvec4& other) const {
        return _mm_cmpeq_ps(val, other);
    }
    fvec4 operator!=(const fvec4& other) const {
        return _mm_cmpneq_ps(val, other);
    }
    fvec4 operator>(const fvec4& other) const {
        return _mm_cmpgt_ps(val, other);
    }
    fvec4 operator<(const fvec4& other) const {
        return _mm_cmplt_ps(val, other);
    }
    fvec4 operator>=(const fvec4& other) const {
        return _mm_cmpge_ps(val, other);
    }
    fvec4 operator<=(const fvec4& other) const {
        return _mm_cmple_ps(val, other);
    }
    operator ivec4() const;
};


/**
 * A four element vector of ints.
 */
class ivec4 {
public:
    __m128i val;

    ivec4() {}
    ivec4(int v) : val(_mm_set1_epi32(v)) {}
    ivec4(int v1, int v2, int v3, int v4) : val(_mm_set_epi32(v4, v3, v2, v1)) {}
    ivec4(__m128i v) : val(v) {}
    ivec4(const int* v) : val(_mm_loadu_si128((const __m128i*) v)) {}
    operator __m128i() const {
        return val;
    }
    int operator[](int i) const {
        int result[4];
        store(result);
        return result[i];
    }
    void print(void) const {
        int result[4];
        store(result);
        printf("%d  %d  %d  %d\n", result[0], result[1], result[2], result[3]);
    }
    void store(int* v) const {
        _mm_storeu_si128((__m128i*) v, val);
    }
    ivec4 operator+(const ivec4& other) const {
        return _mm_add_epi32(val, other);
    }
    ivec4 operator-(const ivec4& other) const {
        return _mm_sub_epi32(val, other);
    }
    ivec4 operator*(const ivec4& other) const {
        /* http://stackoverflow.com/a/10501533 */
#ifdef __SSE4_1__  // modern CPU - use SSE 4.1
        return _mm_mullo_epi32(val, other);
#else               // old CPU - use SSE 2
        __m128i tmp1 = _mm_mul_epu32(val, other); /* mul 2,0*/
        __m128i tmp2 = _mm_mul_epu32( _mm_srli_si128(val, 4), _mm_srli_si128(val ,4)); /* mul 3,1 */
        return _mm_unpacklo_epi32(_mm_shuffle_epi32(tmp1, _MM_SHUFFLE (0,0,2,0)), _mm_shuffle_epi32(tmp2, _MM_SHUFFLE (0,0,2,0))); /* shuffle results to [63..0] and pack */
#endif
    }
    void operator+=(const ivec4& other) {
        val = _mm_add_epi32(val, other);
    }
    void operator-=(const ivec4& other) {
        val = _mm_sub_epi32(val, other);
    }
    void operator*=(const ivec4& other) {
        /* http://stackoverflow.com/a/10501533 */
#ifdef __SSE4_1__  // modern CPU - use SSE 4.1
        val = _mm_mullo_epi32(val, other);
#else               // old CPU - use SSE 2
        __m128i tmp1 = _mm_mul_epu32(val, other); /* mul 2,0*/
        __m128i tmp2 = _mm_mul_epu32( _mm_srli_si128(val, 4), _mm_srli_si128(val ,4)); /* mul 3,1 */
        val = _mm_unpacklo_epi32(_mm_shuffle_epi32(tmp1, _MM_SHUFFLE (0,0,2,0)), _mm_shuffle_epi32(tmp2, _MM_SHUFFLE (0,0,2,0))); /* shuffle results to [63..0] and pack */
#endif
    }
    ivec4 operator-() const {
        return _mm_sub_epi32(_mm_set1_epi32(0), val);
    }
    ivec4 operator&(const ivec4& other) const {
        return _mm_and_si128(val, other);
    }
    ivec4 operator|(const ivec4& other) const {
        return _mm_or_si128(val, other);
    }
    ivec4 operator==(const ivec4& other) const {
        return _mm_cmpeq_epi32(val, other);
    }
    ivec4 operator!=(const ivec4& other) const {
        return _mm_xor_si128(*this==other, _mm_set1_epi32(0xFFFFFFFF));
    }
    ivec4 operator>(const ivec4& other) const {
        return _mm_cmpgt_epi32(val, other);
    }
    ivec4 operator<(const ivec4& other) const {
        return _mm_cmplt_epi32(val, other);
    }
    ivec4 operator>=(const ivec4& other) const {
        return _mm_xor_si128(_mm_cmplt_epi32(val, other), _mm_set1_epi32(0xFFFFFFFF));
    }
    ivec4 operator<=(const ivec4& other) const {
        return _mm_xor_si128(_mm_cmpgt_epi32(val, other), _mm_set1_epi32(0xFFFFFFFF));
    }
    operator fvec4() const;
};

// Conversion operators.

INLINE fvec4::operator ivec4() const {
    return _mm_cvttps_epi32(val);
}

INLINE ivec4::operator fvec4() const {
    return _mm_cvtepi32_ps(val);
}

// Functions that operate on fvec4s.

static INLINE fvec4 load3(const float* ptr) {
    /* Load (x,y,z) into a SSE register, leaving the last entry */
    /* set to zero. */
    __m128 x = _mm_load_ss(&ptr[0]);
    __m128 y = _mm_load_ss(&ptr[1]);
    __m128 z = _mm_load_ss(&ptr[2]);
    __m128 xy = _mm_movelh_ps(x, y);
    return _mm_shuffle_ps(xy, z, _MM_SHUFFLE(2, 0, 2, 0));
}


static INLINE fvec4 floor(const fvec4& v) {
#ifdef __SSE4_1__
    return fvec4(_mm_floor_ps(v.val));
#else
    /* http://dss.stephanierct.com/DevBlog/?p=8 */
    __m128i v0 = _mm_setzero_si128();
    __m128i v1 = _mm_cmpeq_epi32(v0,v0);
    __m128i ji = _mm_srli_epi32( v1, 25);
    __m128 slli = _mm_slli_epi32( ji, 23); //create vector 1.0f
    __m128 j = *(__m128*)& slli;
    __m128i i = _mm_cvttps_epi32(v);
    __m128 fi = _mm_cvtepi32_ps(i);
    __m128 igx = _mm_cmpgt_ps(fi, v);
    j = _mm_and_ps(igx, j);
    return fvec4(_mm_sub_ps(fi, j));
#endif
}

static INLINE fvec4 ceil(const fvec4& v) {
#ifdef __SSE4_1__
    return fvec4(_mm_ceil_ps(v.val));
#else
    __m128i v0 = _mm_setzero_si128();
    __m128i v1 = _mm_cmpeq_epi32(v0,v0);
    __m128i ji = _mm_srli_epi32( v1, 25);
    __m128i ssli = _mm_slli_epi32( ji, 23);
    __m128 j = *(__m128*)& ssli; //create vector 1.0f
    __m128i i = _mm_cvttps_epi32(v);
    __m128 fi = _mm_cvtepi32_ps(i);
    __m128 igx = _mm_cmplt_ps(fi, v);
    j = _mm_and_ps(igx, j);
    return fvec4(_mm_add_ps(fi, j));
#endif
}

static INLINE fvec4 round(const fvec4& v) {
#ifdef __SSE4_1__
    return fvec4(_mm_round_ps(v.val, _MM_FROUND_TO_NEAREST_INT));
#else
    /* http://dss.stephanierct.com/DevBlog/?p=8 */
    __m128 v0 = _mm_setzero_ps();             /* generate the highest value < 2 */
    __m128 v1 = _mm_cmpeq_ps(v0,v0);
    __m128i srli = _mm_srli_epi32( *(__m128i*)& v1, 2);
    __m128 vNearest2 = *(__m128*)& srli;
    __m128i i = _mm_cvttps_epi32(v);
    __m128 aTrunc = _mm_cvtepi32_ps(i);        /* truncate a */
    __m128 rmd = _mm_sub_ps(v, aTrunc);        /* get remainder */
    __m128 rmd2 = _mm_mul_ps( rmd, vNearest2); /* mul remainder by near 2 will yield the needed offset */
    __m128i rmd2i = _mm_cvttps_epi32(rmd2);    /* after being truncated of course */
    __m128 rmd2Trunc = _mm_cvtepi32_ps(rmd2i);
    __m128 r =_mm_add_ps(aTrunc, rmd2Trunc);
    return fvec4(r);
#endif
}

static INLINE fvec4 fmod(const fvec4& a, const fvec4& aDiv){
    /* http://dss.stephanierct.com/DevBlog/?p=8 */
    __m128 c = _mm_div_ps(a.val, aDiv.val);
    __m128i i = _mm_cvttps_epi32(c);
    __m128 cTrunc = _mm_cvtepi32_ps(i);
    __m128 base = _mm_mul_ps(cTrunc, aDiv);
    __m128 r = _mm_sub_ps(a, base);
    return fvec4(r);
}


static INLINE fvec4 min(const fvec4& v1, const fvec4& v2) {
    return fvec4(_mm_min_ps(v1.val, v2.val));
}

static INLINE fvec4 max(const fvec4& v1, const fvec4& v2) {
    return fvec4(_mm_max_ps(v1.val, v2.val));
}

static INLINE fvec4 abs(const fvec4& v) {
    static const __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));
    return fvec4(_mm_and_ps(v.val, mask));
}

static INLINE fvec4 sqrt(const fvec4& v) {
    return fvec4(_mm_sqrt_ps(v.val));
}

static INLINE float dot3(const fvec4& v1, const fvec4& v2) {
    return _mm_cvtss_f32(_mm_dp_ps(v1, v2, 0x71));
}

static INLINE float dot4(const fvec4& v1, const fvec4& v2) {
    return _mm_cvtss_f32(_mm_dp_ps(v1, v2, 0xF1));
}

static INLINE void transpose(fvec4& v1, fvec4& v2, fvec4& v3, fvec4& v4) {
    _MM_TRANSPOSE4_PS(v1, v2, v3, v4);
}

// Functions that operate on ivec4s.

static INLINE ivec4 min(const ivec4& v1, const ivec4& v2) {
    return ivec4(_mm_min_epi32(v1.val, v2.val));
}

static INLINE ivec4 max(const ivec4& v1, const ivec4& v2) {
    return ivec4(_mm_max_epi32(v1.val, v2.val));
}

static INLINE ivec4 abs(const ivec4& v) {
    return ivec4(_mm_abs_epi32(v.val));
}

static INLINE bool any(const ivec4& v) {
#ifdef __SSE4_1__
    return !_mm_test_all_zeros(v, _mm_set1_epi32(0xFFFFFFFF));
#else
    /* http://stackoverflow.com/a/10250306 */
    return !(_mm_movemask_epi8(_mm_cmpeq_epi8(v, _mm_setzero_si128())) == 0xFFFF);
#endif
}

static INLINE long sum(const ivec4& v) {
    int result[4];
    v.store(result);
    return v[0] + v[1] + v[2] + v[3];
}

// Mathematical operators involving a scalar and a vector.

static INLINE fvec4 operator+(float v1, const fvec4& v2) {
    return fvec4(v1)+v2;
}

static INLINE fvec4 operator-(float v1, const fvec4& v2) {
    return fvec4(v1)-v2;
}

static INLINE fvec4 operator*(float v1, const fvec4& v2) {
    return fvec4(v1)*v2;
}

static INLINE fvec4 operator/(float v1, const fvec4& v2) {
    return fvec4(v1)/v2;
}

#endif /*OPENMM_VECTORIZE_SSE_H_*/
