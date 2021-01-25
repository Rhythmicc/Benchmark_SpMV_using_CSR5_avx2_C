#pragma once
#ifndef UTILS_AVX2_H
#define UTILS_AVX2_H

#include "common_avx2.h"

typedef struct anonymouslib_timer anonymouslib_timer, *anonymouslib_timer_t;

struct anonymouslib_timer{
    struct timeval t1, t2;
    struct timezone tzone;

    void (*start)(anonymouslib_timer_t timer);
    double (*stop)(anonymouslib_timer_t timer);
};

void start(anonymouslib_timer_t timer) {
    gettimeofday(&timer->t1, &timer->tzone);
}

double stop(anonymouslib_timer_t timer) {
    gettimeofday(&timer->t2, &timer->tzone);
    double elapsedTime = 0;
    elapsedTime = (timer->t2.tv_sec - timer->t1.tv_sec) * 1000.0;      // sec to ms
    elapsedTime += (timer->t2.tv_usec - timer->t1.tv_usec) / 1000.0;   // us to ms
    return elapsedTime;
}

void init_anonymouslib_timer(anonymouslib_timer_t timer) {
    timer->start = start;
    timer->stop = stop;
}

iT binary_search_right_boundary_kernel(const iT *d_row_pointer,
                                       const iT  key_input,
                                       const iT  size)
{
    iT start = 0;
    iT stop  = size - 1;
    iT median;
    iT key_median;

    while (stop >= start)
    {
        median = (stop + start) / 2;

        key_median = d_row_pointer[median];

        if (key_input >= key_median)
            start = median + 1;
        else
            stop = median - 1;
    }

    return start;
}

// sum up 4 double-precision numbers
inline double hsum_avx(__m256d in256d)
{
    double sum;

    __m256d hsum = _mm256_add_pd(in256d, _mm256_permute2f128_pd(in256d, in256d, 0x1));
    _mm_store_sd(&sum, _mm_hadd_pd( _mm256_castpd256_pd128(hsum), _mm256_castpd256_pd128(hsum) ) );

    return sum;
}

// exclusive scan using a single thread
void scan_single(int       *s_scan,
                 const int l)
{
    int old_val, new_val;

    old_val = s_scan[0];
    s_scan[0] = 0;
    for (int i = 1; i < l; i++)
    {
        new_val = s_scan[i];
        s_scan[i] = old_val + s_scan[i-1];
        old_val = new_val;
    }
}

// inclusive prefix-sum scan 
inline __m256d hscan_avx(__m256d in256d)
{
    __m256d t0, t1;
    t0 = _mm256_permute4x64_pd(in256d, 0x93);
    t1 = _mm256_add_pd(in256d, _mm256_blend_pd(t0, _mm256_set1_pd(0), 0x1));

    t0 = _mm256_permute4x64_pd(in256d, 0x4E);
    t1 = _mm256_add_pd(t1, _mm256_blend_pd(t0, _mm256_set1_pd(0), 0x3));

    t0 = _mm256_permute4x64_pd(in256d, 0x39);
    t1 = _mm256_add_pd(t1, _mm256_blend_pd(t0, _mm256_set1_pd(0), 0x7));

    return t1;
}

#endif // UTILS_AVX2_H