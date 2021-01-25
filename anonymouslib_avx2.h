#pragma once
#ifndef ANONYMOUSLIB_AVX2_H
#define ANONYMOUSLIB_AVX2_H

#include "utils.h"
#include "avx2/utils_avx2.h"

#include "avx2/common_avx2.h"
#include "avx2/format_avx2.h"
#include "avx2/csr5_spmv_avx2.h"

typedef struct anonymouslibHandle anonymouslibHandle, *anonymouslibHandle_t;

struct anonymouslibHandle{
    int (*warmup)(anonymouslibHandle_t self);
    int (*inputCSR)(ANONYMOUSLIB_IT  nnz, ANONYMOUSLIB_IT *csr_row_pointer, ANONYMOUSLIB_IT *csr_column_index, ANONYMOUSLIB_VT *csr_value, anonymouslibHandle_t self);
    int (*asCSR)(anonymouslibHandle_t self);
    int (*asCSR5)(anonymouslibHandle_t self);
    int (*setX)(ANONYMOUSLIB_VT *x, anonymouslibHandle_t self);
    int (*spmv)(const ANONYMOUSLIB_VT alpha, ANONYMOUSLIB_VT *y, anonymouslibHandle_t self);
    int (*destroy)(anonymouslibHandle_t self);
    void (*setSigma)(int sigma, anonymouslibHandle_t self);

    int (*_computeSigma)(anonymouslibHandle_t self);
    int _format;
    ANONYMOUSLIB_IT _m;
    ANONYMOUSLIB_IT _n;
    ANONYMOUSLIB_IT _nnz;

    ANONYMOUSLIB_IT *_csr_row_pointer;
    ANONYMOUSLIB_IT *_csr_column_index;
    ANONYMOUSLIB_VT *_csr_value;

    int _csr5_sigma;
    int _bit_y_offset;
    int _bit_scansum_offset;
    int _num_packet;
    ANONYMOUSLIB_IT _tail_partition_start;

    ANONYMOUSLIB_IT _p;
    ANONYMOUSLIB_UIT *_csr5_partition_pointer;
    ANONYMOUSLIB_UIT *_csr5_partition_descriptor;

    ANONYMOUSLIB_IT   _num_offsets;
    ANONYMOUSLIB_IT  *_csr5_partition_descriptor_offset_pointer;
    ANONYMOUSLIB_IT  *_csr5_partition_descriptor_offset;
    ANONYMOUSLIB_VT  *_temp_calibrator;

    ANONYMOUSLIB_VT         *_x;
};

int AnonymouslibHandle_warmup(anonymouslibHandle_t self) {
    return ANONYMOUSLIB_SUCCESS;
}

int anonymouslibHandle_inputCSR(ANONYMOUSLIB_IT nnz,
                                ANONYMOUSLIB_IT *csr_row_pointer,
                                ANONYMOUSLIB_IT *csr_column_index,
                                ANONYMOUSLIB_VT *csr_value,
                                anonymouslibHandle_t self) {
    self->_format = ANONYMOUSLIB_FORMAT_CSR;

    self->_nnz = nnz;

    self->_csr_row_pointer = csr_row_pointer;
    self->_csr_column_index = csr_column_index;
    self->_csr_value = csr_value;

    return ANONYMOUSLIB_SUCCESS;
}

int anonymouslibHandle_asCSR(anonymouslibHandle_t self) {
    int err = ANONYMOUSLIB_SUCCESS;

    if (self->_format == ANONYMOUSLIB_FORMAT_CSR)
        return err;

    if (self->_format == ANONYMOUSLIB_FORMAT_CSR5) {
        // convert csr5 data to csr data
        err = aosoa_transpose(self->_csr5_sigma, self->_nnz,
                              self->_csr5_partition_pointer, self->_csr_column_index, self->_csr_value, false);

        // free the two newly added CSR5 arrays
        _mm_free(self->_csr5_partition_pointer);
        _mm_free(self->_csr5_partition_descriptor);
        _mm_free(self->_temp_calibrator);
        _mm_free(self->_csr5_partition_descriptor_offset_pointer);
        if (self->_num_offsets) _mm_free(self->_csr5_partition_descriptor_offset);

        self->_format = ANONYMOUSLIB_FORMAT_CSR;
    }

    return err;
}

int anonymouslibHandle_asCSR5(anonymouslibHandle_t self) {
    int err = ANONYMOUSLIB_SUCCESS;

    if (self->_format == ANONYMOUSLIB_FORMAT_CSR5)
        return err;

    if (self->_format == ANONYMOUSLIB_FORMAT_CSR) {
        double malloc_time = 0, tile_ptr_time = 0, tile_desc_time = 0, transpose_time = 0;
        anonymouslib_timer malloc_timer, tile_ptr_timer, tile_desc_timer, transpose_timer;

        init_anonymouslib_timer(&malloc_timer);
        init_anonymouslib_timer(&tile_desc_timer);
        init_anonymouslib_timer(&tile_desc_timer);
        init_anonymouslib_timer(&transpose_timer);

        // compute sigma
        self->_csr5_sigma = self->_computeSigma(self);
        printf("omega = %d, sigma = %d. ", ANONYMOUSLIB_CSR5_OMEGA, self->_csr5_sigma);

        // compute how many bits required for `y_offset' and `carry_offset'
        int base = 2;
        self->_bit_y_offset = 1;
        while (base < ANONYMOUSLIB_CSR5_OMEGA * self->_csr5_sigma) {
            base *= 2;
            self->_bit_y_offset++;
        }

        base = 2;
        self->_bit_scansum_offset = 1;
        while (base < ANONYMOUSLIB_CSR5_OMEGA) {
            base *= 2;
            self->_bit_scansum_offset++;
        }

        if (self->_bit_y_offset + self->_bit_scansum_offset >
            sizeof(ANONYMOUSLIB_UIT) * 8 - 1) //the 1st bit of bit-flag should be in the first packet
            return ANONYMOUSLIB_UNSUPPORTED_CSR5_OMEGA;

        int bit_all = self->_bit_y_offset + self->_bit_scansum_offset + self->_csr5_sigma;
        self->_num_packet = ceil((double) bit_all / (double) (sizeof(ANONYMOUSLIB_UIT) * 8));

        // calculate the number of partitions
        self->_p = ceil((double) self->_nnz / (double) (ANONYMOUSLIB_CSR5_OMEGA * self->_csr5_sigma));
        printf("#partition = %d\n", self->_p);

        malloc_timer.start(&malloc_timer);
        // malloc the newly added arrays for CSR5
        self->_csr5_partition_pointer = (ANONYMOUSLIB_UIT *) _mm_malloc((self->_p + 1) * sizeof(ANONYMOUSLIB_UIT),
                                                                  ANONYMOUSLIB_X86_CACHELINE);

        self->_csr5_partition_descriptor = (ANONYMOUSLIB_UIT *) _mm_malloc(
                self->_p * ANONYMOUSLIB_CSR5_OMEGA * self->_num_packet * sizeof(ANONYMOUSLIB_UIT), ANONYMOUSLIB_X86_CACHELINE);
        memset(self->_csr5_partition_descriptor, 0, self->_p * ANONYMOUSLIB_CSR5_OMEGA * self->_num_packet * sizeof(ANONYMOUSLIB_UIT));

        int num_thread = omp_get_max_threads();
        self->_temp_calibrator = (ANONYMOUSLIB_VT * )
        _mm_malloc(num_thread * ANONYMOUSLIB_X86_CACHELINE, ANONYMOUSLIB_X86_CACHELINE);
        memset(self->_temp_calibrator, 0, num_thread * ANONYMOUSLIB_X86_CACHELINE);

        self->_csr5_partition_descriptor_offset_pointer = (ANONYMOUSLIB_IT * )
        _mm_malloc((self->_p + 1) * sizeof(ANONYMOUSLIB_IT), ANONYMOUSLIB_X86_CACHELINE);
        memset(self->_csr5_partition_descriptor_offset_pointer, 0, (self->_p + 1) * sizeof(ANONYMOUSLIB_IT));
        malloc_time += malloc_timer.stop(&malloc_timer);

        // convert csr data to csr5 data (3 steps)
        // step 1. generate partition pointer
        tile_ptr_timer.start(&tile_ptr_timer);
        err = generate_partition_pointer(self->_csr5_sigma, self->_p, self->_m, self->_nnz,
                                         self->_csr5_partition_pointer, self->_csr_row_pointer);
        if (err != ANONYMOUSLIB_SUCCESS)
            return ANONYMOUSLIB_CSR_TO_CSR5_FAILED;
        tile_ptr_time += tile_ptr_timer.stop(&tile_ptr_timer);

        self->_tail_partition_start = (self->_csr5_partition_pointer[self->_p - 1] << 1) >> 1;
        //cout << "_tail_partition_start = " << _tail_partition_start << endl;

        // step 2. generate partition descriptor
        tile_desc_timer.start(&tile_desc_timer);
        self->_num_offsets = 0;
        err = generate_partition_descriptor(self->_csr5_sigma, self->_p, self->_m,
                                            self->_bit_y_offset, self->_bit_scansum_offset,
                                            self->_num_packet,
                                            self->_csr_row_pointer,
                                            self->_csr5_partition_pointer,
                                            self->_csr5_partition_descriptor,
                                            self->_csr5_partition_descriptor_offset_pointer,
                                            &self->_num_offsets);
        if (err != ANONYMOUSLIB_SUCCESS)
            return ANONYMOUSLIB_CSR_TO_CSR5_FAILED;
        tile_desc_time += tile_desc_timer.stop(&tile_desc_timer);

        if (self->_num_offsets) {
            //cout << "has empty rows, _num_offsets = " << _num_offsets << endl;
            malloc_timer.start(&malloc_timer);
            self->_csr5_partition_descriptor_offset = (ANONYMOUSLIB_IT * )
            _mm_malloc(self->_num_offsets * sizeof(ANONYMOUSLIB_IT), ANONYMOUSLIB_X86_CACHELINE);
            //memset(_csr5_partition_descriptor_offset, 0, _num_offsets * sizeof(ANONYMOUSLIB_IT));
            malloc_time += malloc_timer.stop(&malloc_timer);

            tile_desc_timer.start(&tile_desc_timer);
            err = generate_partition_descriptor_offset(self->_csr5_sigma, self->_p,
                                                       self->_bit_y_offset,
                                                       self->_bit_scansum_offset,
                                                       self->_num_packet,
                                                       self->_csr_row_pointer,
                                                       self->_csr5_partition_pointer,
                                                       self->_csr5_partition_descriptor,
                                                       self->_csr5_partition_descriptor_offset_pointer,
                                                       self->_csr5_partition_descriptor_offset);

            //for (int i = 0; i < _num_offsets; i++)
            //    cout << "_csr5_partition_descriptor_offset = " << _csr5_partition_descriptor_offset[i] << endl;
            if (err != ANONYMOUSLIB_SUCCESS)
                return ANONYMOUSLIB_CSR_TO_CSR5_FAILED;
            tile_desc_time += tile_desc_timer.stop(&tile_desc_timer);
        }

        // step 3. transpose column_index and value arrays
        transpose_timer.start(&transpose_timer);
        err = aosoa_transpose(self->_csr5_sigma, self->_nnz,
                              self->_csr5_partition_pointer, self->_csr_column_index, self->_csr_value, true);
        if (err != ANONYMOUSLIB_SUCCESS)
            return ANONYMOUSLIB_CSR_TO_CSR5_FAILED;
        transpose_time += transpose_timer.stop(&transpose_timer);

        printf("CSR->CSR5 malloc time = %f ms\n", malloc_time);
        printf("CSR->CSR5 tile_ptr time = %f ms\n", tile_ptr_time);
        printf("CSR->CSR5 tile_desc time = %f ms\n", tile_desc_time);
        printf("CSR->CSR5 transpose time = %f ms\n", transpose_time);

        self->_format = ANONYMOUSLIB_FORMAT_CSR5;
    }

    return err;
}

int anonymouslibHandle_setX(ANONYMOUSLIB_VT *x, anonymouslibHandle_t self) {
    int err = ANONYMOUSLIB_SUCCESS;

    self->_x = x;

    return err;
}

int anonymouslibHandle_spmv(const ANONYMOUSLIB_VT alpha, ANONYMOUSLIB_VT *y, anonymouslibHandle_t self) {
    int err = ANONYMOUSLIB_SUCCESS;

    if (self->_format == ANONYMOUSLIB_FORMAT_CSR) {
        return ANONYMOUSLIB_UNSUPPORTED_CSR_SPMV;
    }

    if (self->_format == ANONYMOUSLIB_FORMAT_CSR5) {
        csr5_spmv(self->_csr5_sigma, self->_p, self->_m,
                  self->_bit_y_offset, self->_bit_scansum_offset, self->_num_packet,
                  self->_csr_row_pointer, self->_csr_column_index, self->_csr_value,
                  self->_csr5_partition_pointer,
                  self->_csr5_partition_descriptor,
                  self->_csr5_partition_descriptor_offset_pointer,
                  self->_csr5_partition_descriptor_offset,
                  self->_temp_calibrator, self->_tail_partition_start,
                  alpha, self->_x, y);
    }

    return err;
}

int anonymouslibHandle_destroy(anonymouslibHandle_t self) {
    return self->asCSR(self);
}

void anonymouslibHandle_setSigma(int sigma, anonymouslibHandle_t self) {
    self->_csr5_sigma = sigma;
}

int anonymouslibHandle_computeSigma(anonymouslibHandle_t self) {
    return self->_csr5_sigma;
}

void anonymouslibHandle_init(ANONYMOUSLIB_IT m, ANONYMOUSLIB_IT n, anonymouslibHandle_t self) {
    self->_m = m;
    self->_n = n;
    self->asCSR = anonymouslibHandle_asCSR;
    self->asCSR5 = anonymouslibHandle_asCSR5;
    self->destroy = anonymouslibHandle_destroy;
    self->inputCSR = anonymouslibHandle_inputCSR;
    self->setSigma = anonymouslibHandle_setSigma;
    self->setX = anonymouslibHandle_setX;
    self->spmv = anonymouslibHandle_spmv;
    self->warmup = AnonymouslibHandle_warmup;
    self->_computeSigma = anonymouslibHandle_computeSigma;
}

#endif // ANONYMOUSLIB_AVX2_H
