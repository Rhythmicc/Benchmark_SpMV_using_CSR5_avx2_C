#include <math.h>
#include "anonymouslib_avx2.h"
#include "mmio.h"

#ifndef VALUE_TYPE
#define VALUE_TYPE double
#endif

#ifndef NUM_RUN
#define NUM_RUN 100
#endif

int call_anonymouslib(int m, int n, int nnzA,
                      int *csrRowPtrA, int *csrColIdxA, VALUE_TYPE *csrValA,
                      VALUE_TYPE *x, VALUE_TYPE *y, VALUE_TYPE alpha) {
    int err = 0;

    memset(y, 0, sizeof(VALUE_TYPE) * m);

    double gb = getB(m, nnzA);
    double gflop = getFLOP(nnzA);

    anonymouslibHandle A;
    anonymouslibHandle_init(m, n, &A);
    err = A.inputCSR(nnzA, csrRowPtrA, csrColIdxA, csrValA, &A);
    //cout << "inputCSR err = " << err << endl;

    err = A.setX(x, &A); // you only need to do it once!
    //cout << "setX err = " << err << endl;

    VALUE_TYPE *y_bench = (VALUE_TYPE *) malloc(m * sizeof(VALUE_TYPE));

    int sigma = ANONYMOUSLIB_CSR5_SIGMA; //nnzA/(8*ANONYMOUSLIB_CSR5_OMEGA);
    A.setSigma(sigma, &A);

    for (int i = 0; i < 5; i++) {
        err = A.asCSR5(&A);
        err = A.asCSR(&A);
    }

    anonymouslib_timer asCSR5_timer;
    init_anonymouslib_timer(&asCSR5_timer);
    asCSR5_timer.start(&asCSR5_timer);

    err = A.asCSR5(&A);

    printf("CSR->CSR5 time = %g ms.\n", asCSR5_timer.stop(&asCSR5_timer));

    // check correctness by running 1 time
    err = A.spmv(alpha, y, &A);
    //cout << "spmv err = " << err << endl;

    // warm up by running 50 times
    if (NUM_RUN) {
        for (int i = 0; i < 50; i++) {
            memset(y, 0, sizeof(double) * m);
            err = A.spmv(alpha, y, &A);
        }

        anonymouslib_timer CSR5Spmv_timer;
        CSR5Spmv_timer.start(&CSR5Spmv_timer);

        for (int i = 0; i < NUM_RUN; i++) {
            err = A.spmv(alpha, y_bench, &A);
        }

        double CSR5Spmv_time = CSR5Spmv_timer.stop(&CSR5Spmv_timer) / (double) NUM_RUN;

        printf("CSR5-based SpMV time = %g,  ms. Bandwidth = %g GB/s. GFlops = %g GFlops.\n", CSR5Spmv_time,
               gb / (1.0e+6 * CSR5Spmv_time), gflop / (1.0e+6 * CSR5Spmv_time));
    }

    free(y_bench);

    A.destroy(&A);

    return err;
}

int main(int argc, char **argv) {
    // report precision of floating-point
    puts("------------------------------------------------------");
    char *precision;
    if (sizeof(VALUE_TYPE) == 4) precision = "32-bit Single Precision";
    else if (sizeof(VALUE_TYPE) == 8) precision = "64-bit Double Precision";
    else {
        puts("Wrong precision. Program exit!");
        return 0;
    }

    printf("PRECISION = %p", precision);
    puts("------------------------------------------------------");

    int m, n, nnzA;
    int *csrRowPtrA;
    int *csrColIdxA;
    VALUE_TYPE *csrValA;

    //ex: ./spmv webbase-1M.mtx
    int argi = 1;

    char *filename;
    if (argc > argi) {
        filename = argv[argi];
        argi++;
    }
    printf("--------------%s--------------\n", filename);

    // read matrix from mtx file
    int ret_code;
    MM_typecode matcode;
    FILE *f;

    int nnzA_mtx_report;
    int isInteger = 0, isReal = 0, isPattern = 0, isSymmetric = 0;

    // load matrix
    if ((f = fopen(filename, "r")) == NULL)
        return -1;

    if (mm_read_banner(f, &matcode) != 0) {
        puts("Could not process Matrix Market banner.");
        return -2;
    }

    if (mm_is_complex(matcode)) {
        puts("Sorry, data type 'COMPLEX' is not supported. ");
        return -3;
    }

    if (mm_is_pattern(matcode)) { isPattern = 1; /*cout << "type = Pattern" << endl;*/ }
    if (mm_is_real (matcode)) { isReal = 1; /*cout << "type = real" << endl;*/ }
    if (mm_is_integer (matcode)) { isInteger = 1; /*cout << "type = integer" << endl;*/ }

    /* find out size of sparse matrix .... */
    ret_code = mm_read_mtx_crd_size(f, &m, &n, &nnzA_mtx_report);
    if (ret_code != 0)
        return -4;

    if (mm_is_symmetric(matcode) || mm_is_hermitian(matcode)) {
        isSymmetric = 1;
        //cout << "symmetric = true" << endl;
    } else {
        //cout << "symmetric = false" << endl;
    }

    int *csrRowPtrA_counter = (int *) malloc((m + 1) * sizeof(int));
    memset(csrRowPtrA_counter, 0, (m + 1) * sizeof(int));

    int *csrRowIdxA_tmp = (int *) malloc(nnzA_mtx_report * sizeof(int));
    int *csrColIdxA_tmp = (int *) malloc(nnzA_mtx_report * sizeof(int));
    VALUE_TYPE *csrValA_tmp = (VALUE_TYPE *) malloc(nnzA_mtx_report * sizeof(VALUE_TYPE));

    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (int i = 0; i < nnzA_mtx_report; i++) {
        int idxi, idxj;
        double fval;
        int ival;

        if (isReal)
            fscanf(f, "%d %d %lg\n", &idxi, &idxj, &fval);
        else if (isInteger) {
            fscanf(f, "%d %d %d\n", &idxi, &idxj, &ival);
            fval = ival;
        } else if (isPattern) {
            fscanf(f, "%d %d\n", &idxi, &idxj);
            fval = 1.0;
        }

        // adjust from 1-based to 0-based
        idxi--;
        idxj--;

        csrRowPtrA_counter[idxi]++;
        csrRowIdxA_tmp[i] = idxi;
        csrColIdxA_tmp[i] = idxj;
        csrValA_tmp[i] = fval;
    }

    if (f != stdin)
        fclose(f);

    if (isSymmetric) {
        for (int i = 0; i < nnzA_mtx_report; i++) {
            if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i])
                csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
        }
    }

    // exclusive scan for csrRowPtrA_counter
    int old_val, new_val;

    old_val = csrRowPtrA_counter[0];
    csrRowPtrA_counter[0] = 0;
    for (int i = 1; i <= m; i++) {
        new_val = csrRowPtrA_counter[i];
        csrRowPtrA_counter[i] = old_val + csrRowPtrA_counter[i - 1];
        old_val = new_val;
    }

    nnzA = csrRowPtrA_counter[m];
    csrRowPtrA = (int *) _mm_malloc((m + 1) * sizeof(int), ANONYMOUSLIB_X86_CACHELINE);
    memcpy(csrRowPtrA, csrRowPtrA_counter, (m + 1) * sizeof(int));
    memset(csrRowPtrA_counter, 0, (m + 1) * sizeof(int));

    csrColIdxA = (int *) _mm_malloc(nnzA * sizeof(int), ANONYMOUSLIB_X86_CACHELINE);
    csrValA = (VALUE_TYPE *) _mm_malloc(nnzA * sizeof(VALUE_TYPE), ANONYMOUSLIB_X86_CACHELINE);

    if (isSymmetric) {
        for (int i = 0; i < nnzA_mtx_report; i++) {
            if (csrRowIdxA_tmp[i] != csrColIdxA_tmp[i]) {
                int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
                csrColIdxA[offset] = csrColIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;

                offset = csrRowPtrA[csrColIdxA_tmp[i]] + csrRowPtrA_counter[csrColIdxA_tmp[i]];
                csrColIdxA[offset] = csrRowIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrColIdxA_tmp[i]]++;
            } else {
                int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
                csrColIdxA[offset] = csrColIdxA_tmp[i];
                csrValA[offset] = csrValA_tmp[i];
                csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
            }
        }
    } else {
        for (int i = 0; i < nnzA_mtx_report; i++) {
            int offset = csrRowPtrA[csrRowIdxA_tmp[i]] + csrRowPtrA_counter[csrRowIdxA_tmp[i]];
            csrColIdxA[offset] = csrColIdxA_tmp[i];
            csrValA[offset] = csrValA_tmp[i];
            csrRowPtrA_counter[csrRowIdxA_tmp[i]]++;
        }
    }

    // free tmp space
    free(csrColIdxA_tmp);
    free(csrValA_tmp);
    free(csrRowIdxA_tmp);
    free(csrRowPtrA_counter);

    srand(time(NULL));

    // set csrValA to 1, easy for checking floating-point results
    for (int i = 0; i < nnzA; i++) {
        csrValA[i] = rand() % 10;
    }

    printf("(%d, %d) nnz = %d\n", m, n, nnzA);

    VALUE_TYPE *x = (VALUE_TYPE *) _mm_malloc(n * sizeof(VALUE_TYPE), ANONYMOUSLIB_X86_CACHELINE);
    for (int i = 0; i < n; i++)
        x[i] = rand() % 10;

    VALUE_TYPE *y = (VALUE_TYPE *) _mm_malloc(m * sizeof(VALUE_TYPE), ANONYMOUSLIB_X86_CACHELINE);
    VALUE_TYPE *y_ref = (VALUE_TYPE *) _mm_malloc(m * sizeof(VALUE_TYPE), ANONYMOUSLIB_X86_CACHELINE);

    double gb = getB(m, nnzA);
    double gflop = getFLOP(nnzA);

    VALUE_TYPE alpha = 1.0;

    // compute reference results on a cpu core
    anonymouslib_timer ref_timer;
    init_anonymouslib_timer(&ref_timer);
    ref_timer.start(&ref_timer);

    int ref_iter = 1;
    for (int iter = 0; iter < ref_iter; iter++) {
        for (int i = 0; i < m; i++) {
            VALUE_TYPE sum = 0;
            for (int j = csrRowPtrA[i]; j < csrRowPtrA[i + 1]; j++)
                sum += x[csrColIdxA[j]] * csrValA[j] * alpha;
            y_ref[i] = sum;
        }
    }

    double ref_time = ref_timer.stop(&ref_timer) / (double) ref_iter;
    printf("cpu sequential time = %g ms. Bandwidth = %g GB/s. GFlops = %g GFlops.\n\n", ref_time,
           gb / (1.0e+6 * ref_time), gflop / (1.0e+6 * ref_time));

    // launch compute
    call_anonymouslib(m, n, nnzA, csrRowPtrA, csrColIdxA, csrValA, x, y, alpha);

    // compare reference and anonymouslib results
    int error_count = 0;
    for (int i = 0; i < m; i++)
        if (fabs(y_ref[i] - y[i]) > 0.01 * fabs(y_ref[i])) {
            error_count++;
//            cout << "ROW [ " << i << " ], NNZ SPAN: "
//                 << csrRowPtrA[i] << " - "
//                 << csrRowPtrA[i+1]
//                 << "\t ref = " << y_ref[i]
//                 << ", \t csr5 = " << y[i]
//                 << ", \t error = " << y_ref[i] - y[i]
//                 << endl;
//            break;
        }

    if (error_count == 0)
        puts("Check... PASS!");
    else
        printf("Check... NO PASS! #Error = %d out of %d entries.", error_count, m);

    puts("------------------------------------------------------");

    _mm_free(csrRowPtrA);
    _mm_free(csrColIdxA);
    _mm_free(csrValA);
    _mm_free(x);
    _mm_free(y);
    _mm_free(y_ref);

    return 0;
}
