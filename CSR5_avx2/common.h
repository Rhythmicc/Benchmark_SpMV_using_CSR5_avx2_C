#pragma once
#ifndef COMMON_H
#define COMMON_H

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define iT int
#define uiT unsigned int
#define vT double

#define ANONYMOUSLIB_SUCCESS                   0
#define ANONYMOUSLIB_UNKOWN_FORMAT            -1
#define ANONYMOUSLIB_UNSUPPORTED_CSR5_OMEGA   -2
#define ANONYMOUSLIB_CSR_TO_CSR5_FAILED       -3
#define ANONYMOUSLIB_UNSUPPORTED_CSR_SPMV     -4
#define ANONYMOUSLIB_UNSUPPORTED_VALUE_TYPE   -5

#define ANONYMOUSLIB_FORMAT_CSR  0
#define ANONYMOUSLIB_FORMAT_CSR5 1
#define ANONYMOUSLIB_FORMAT_HYB5 2

#define ANONYMOUSLIB_IT int
#define ANONYMOUSLIB_UIT unsigned int
#define ANONYMOUSLIB_VT double

typedef enum {false, true} bool;


#endif // COMMON_H
