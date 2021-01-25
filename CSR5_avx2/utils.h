#pragma once
#ifndef UTILS_H
#define UTILS_H

#include "common.h"

#include <sys/time.h>
#include <sys/types.h>
#include <dirent.h>

double getB(const iT m, const iT nnz) {
    return (double) ((m + 1 + nnz) * sizeof(iT) + (2 * nnz + m) * sizeof(vT));
}

double getFLOP(const iT nnz) {
    return (double) (2 * nnz);
}

#endif // UTILS_H
