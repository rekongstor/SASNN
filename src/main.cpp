#include <iostream>
#include <cmath>
#include "../include/Matrix2D.h"
#include "../include/Dataset.h"
#include "../include/functors.h"

int main() {
    Matrix2D matrix2D(4,4);
    Matrix2D matrix2Darg(4,4);
    matrix2D.CellOperator(matrix2Darg,expf);
    Matrix2D m1(1,1);
    m1.MergeAllOperator(matrix2D,sum);
    return 0;
}
