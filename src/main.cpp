#include <iostream>
#include <cmath>
#include "../include/Core/Matrix2D.h"
#include "../include/Dataset/Dataset.h"
#include "../include/Core/functors.h"

int main() {
    Matrix2D matrix2D(4, 3, true);
    Matrix2D matrix2Darg(4, 3);
    matrix2D.CellOperator(matrix2Darg, expf);
    matrix2D.CellOperator(matrix2Darg, expf);
    Matrix2D s(4,1);
    s.MergeColOperator(matrix2D,mul);

    for (size_t i = 0; i < s.getRows(); ++i) {
        for (size_t j = 0; j < s.getCols(); ++j)
            std::cout << s(i, j) << " ";
        std::cout << std::endl;
    }
    return 0;
}
