#include <iostream>
#include <cmath>
#include "../include/Matrix2D.h"
#include "../include/Dataset.h"
#include "../include/functors.h"

int main() {
    Matrix2D matrix2D(4, 3, true);
    Matrix2D matrix2Darg(4, 3);
    matrix2D.CellOperator(matrix2Darg, expf);
    matrix2D.CellOperator(matrix2Darg, expf);
    matrix2D.CellOperator(matrix2Darg, expf);


    for (size_t i = 0; i < matrix2D.getRows(); ++i) {
        for (size_t j = 0; j < matrix2D.getCols(); ++j)
            std::cout << matrix2D(i, j) << " ";
        std::cout << std::endl;
    }
    return 0;
}
