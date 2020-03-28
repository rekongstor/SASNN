#include <iostream>
#include "../include/Matrix2D.h"
#include "../include/Dataset.h"

int main() {
    f32 m[] = {1, 2,
               3, 4,
               5, 6,
               7, 8};
    f32 d[] = {1, 2, 3, 4,
               5, 6, 7, 8,
               1, 2, 3, 4,
               5, 6, 7, 8};
    Dataset dataset(2, 4, m, d, m, d, 2, 2, 2);
    return 0;
}
