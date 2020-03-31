#include <iostream>
#include <fstream>
#include <filesystem>
#include "../include/Dataset/DatasetStandard.h"
#include "../include/NeuralNetwork/ClassificationNN.h"

const char *fileName = "";
size_t batchSize = 512;
f32 testCoef = 0.1f;

int main(s32 argc, const s8 *argv[]) {
    if (argc < 2) {
        std::cout << "Please specify an input file" << std::endl;
        return 4221;
    } else if (argc > 1)
        fileName = argv[1];
    if (!std::filesystem::exists(fileName)) {
        std::cout << "Invalid file name: " << argv[1] << std::endl;
        return 1;
    }
    if (argc > 2)
        try {
            batchSize = std::stoi(argv[2]);
        }
        catch (std::exception const &e) {
            std::cout << "Invalid batch size" << argv[2] << std::endl;
            return 2;
        }
    if (argc > 3)
        try {
            testCoef = std::stof(argv[3]);
        }
        catch (std::exception const &e) {
            std::cout << "Invalid test coef" << argv[3] << std::endl;
            return 3;
        }


    DatasetStandard datasetStandard(std::ifstream(fileName, std::ios::binary), batchSize, testCoef);
    ClassificationNN classificationNn(datasetStandard, 1u, 15u);

    NeuralNetwork &NN = classificationNn;

    NN.Train(500);
    return 0;
}
