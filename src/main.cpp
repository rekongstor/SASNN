#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include "../include/Dataset/DatasetStandard.h"
#include "../include/NeuralNetwork/ClassificationNN.h"

const char *fileName = "";
size_t batchSize = 512;
f32 testCoef = 0.1f;
f32 validationCoef = 0.1f;

int main(int argc, const char *argv[]) {
    if (argc < 2) {
        std::cout << "Please specify an input file" << std::endl;
        return 4221;
    } else
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
    if (argc > 4)
        try {
            validationCoef = std::stof(argv[4]);
        }
        catch (std::exception const &e) {
            std::cout << "Invalid validation coef" << argv[3] << std::endl;
            return 4;
        }


    DatasetStandard datasetStandard(fileName, batchSize, testCoef, validationCoef);
    ClassificationNN classificationNn(datasetStandard, 16u, 16u);

    Dataset &dataset = datasetStandard;
    dataset.PreprocessMean();
    NeuralNetwork &NN = classificationNn;
    NN.ModifyParam('g', 0.01f);
    NN.ModifyParam('l', 50.f);
    for (int i = 0; i < 100; ++i)
        std::cout << NN.Train(20) << std::endl;
    std::cout << "Test: " << NN.Test() << std::endl;
    NN.Serialize("nn.txt");
    system("pause");
    return 0;
}
