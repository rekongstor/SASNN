#include <iostream>
#include <filesystem>
#include <string>
#include "../include/Dataset/DatasetStandard.h"
#include "../include/NeuralNetwork/ClassificationNN.h"
#include "../include/NeuralNetwork/RegressionNN.h"

const char *fileName = "";
size_t batchSize = 512;
f32 testCoef = 0.1f;
f32 validationCoef = 0.1f;

void SerializeNN(NeuralNetwork &NN, const char *filename) {
    NN.Serialize(filename);
}

void DeserializeNN(NeuralNetwork &NN, const char *filename) {
    NN.Deserialize(filename);
}

void TrainNN(NeuralNetwork &NN) {
    //NN.ModifyParam('l', 0.0003f);
    NN.ModifyParam('r', 0.0000f);
    NN.ModifyParam('a', 0.5f);
    for (int i = 0; i < 1; ++i) {
        auto[train_acc, val_acc] = NN.Train();
        printf("Accuracy Train/Validation: [%.4f]/[%.4f] Diff: %.4f\n", train_acc, val_acc, static_cast<f64>(train_acc) - val_acc);
    }
}

void TestNN(NeuralNetwork &NN) {
    auto[val_acc, test_acc] = NN.Test();
    printf("Accuracy Validation/Test: [%.4f]/[%.4f] Diff: %.4f\n", val_acc, test_acc, static_cast<f64>(val_acc) - test_acc);
}

void UseNN(NeuralNetwork &NN) {
    NN.Use();
}

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
            batchSize = static_cast<size_t>(std::stoi(argv[2]));
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
    //DatasetStandard datasetStandard("D:\\SASTEST", 1100, 1100, 1100);
    Dataset &dataset = datasetStandard;
    //dataset.PreprocessMean();
    RegressionNN regressionNN(datasetStandard, 64, 64, 64, 64);

    {
//        TrainNN(regressionNN);
//        SerializeNN(regressionNN, "SAS.NN");

        for (int i = 0; i < 20; ++i) {
            DeserializeNN(regressionNN, "SAS.NN");
            TrainNN(regressionNN);
            SerializeNN(regressionNN, "SAS.NN");
        }
        TestNN(regressionNN);
    }

    DatasetStandard datasetStandard1("D:\\SASTEST", 1, 5847, 2);
    //DatasetStandard datasetStandard1("D:\\SASTEST", 1100, 1100, 1100);
    Dataset &dataset1 = datasetStandard1;

    RegressionNN regressionNN1(datasetStandard1, 64, 64, 64, 64);
    {
        DeserializeNN(regressionNN1, "SAS.NN");
        TestNN(regressionNN1);
        UseNN(regressionNN1);
    }

    return 0;
}
