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
    NN.ModifyParam('l', 0.0003f);
    NN.ModifyParam('r', 1.0f);
    NN.ModifyParam('a', 0.5f);
    for (int i = 0; i < 80; ++i) {
        auto[train_acc, val_acc] = NN.Train();
        printf("Accuracy Train/Validation: [%.4f]/[%.4f] Diff: %.4f\n", train_acc, val_acc, static_cast<f64>(train_acc) - val_acc);
    }
    auto[val_acc, test_acc] = NN.Test();
    printf("Accuracy Validation/Test: [%.4f]/[%.4f] Diff: %.4f\n", val_acc, test_acc, static_cast<f64>(val_acc) - test_acc);
    NN.Serialize("SAS.NN");
}

void UseNN(NeuralNetwork &NN, Matrix2D &input, Matrix2D &out) {
    NN.Use(input,out);
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
    Dataset &dataset = datasetStandard;
    //dataset.PreprocessMean();

    RegressionNN regressionNN(datasetStandard, 32, 32, 32, 32);

    {
        TrainNN(regressionNN);
        SerializeNN(regressionNN, "SAS.NN");
    }
//    {
//        DeserializeNN(regressionNN, "SAS.NN");
//        Matrix2D in(1, 6);
//        f32 src[] = {0.11,0.69,0.29,0.10,-0.75,0.83};
//        in.AssignData(src);
//        Matrix2D out(1, 2);
//        UseNN(regressionNN, in, out);
//        std::cout << out(0,0);
//    }

    return 0;
}
