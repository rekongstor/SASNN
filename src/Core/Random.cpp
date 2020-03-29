#include "../../include/Core/Random.h"
#include <ctime>

Random::Random(u64 seed) : engine(seed) {
}


RandomUniform::RandomUniform(f32 minRange, f32 maxRange, u64 seed) : Random(seed == 0 ? std::time(nullptr) : seed) {
    distribution = std::uniform_real_distribution<f32>(minRange, maxRange);
}

f32 RandomUniform::Next() {
    return distribution(engine);
}

RandomGaussian::RandomGaussian(f32 mean, f32 dev, u64 seed) : Random(seed == 0 ? std::time(nullptr) : seed) {
    distribution = std::normal_distribution<f32>(mean, dev);
}

f32 RandomGaussian::Next() {
    return distribution(engine);
}
