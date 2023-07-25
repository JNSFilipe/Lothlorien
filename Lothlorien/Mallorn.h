// Mallorn.h : Include file for the Mallorn class

#pragma once

#include "SapWood.h"

class Mallorn {
public:
    Mallorn(int input_size, bool disable_sgd=false):
        root(input_size, disable_sgd),
        disable_sgd(disable_sgd)
    {}

    void train(const torch::Tensor& inputs, const torch::Tensor& targets, int num_epochs, float learning_rate, int batch_size, int stop_patience, float lr_annealing_factor, int min_samples, int max_depth) {
        root.train(inputs, targets, num_epochs, learning_rate, batch_size, stop_patience, lr_annealing_factor, min_samples, max_depth);
    }

    torch::Tensor operator()(const torch::Tensor& inputs) {
        return predict(inputs);
    }

    torch::Tensor predict(const torch::Tensor& inputs) {
        return root.predict(inputs);
    }

private:
    SapWood root;
    bool disable_sgd;
};

