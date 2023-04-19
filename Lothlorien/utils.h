// utils.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <torch/torch.h>

namespace utils{

    float accuracy(const torch::Tensor& predictions, const torch::Tensor& targets) {

        auto correct = predictions.eq(targets).sum().item<int64_t>();
        // Calculate the total number of predictions
        auto total = targets.size(0);
        // Compute the accuracy
        float accuracy = static_cast<float>(correct) / static_cast<float>(total);
        return accuracy;
    }
        
}

