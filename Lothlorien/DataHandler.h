// DataHandler.h : Include file for the DataHandler class

#pragma once

#include <tuple>
#include <functional>
#include <torch/torch.h>

class DataHandler {
public:
//    DataHandler():
//    {}

    std::tuple<torch::Tensor, torch::Tensor> synthetize_data(unsigned int n_samples, unsigned int n_dims, std::function<torch::Tensor(torch::Tensor)> condition, int verbose=1){

        std::vector<torch::Tensor> inputs_vec, targets_vec;

        for (int i = 0; i < n_samples; i++)
            inputs_vec.push_back(torch::rand(n_dims));

        for (int i = 0; i < n_samples; i++)
            targets_vec.push_back(condition(inputs_vec[i]));

        torch::Tensor inputs = torch::stack(inputs_vec);
        torch::Tensor targets = torch::stack(targets_vec);

        if(verbose > 0){
            std::cout << "Percentage of 1's:\t" << (targets.sum(-1) / targets.size(0)) * 100 << std::endl << std::endl;
        }

        return std::make_tuple(inputs, targets);

    }


private:
};