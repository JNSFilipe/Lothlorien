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

    std::tuple<torch::Tensor, torch::Tensor> read_data_from_csv(const std::string& filename, const char delimiter = ',', int target_column = -1, int verbose = 1) {
        std::vector<std::vector<float>> data_vec;

        // Open the CSV file
        std::ifstream file(filename);
        if (!file) {
            throw std::runtime_error("Error opening file: " + filename);
        }

        // Read data from CSV line by line
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string cell;
            std::vector<float> row_values;
            while (std::getline(iss, cell, delimiter)) {
                // Assuming the values in the CSV are floating-point numbers
                row_values.push_back(std::stof(cell));
            }
            data_vec.push_back(row_values);
        }

        file.close();

        // Determine the number of columns in the CSV (assuming each row has the same number of columns)
        unsigned int num_columns = data_vec[0].size();
        unsigned int num_rows = data_vec.size();

        // Use last column as target by default
        target_column = target_column < 0 ? num_columns - 1 : target_column;

        // Create Torch tensors for inputs and targets
        torch::TensorOptions options(torch::kFloat32);
        torch::Tensor inputs = torch::empty({static_cast<long>(num_rows), static_cast<long>(num_columns - 1)}, options);
        torch::Tensor targets = torch::empty({static_cast<long>(num_rows)}, options);

        // Populate tensors with data from data_vec
        for (unsigned int i = 0; i < num_rows; ++i) {
            for (unsigned int j = 0; j < num_columns; ++j) {
                if (static_cast<int>(j) != target_column) {
                    inputs[i][j - (j > target_column)] = data_vec[i][j];
                    //inputs[i][j] = data_vec[i][j];
                }
                else {
                    targets[i] = data_vec[i][j];
                }
            }
        }

        if (verbose > 0) {
            std::cout << "Percentage of 1's:\t" << (targets.sum(-1) / targets.size(0)) * 100 << std::endl << std::endl;
        }

        return std::make_tuple(inputs, targets);
    }

private:

};