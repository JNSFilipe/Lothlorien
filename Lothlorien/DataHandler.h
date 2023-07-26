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
        std::vector<torch::Tensor> inputs_vec, targets_vec;

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
            int column = 0;
            while (std::getline(iss, cell, delimiter)) {
                if (column != target_column) {
                    // Assuming the values in the CSV are floating-point numbers
                    row_values.push_back(std::stof(cell));
                }
                column++;
            }

            // Create a Torch Tensor from the row values (excluding the target column)
            torch::Tensor input_tensor = torch::from_blob(row_values.data(), {1, static_cast<long>(row_values.size())});

            // Get the target value from the target column
            torch::Tensor target_tensor;
            if (target_column >= 0) {
                std::istringstream target_iss(line);
                std::string target_cell;
                int target_column_count = 0;
                while (std::getline(target_iss, target_cell, delimiter)) {
                    if (target_column_count == target_column) {
                        // Assuming the target value is a floating-point number
                        float target_value = std::stof(target_cell);
                        target_tensor = torch::tensor({target_value});
                        break;
                    }
                    target_column_count++;
                }
            }

            inputs_vec.push_back(input_tensor);
            targets_vec.push_back(target_tensor);
        }

        file.close();

        torch::Tensor inputs = torch::cat(inputs_vec, 0);    // Concatenate tensors along the first dimension
        torch::Tensor targets = torch::cat(targets_vec, 0);  // Concatenate tensors along the first dimension

        if (verbose > 0) {
            std::cout << "Percentage of 1's:\t" << (targets.sum(-1) / targets.size(0)) * 100 << std::endl << std::endl;
        }

        return std::make_tuple(inputs, targets);
    }


private:

};