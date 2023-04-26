// YetMoreTrees.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <limits>
#include <iostream>
#include <torch/torch.h>

#include "utils.h"

class HeartWood {
public:
    HeartWood(int input_size):
        w(torch::randn({ input_size }, torch::requires_grad(true))),
        k(torch::randn(1, torch::requires_grad(true)))
    {}

    torch::Tensor forward(const torch::Tensor& x) {
        torch::Tensor result = 0.5 + 0.5 * torch::tanh(     (k - (w * x).sum(-1))     );
        return result;
    }

    void train_adam(const torch::Tensor& inputs, const torch::Tensor& targets, int num_epochs, float learning_rate, int batch_size, int patience, float lr_annealing_factor, int verbose = 0) {
        // torch::optim::SGD optimizer({ w, k }, learning_rate);
        torch::optim::Adam optimizer({ w, k }, learning_rate);
        float best_loss = std::numeric_limits<float>::max();
        int no_improvement_counter = 0;

        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            // Create batches using torch::chunk
            auto input_batches = inputs.chunk((inputs.size(0) + batch_size - 1) / batch_size, 0);
            auto target_batches = targets.chunk((targets.size(0) + batch_size - 1) / batch_size, 0);

            float epoch_loss = 0.0;

            for (size_t batch_idx = 0; batch_idx < input_batches.size(); ++batch_idx) {
                // Forward pass
                torch::Tensor input_batch = input_batches[batch_idx];
                torch::Tensor target_batch = target_batches[batch_idx];
                torch::Tensor output_batch = forward(input_batch);

                // Reshape the tensors to the correct dimensions
                output_batch = output_batch.view({ -1, 1 });
                target_batch = target_batch.view({ -1, 1 });

                // Calculate loss
                torch::Tensor loss = torch::binary_cross_entropy(output_batch, target_batch);
                // torch::Tensor loss = torch::mse_loss(output_batch, target_batch);
                epoch_loss += loss.item<float>();

                // Backward pass
                optimizer.zero_grad();
                loss.backward();
                optimizer.step();
            }

            epoch_loss /= input_batches.size();
            if(verbose == 1)
                std::cout << "Epoch: " << epoch + 1 << ", Loss: " << epoch_loss << std::endl;

            // Early stopping
            if (epoch_loss < best_loss) {
                best_loss = epoch_loss;
                no_improvement_counter = 0;
            }
            else {
                no_improvement_counter++;

                if (no_improvement_counter >= patience) {
                    // Apply learning rate annealing
                    learning_rate *= lr_annealing_factor;
                    for (auto& param_group : optimizer.param_groups()) {
                        param_group.options().set_lr(learning_rate);
                    }
                    if(verbose == 1)
                        std::cout << "New learning rate: " << learning_rate << std::endl;

                    // Check if learning rate has been reduced patience times
                    if (no_improvement_counter >= 2 * patience) {
                        if(verbose == 1)
                            std::cout << "Early stopping..." << std::endl;
                        break;
                    }
                }
            }
        }
        this->adam = true;
    }

    void train_impurity(const torch::Tensor& inputs, const torch::Tensor& targets) {

        // Initialize the minimum Gini impurity and best w and k values
        float min_gini_impurity = std::numeric_limits<float>::max();
        torch::Tensor best_w;
        torch::Tensor best_k;

        // Iterate over dimensions
        for (int64_t dim = 0; dim < inputs.size(1); ++dim) {
            // Find the minimum and maximum values in the dimension
            float min_val = inputs.select(1, dim).min().item<float>();
            float max_val = inputs.select(1, dim).max().item<float>();

            // Iterate over possible cut values (k) between the minimum and maximum values
            int num_cuts = 100;
            for (int i = 0; i < num_cuts; ++i) {
                float cut = min_val + (max_val - min_val) * i / (num_cuts - 1);

                // Calculate the Gini impurity for the current cut and dimension
                int num_samples = inputs.size(0);
                int count_left = 0;
                int count_right = 0;
                int count_left_positive = 0;
                int count_right_positive = 0;

                for (int j = 0; j < num_samples; ++j) {
                    float value = inputs[j][dim].item<float>();
                    bool target = targets[j].item<float>() > 0.5;

                    if (value < cut) {
                        count_left++;
                        if (target) {
                            count_left_positive++;
                        }
                    }
                    else {
                        count_right++;
                        if (target) {
                            count_right_positive++;
                        }
                    }
                }

                if (count_left > 0 && count_right > 0) {
                    float gini_impurity = this->gini_impurity(count_left, count_left_positive, count_right, count_right_positive, num_samples);

                    // Update the minimum Gini impurity and best w and k values if necessary
                    if (gini_impurity < min_gini_impurity) {
                        min_gini_impurity = gini_impurity;
                        best_w = torch::zeros({ inputs.size(1) });
                        best_w[dim] = 1.0;
                        best_k = torch::full(1, cut);
                    }
                }
            }
            this->adam = false;
        }

        torch::Tensor preds     = torch::heaviside(best_k - (best_w * inputs).sum(-1), torch::tensor(0.0));
        torch::Tensor inv_preds = torch::heaviside((-best_k) - ((-best_w) * inputs).sum(-1), torch::tensor(0.0));
        
        // Flip lables
        if (utils::accuracy(inv_preds, targets) > utils::accuracy(preds, targets)){
            best_w = -best_w;
            best_k = -best_k;
        }

        // Set the best w and k values
        w = best_w.clone().detach().requires_grad_(false);
        k = best_k.clone().detach().requires_grad_(false);
    }

    void train(const torch::Tensor& inputs, const torch::Tensor& targets, int num_epochs, float learning_rate, int batch_size, int stop_patience, float lr_annealing_factor) {

        // Train using SGD
        this->train_adam(inputs, targets, num_epochs, learning_rate, batch_size, stop_patience, lr_annealing_factor);

        // Calculate the gini impurity for SGD
        auto left_right_counts = count_left_right_positive(inputs, targets);
        float gini_sgd = gini_impurity(left_right_counts[0], left_right_counts[1], left_right_counts[2], left_right_counts[3], inputs.size(0));

        // Save the resulting w and k
        torch::Tensor w_sgd = w.clone();
        torch::Tensor k_sgd = k.clone();

        // Find best split without SGD
        this->train_impurity(inputs, targets);

        // Calculate the gini impurity for No SGD
        left_right_counts = count_left_right_positive(inputs, targets);
        float gini_no_sgd = gini_impurity(left_right_counts[0], left_right_counts[1], left_right_counts[2], left_right_counts[3], inputs.size(0));

        // Choose the values of w and k that minimize the gini impurity
        if ( gini_sgd < gini_no_sgd ) {
            this->w = w_sgd.clone();
            this->k = k_sgd.clone();
            this->adam = true;
        }
    }


    torch::Tensor operator()(const torch::Tensor& x){
        // torch::Tensor result = torch::heaviside(k - torch::dot(w, x), torch::tensor( 0.0 ));
        torch::Tensor result = torch::heaviside(k - (w * x).sum(-1), torch::tensor(0.0));
        return result;
    }

    torch::Tensor get_k() {
        return this->k.clone().detach();
    }

    torch::Tensor get_w() {
        return this->w.clone().detach();
    }

    bool is_adam() {
        return this->adam;
    }

private:
    torch::Tensor w;
    torch::Tensor k;
    bool adam = false

    float gini_impurity(int count_left, int count_left_positive, int count_right, int count_right_positive, int num_samples) {
        float prob_left_positive = static_cast<float>(count_left_positive) / count_left;
        float prob_left_negative = 1.0 - prob_left_positive;
        float prob_right_positive = static_cast<float>(count_right_positive) / count_right;
        float prob_right_negative = 1.0 - prob_right_positive;

        return (count_left * (prob_left_positive * prob_left_negative) +
            count_right * (prob_right_positive * prob_right_negative)) / num_samples;
    }

    std::vector<int> count_left_right_positive(const torch::Tensor& inputs, const torch::Tensor& targets) {
        // TODO: CHECK IF THIS IS RIGHT!!!!!!!!!!!!!!!!!!!!!!!!!!
        torch::Tensor output = forward(inputs);
        torch::Tensor left_mask = output.le(0.5);
        torch::Tensor right_mask = output.gt(0.5);

        int count_left_positive = (left_mask * targets).sum().item<int>();
        int count_right_positive = (right_mask * targets).sum().item<int>();
        int count_left = left_mask.sum().item<int>();
        int count_right = right_mask.sum().item<int>();

        return { count_left, count_left_positive, count_right, count_right_positive };
    }

};

