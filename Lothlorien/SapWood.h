class SapWood {
private:
    HeartWood hw;
    int input_dim;
    std::unique_ptr<SapWood> rights = nullptr;
    std::unique_ptr<SapWood> wrongs = nullptr;

public:
    SapWood(int input_dim) : hw(input_dim), input_dim(input_dim) {}

    void train(const torch::Tensor& inputs, const torch::Tensor& targets, int num_epochs, float learning_rate, int batch_size, int stop_patience, float lr_annealing_factor, int max_depth) {
        // Train the current HeartWood
        hw.train(inputs, targets, num_epochs, learning_rate, batch_size, stop_patience, lr_annealing_factor);

        if (max_depth > 0) {
            // Split the inputs and targets based on the current HeartWood
            torch::Tensor outputs = hw(inputs);
            auto mask_right = outputs.gt(0.5);
            auto mask_wrong = outputs.le(0.5);

            auto inputs_right = inputs.index({mask_right});
            auto targets_right = targets.index({mask_right});

            auto inputs_wrong = inputs.index({mask_wrong});
            auto targets_wrong = targets.index({mask_wrong});

            // Recursively train the right and wrong branches
            rights = std::make_unique<SapWood>(input_dim);
            rights->train(inputs_right, targets_right, num_epochs, learning_rate, batch_size, stop_patience, lr_annealing_factor, max_depth - 1);

            wrongs = std::make_unique<SapWood>(input_dim);
            wrongs->train(inputs_wrong, targets_wrong, num_epochs, learning_rate, batch_size, stop_patience, lr_annealing_factor, max_depth - 1);
        }
    }

    torch::Tensor predict(const torch::Tensor& inputs) {
        if (rights && wrongs) {
            torch::Tensor preds = hw(inputs);
            torch::Tensor mask = preds.eq(torch::ones_like(preds));

            torch::Tensor right_inputs = inputs.masked_select(mask.unsqueeze(1)).reshape({ -1, inputs.size(1) });
            torch::Tensor wrong_inputs = inputs.masked_select(mask.bitwise_not().unsqueeze(1)).reshape({ -1, inputs.size(1) });

            torch::Tensor right_preds = rights->predict(right_inputs);
            torch::Tensor wrong_preds = wrongs->predict(wrong_inputs);

            torch::Tensor preds_new = torch::empty_like(preds);
            preds_new.masked_scatter_(mask, right_preds);
            preds_new.masked_scatter_(mask.bitwise_not(), wrong_preds);

            return preds_new;
        }
        else {
            return hw(inputs);
        }
    }
};
