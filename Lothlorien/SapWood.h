class SapWood {
public:
    SapWood(int input_size) : hw(input_size), input_dim(input_size), is_leaf(false) {}

    void train(const torch::Tensor& inputs, const torch::Tensor& targets, int num_epochs, float learning_rate, int batch_size, int stop_patience, float lr_annealing_factor, int min_samples, int max_depth) {
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

            // TODO: According with https://github.com/ZexinYan/RandomForest-CPP/blob/cc41dc8335ad0b654b954ebdfdc6229a2c7e2729/src/DecisionTree.cpp#L179, when one of the branches does not have enough samples, it is a leaf node. However, they should be handled individually.
            if (inputs_right.size(0) < min_samples || inputs_wrong.size(0) < min_samples){
                is_leaf = true;
            }

            // Recursively train the right and wrong branches only if there's enough data
            if (!is_leaf) {
                rights = std::make_unique<SapWood>(input_dim);
                rights->train(inputs_right, targets_right, num_epochs, learning_rate, batch_size, stop_patience, lr_annealing_factor, min_samples, max_depth - 1);

                wrongs = std::make_unique<SapWood>(input_dim);
                wrongs->train(inputs_wrong, targets_wrong, num_epochs, learning_rate, batch_size, stop_patience, lr_annealing_factor, min_samples, max_depth - 1);
            }

        } else {
            is_leaf = true;
        }
    }

    torch::Tensor predict(const torch::Tensor& inputs) {
        if (is_leaf) {
            return hw(inputs);
        } else {
            torch::Tensor preds = hw(inputs);
            torch::Tensor mask = preds.eq(torch::ones_like(preds));

            torch::Tensor right_inputs = inputs.masked_select(mask.unsqueeze(1)).reshape({ -1, inputs.size(1) });
            torch::Tensor wrong_inputs = inputs.masked_select(mask.bitwise_not().unsqueeze(1)).reshape({ -1, inputs.size(1) });

            torch::Tensor right_preds = rights ? rights->predict(right_inputs) : torch::zeros(right_inputs.size(0));
            torch::Tensor wrong_preds = wrongs ? wrongs->predict(wrong_inputs) : torch::zeros(wrong_inputs.size(0));

            torch::Tensor preds_new = torch::empty_like(preds);
            preds_new.masked_scatter_(mask, right_preds);
            preds_new.masked_scatter_(mask.bitwise_not(), wrong_preds);

            return preds_new;
        }
    }

private:
    HeartWood hw;
    std::unique_ptr<SapWood> rights = nullptr;
    std::unique_ptr<SapWood> wrongs = nullptr;
    bool is_leaf;
    int input_dim;
};
