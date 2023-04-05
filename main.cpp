// YetMoreTrees.cpp : Defines the entry point for the application.
//

#include <gtest/gtest.h>

#include "Lothlorien/HeartWood.h"

#define EPOCHS    10
#define LR        0.1
#define BATCH     128
#define PATIENCE  5
#define ANNEALING 0.5

using namespace std;

TEST(HeartWood, adam_test_positive) {
    torch::Tensor ref_k = 1.0 * torch::tensor(4.5);
    torch::Tensor ref_w = 1.0 * torch::tensor({3.0, 6.0, 0.0});
    const unsigned int n_samples = 1e4;

    vector<torch::Tensor> inputs_vec, targets_vec;

    for (int i = 0; i < n_samples; i++)
        inputs_vec.push_back(torch::rand(3));

    for (int i = 0; i < n_samples; i++)
        targets_vec.push_back(torch::heaviside(ref_k - torch::dot(ref_w, inputs_vec[i]), torch::tensor(0.0)));

    torch::Tensor inputs = torch::stack(inputs_vec);
    torch::Tensor targets = torch::stack(targets_vec);

    cout << "Percentage of 1's:\t" << (targets.sum(-1) / targets.size(0)) * 100 << endl << endl;

    HeartWood heart_wood(3);
    heart_wood.train_adam(inputs, targets, EPOCHS, LR, BATCH, PATIENCE, ANNEALING);

    SUCCEED();

    torch::Tensor w = heart_wood.get_w();
    torch::Tensor k = heart_wood.get_k();

    k = k/torch::max(torch::abs(w));
    w = w/torch::max(torch::abs(w));

    ref_k = ref_k / torch::max(torch::abs(ref_w));
    ref_w = ref_w / torch::max(torch::abs(ref_w));

    cout << "Ref k\t: \t" << std::setprecision(2) << ref_k.item<float>() << endl;
    cout << "k\t: \t"     << std::setprecision(2) << k.item<float>()     << endl;

    cout << "Ref w\t: \t";
    vector<float> v_ref_w(ref_w.data_ptr<float>(), ref_w.data_ptr<float>() + ref_w.numel());
    for (const auto & i : v_ref_w) cout << i << "\t";
    cout << endl;

    cout << "w\t: \t";
    vector<float> v_w(w.data_ptr<float>(), w.data_ptr<float>() + w.numel());
    for (const auto & i : v_w) cout << i << "\t";
    cout << endl;

    // Compare k
    EXPECT_NEAR(ref_k.item<float>(), k.item<float>(), 0.1);

    // Compare w
    for (int i = 0; i < v_ref_w.size(); i++)
        EXPECT_NEAR(v_ref_w[i], v_w[i], 0.1);
}


TEST(HeartWood, adam_test_negative) {
    torch::Tensor ref_k = -1.0 * torch::tensor(4.5);
    torch::Tensor ref_w = -1.0 * torch::tensor({ 3.0, 6.0, 0.0 });
    const unsigned int n_samples = 1e4;

    vector<torch::Tensor> inputs_vec, targets_vec;

    for (int i = 0; i < n_samples; i++)
        inputs_vec.push_back(torch::rand(3));

    for (int i = 0; i < n_samples; i++)
        targets_vec.push_back(torch::heaviside(ref_k - torch::dot(ref_w, inputs_vec[i]), torch::tensor(0.0)));

    torch::Tensor inputs = torch::stack(inputs_vec);
    torch::Tensor targets = torch::stack(targets_vec);

    cout << "Percentage of 1's:\t" << (targets.sum(-1) / targets.size(0)) * 100 << endl << endl;

    HeartWood heart_wood(3);
    heart_wood.train_adam(inputs, targets, EPOCHS, LR, BATCH, PATIENCE, ANNEALING);

    SUCCEED();

    torch::Tensor w = heart_wood.get_w();
    torch::Tensor k = heart_wood.get_k();

    k = k / torch::max(torch::abs(w));
    w = w / torch::max(torch::abs(w));

    ref_k = ref_k / torch::max(torch::abs(ref_w));
    ref_w = ref_w / torch::max(torch::abs(ref_w));

    cout << "Ref k\t: \t" << std::setprecision(2) << ref_k.item<float>() << endl;
    cout << "k\t: \t" << std::setprecision(2) << k.item<float>() << endl;

    cout << "Ref w\t: \t";
    vector<float> v_ref_w(ref_w.data_ptr<float>(), ref_w.data_ptr<float>() + ref_w.numel());
    for (const auto& i : v_ref_w) cout << i << "\t";
    cout << endl;

    cout << "w\t: \t";
    vector<float> v_w(w.data_ptr<float>(), w.data_ptr<float>() + w.numel());
    for (const auto& i : v_w) cout << i << "\t";
    cout << endl;

    // Compare k
    EXPECT_NEAR(ref_k.item<float>(), k.item<float>(), 0.1);

    // Compare w
    for (int i = 0; i < v_ref_w.size(); i++)
        EXPECT_NEAR(v_ref_w[i], v_w[i], 0.1);
}


TEST(HeartWood, impurity_test_positive) {
    torch::Tensor ref_k = 1.0 * torch::tensor(0.18);
    torch::Tensor ref_w = 1.0 * torch::tensor({ 0.0, 0.0, 1.0 });
    const unsigned int n_samples = 1e4;

    vector<torch::Tensor> inputs_vec, targets_vec;

    for (int i = 0; i < n_samples; i++)
        inputs_vec.push_back(torch::rand(3));

    for (int i = 0; i < n_samples; i++)
        targets_vec.push_back(torch::heaviside(ref_k - torch::dot(ref_w, inputs_vec[i]), torch::tensor(0.0)));

    torch::Tensor inputs = torch::stack(inputs_vec);
    torch::Tensor targets = torch::stack(targets_vec);

    HeartWood heart_wood(3);
    heart_wood.train_impurity(inputs, targets);

    SUCCEED();

    torch::Tensor w = heart_wood.get_w();
    torch::Tensor k = heart_wood.get_k();

    k = k / torch::max(torch::abs(w));
    w = w / torch::max(torch::abs(w));

    ref_k = ref_k / torch::max(torch::abs(ref_w));
    ref_w = ref_w / torch::max(torch::abs(ref_w));

    cout << "Ref k\t: \t" << std::setprecision(2) << ref_k.item<float>() << endl;
    cout << "k\t: \t" << std::setprecision(2) << k.item<float>() << endl;

    cout << "Ref w\t: \t";
    vector<float> v_ref_w(ref_w.data_ptr<float>(), ref_w.data_ptr<float>() + ref_w.numel());
    for (const auto& i : v_ref_w) cout << i << "\t";
    cout << endl;

    cout << "w\t: \t";
    vector<float> v_w(w.data_ptr<float>(), w.data_ptr<float>() + w.numel());
    for (const auto& i : v_w) cout << i << "\t";
    cout << endl;

    // Compare k
    EXPECT_NEAR(ref_k.item<float>(), k.item<float>(), 0.1);

    // Compare w
    for (int i = 0; i < v_ref_w.size(); i++)
        EXPECT_NEAR(v_ref_w[i], v_w[i], 0.1);
}


TEST(HeartWood, impurity_test_negative) {
    torch::Tensor ref_k = -1.0 * torch::tensor(0.18);
    torch::Tensor ref_w = -1.0 * torch::tensor({ 0.0, 0.0, 1.0 });
    const unsigned int n_samples = 1e4;

    vector<torch::Tensor> inputs_vec, targets_vec;

    for (int i = 0; i < n_samples; i++)
        inputs_vec.push_back(torch::rand(3));

    for (int i = 0; i < n_samples; i++)
        targets_vec.push_back(torch::heaviside(ref_k - torch::dot(ref_w, inputs_vec[i]), torch::tensor(0.0)));

    torch::Tensor inputs = torch::stack(inputs_vec);
    torch::Tensor targets = torch::stack(targets_vec);

    HeartWood heart_wood(3);
    heart_wood.train_impurity(inputs, targets);

    SUCCEED();

    torch::Tensor w = heart_wood.get_w();
    torch::Tensor k = heart_wood.get_k();

    k = k / torch::max(torch::abs(w));
    w = w / torch::max(torch::abs(w));

    ref_k = ref_k / torch::max(torch::abs(ref_w));
    ref_w = ref_w / torch::max(torch::abs(ref_w));

    cout << "Ref k\t: \t" << std::setprecision(2) << ref_k.item<float>() << endl;
    cout << "k\t: \t" << std::setprecision(2) << k.item<float>() << endl;

    cout << "Ref w\t: \t";
    vector<float> v_ref_w(ref_w.data_ptr<float>(), ref_w.data_ptr<float>() + ref_w.numel());
    for (const auto& i : v_ref_w) cout << i << "\t";
    cout << endl;

    cout << "w\t: \t";
    vector<float> v_w(w.data_ptr<float>(), w.data_ptr<float>() + w.numel());
    for (const auto& i : v_w) cout << i << "\t";
    cout << endl;

    // Compare k
    EXPECT_NEAR(ref_k.item<float>(), k.item<float>(), 0.1);

    // Compare w
    for (int i = 0; i < v_ref_w.size(); i++)
        EXPECT_NEAR(v_ref_w[i], v_w[i], 0.1);
}


TEST(HeartWood, full_train_test) {
    torch::Tensor ref_k = torch::tensor(4.0);
    torch::Tensor ref_w = torch::tensor({ 3.0, 6.0, 0.0 });
    const unsigned int n_samples = 1e4;

    vector<torch::Tensor> inputs_vec, targets_vec;

    for (int i = 0; i < n_samples; i++)
        inputs_vec.push_back(torch::rand(3));

    for (int i = 0; i < n_samples; i++)
        targets_vec.push_back(torch::heaviside(ref_k - torch::dot(ref_w, inputs_vec[i]), torch::tensor(0.0)));

    torch::Tensor inputs = torch::stack(inputs_vec);
    torch::Tensor targets = torch::stack(targets_vec);

    HeartWood heart_wood(3);
    heart_wood.train(inputs, targets, EPOCHS, LR, BATCH, PATIENCE, ANNEALING);

    SUCCEED();

    torch::Tensor w = heart_wood.get_w();
    torch::Tensor k = heart_wood.get_k();

    k = k / torch::max(torch::abs(w));
    w = w / torch::max(torch::abs(w));

    ref_k = ref_k / torch::max(torch::abs(ref_w));
    ref_w = ref_w / torch::max(torch::abs(ref_w));

    cout << "Ref k\t: \t" << std::setprecision(2) << ref_k.item<float>() << endl;
    cout << "k\t: \t" << std::setprecision(2) << k.item<float>() << endl;

    cout << "Ref w\t: \t";
    vector<float> v_ref_w(ref_w.data_ptr<float>(), ref_w.data_ptr<float>() + ref_w.numel());
    for (const auto& i : v_ref_w) cout << i << "\t";
    cout << endl;

    cout << "w\t: \t";
    vector<float> v_w(w.data_ptr<float>(), w.data_ptr<float>() + w.numel());
    for (const auto& i : v_w) cout << i << "\t";
    cout << endl;

    // Compare k
    EXPECT_NEAR(ref_k.item<float>(), k.item<float>(), 0.1);

    // Compare w
    for (int i = 0; i < v_ref_w.size(); i++)
        EXPECT_NEAR(v_ref_w[i], v_w[i], 0.1);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// NAMES FOR CLASSES IN THE FUTURE:
// - Galahd
// - Mallorn
// - Taur
