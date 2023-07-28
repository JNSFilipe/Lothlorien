// YetMoreTrees.cpp : Defines the entry point for the application.
//

#include <gtest/gtest.h>

#include "Lothlorien/DataHandler.h"
#include "Lothlorien/HeartWood.h"
#include "Lothlorien/Mallorn.h"

#define NSAMPLES   1e4
#define NDIMS      3
#define EPOCHS     10
#define LR         0.1
#define BATCH      128
#define PATIENCE   5
#define ANNEALING  0.5
#define MINSAMPLES 10

using namespace std;

TEST(HeartWood, adam_test_positive) {
    torch::Tensor ref_k = 1.0 * torch::tensor(4.5);
    torch::Tensor ref_w = 1.0 * torch::tensor({3.0, 6.0, 0.0});

    auto condition = [=](const torch::Tensor x) -> torch::Tensor { return torch::heaviside(ref_k - torch::dot(ref_w, x), torch::tensor(0.0));};
    torch::Tensor inputs, targets;
    DataHandler dh;
    std::tie(inputs, targets) = dh.synthetize_data(NSAMPLES, NDIMS,  condition);

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

    auto condition = [=](const torch::Tensor x) -> torch::Tensor { return torch::heaviside(ref_k - torch::dot(ref_w, x), torch::tensor(0.0));};
    torch::Tensor inputs, targets;
    DataHandler dh;
    std::tie(inputs, targets) = dh.synthetize_data(NSAMPLES, NDIMS,  condition);

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

    auto condition = [=](const torch::Tensor x) -> torch::Tensor { return torch::heaviside(ref_k - torch::dot(ref_w, x), torch::tensor(0.0));};
    torch::Tensor inputs, targets;
    DataHandler dh;
    std::tie(inputs, targets) = dh.synthetize_data(NSAMPLES, NDIMS,  condition);

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

    auto condition = [=](const torch::Tensor x) -> torch::Tensor { return torch::heaviside(ref_k - torch::dot(ref_w, x), torch::tensor(0.0));};
    torch::Tensor inputs, targets;
    DataHandler dh;
    std::tie(inputs, targets) = dh.synthetize_data(NSAMPLES, NDIMS,  condition);

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

    auto condition = [=](const torch::Tensor x) -> torch::Tensor { return torch::heaviside(ref_k - torch::dot(ref_w, x), torch::tensor(0.0));};
    torch::Tensor inputs, targets;
    DataHandler dh;
    std::tie(inputs, targets) = dh.synthetize_data(NSAMPLES, NDIMS,  condition);

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


TEST(Mallorn, simple_binary_classification_and_reproducibility) {

    const int depth = 3;

    torch::Tensor ref_k = torch::tensor(4.0);
    torch::Tensor ref_w = torch::tensor({ 3.0, 6.0, 0.0 });

    auto condition = [=](const torch::Tensor x) -> torch::Tensor { return torch::heaviside(ref_k - torch::dot(ref_w, x), torch::tensor(0.0));};
    torch::Tensor inputs, targets;
    DataHandler dh;
    std::tie(inputs, targets) = dh.synthetize_data(NSAMPLES, NDIMS,  condition);

    Mallorn tree1(3, false, 42);
    tree1.train(inputs, targets, EPOCHS, LR, BATCH, PATIENCE, ANNEALING, MINSAMPLES, depth);

    Mallorn tree2(3, false, 18);
    tree2.train(inputs, targets, EPOCHS, LR, BATCH, PATIENCE, ANNEALING, MINSAMPLES, depth);

    Mallorn tree3(3, false, 18);
    tree3.train(inputs, targets, EPOCHS, LR, BATCH, PATIENCE, ANNEALING, MINSAMPLES, depth);

    torch::Tensor preds1 = tree1(inputs);
    torch::Tensor preds2 = tree2(inputs);
    torch::Tensor preds3 = tree3(inputs);
    float acc1 = utils::accuracy(preds1, targets);
    float acc2 = utils::accuracy(preds2, targets);
    float acc3 = utils::accuracy(preds3, targets);
    cout << acc1 << endl;
    cout << acc2 << endl;
    cout << acc3 << endl;
    ASSERT_TRUE(acc1 >= 0.8);
    ASSERT_TRUE(acc2 == acc3);
    ASSERT_TRUE(acc1 != acc2);
}

TEST(Mallorn, simple_sgd_vs_no_sgd) {

    torch::Tensor ref_k = torch::tensor(4.0);
    torch::Tensor ref_w = torch::tensor({ 3.0, 6.0, 0.0 });

    auto condition = [=](const torch::Tensor x) -> torch::Tensor { return torch::heaviside(ref_k - torch::dot(ref_w, x), torch::tensor(0.0));};
    torch::Tensor inputs, targets;
    DataHandler dh;
    std::tie(inputs, targets) = dh.synthetize_data(NSAMPLES, NDIMS,  condition);

    Mallorn tree_sgd(3, false);
    tree_sgd.train(inputs, targets, EPOCHS, LR, BATCH, PATIENCE, ANNEALING, MINSAMPLES, 2);

    Mallorn tree_no_sgd(3, true);
    tree_no_sgd.train(inputs, targets, EPOCHS, LR, BATCH, PATIENCE, ANNEALING, MINSAMPLES, 2);

    torch::Tensor preds_sgd = tree_sgd(inputs);
    torch::Tensor preds_no_sgd = tree_no_sgd(inputs);
    float acc_sgd    = utils::accuracy(preds_sgd, targets);
    float acc_no_sgd = utils::accuracy(preds_no_sgd, targets);

    cout << "Acc. SGD\t: \t" << acc_sgd*100 << "%" << endl;
    cout << "Acc. No SGD\t: \t" << acc_no_sgd*100 << "%" << endl;

    ASSERT_TRUE(acc_sgd > acc_no_sgd);
}

TEST(DataHandler, read_csv_high_depth) {
    torch::Tensor inputs, targets;
    DataHandler dh;
    std::tie(inputs, targets) = dh.read_data_from_csv("../../Assets/data_banknote_authentication.txt");

    SUCCEED();

    Mallorn tree((int) inputs.size(1) , false);
    tree.train(inputs, targets, EPOCHS, LR, BATCH, PATIENCE, ANNEALING, MINSAMPLES, 100);

    torch::Tensor preds = tree(inputs);
    float acc = utils::accuracy(preds, targets);

    cout << "Acc. SGD\t: \t" << acc*100 << "%" << endl;

    ASSERT_TRUE(0.9 < acc);
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// NAMES FOR CLASSES IN THE FUTURE:
// - Galahd
// - Mallorn
// - Taur
