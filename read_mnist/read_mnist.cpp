using namespace std;
#include "./read_mnist.h"

namespace N {

    template <typename T>
    void print_array(const vector<T> &v) {
        if (v.empty()) {
            cout << "[]";
            return;
        }
        cout << "[" << v[0];
        for (int i = 1; i < v.size(); ++i) {
            cout << ", " << v[i];
        }
        cout << "]\n";
    }

    const string parent_directory = "./data/";

}

int main() {

    //loads MNIST dataset {

    vector<vector<int>> image_train;
    vector<int> label_train;
    vector<vector<int>> label_train_one_hot;
    vector<vector<int>> image_test;
    vector<vector<int>> label_test_one_hot;
    vector<int> label_test;

    if (!read_mnist::read_image_train(image_train, N::parent_directory)) {
        return 1;
    }
    if (!read_mnist::read_label_train(label_train, N::parent_directory)) {
        return 1;
    }
    if (!read_mnist::read_label_train_one_hot(label_train_one_hot, N::parent_directory)) {
        return 1;
    }
    if (!read_mnist::read_image_test(image_test, N::parent_directory)) {
        return 1;
    }
    if (!read_mnist::read_label_test(label_test, N::parent_directory)) {
        return 1;
    }
    if (!read_mnist::read_label_test_one_hot(label_test_one_hot, N::parent_directory)) {
        return 1;
    }

    //} loads MNIST dataset

    cout << "image_train.size(): " << image_train.size() << "\n";
    cout << "image_train[0].size(): " << image_train[0].size() << "\n";
    cout << "--- image_train[0] ---\n";
    N::print_array(image_train[0]);
    cout << "--- image_train[59999] ---\n";
    N::print_array(image_train[59999]);

    cout << "\n";

    cout << "label_train.size(): " << label_train.size() << "\n";
    cout << "label_train[0]: " << label_train[0] << "\n";
    cout << "label_train[59999]: " << label_train[59999] << "\n";

    cout << "\n";

    cout << "label_train_one_hot.size(): " << label_train_one_hot.size() << "\n";
    cout << "--- label_train_one_hot[0] ---\n";
    N::print_array(label_train_one_hot[0]);
    cout << "--- label_train_one_hot[59999] ---\n";
    N::print_array(label_train_one_hot[59999]);

    cout << "\n";

    cout << "image_test.size(): " << image_test.size() << "\n";
    cout << "image_test[0].size(): " << image_test[0].size() << "\n";
    cout << "--- image_test[0] ---\n";
    N::print_array(image_test[0]);
    cout << "--- image_test[9999] ---\n";
    N::print_array(image_test[9999]);

    cout << "\n";

    cout << "label_test.size(): " << label_test.size() << "\n";
    cout << "label_test[0]: " << label_test[0] << "\n";
    cout << "label_test[9999]: " << label_test[9999] << "\n";

    cout << "\n";

    cout << "label_test_one_hot.size(): " << label_test_one_hot.size() << "\n";
    cout << "--- label_test_one_hot[0] ---\n";
    N::print_array(label_test_one_hot[0]);
    cout << "--- label_test_one_hot[9999] ---\n";
    N::print_array(label_test_one_hot[9999]);

}

