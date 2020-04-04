using namespace std;
#include <iostream>

#define NUM_THREAD 4

// #define NDEBUG

#define MNIST_DEBUG 3
// #define MNIST_GRADIENT_CHECK

//0: backpropagation
//1: central difference
// #define MNIST_GRADIENT_TYPE 0

#include "header/mnist.h"

namespace prm {
    
    mnist::MNIST::activation_function_type_ activation_function_type = mnist::MNIST::sigmoid_;
//     mnist::MNIST::activation_function_type_ activation_function_type = mnist::MNIST::relu_;

    mnist::MNIST::loss_function_type_ loss_function_type = mnist::MNIST::cross_entropy_;

//     const vector<unsigned> num_node_of_hidden_layer({50, 30});
//     const vector<unsigned> num_node_of_hidden_layer({50});
    const vector<unsigned> num_node_of_hidden_layer({10});
//     const vector<unsigned> num_node_of_hidden_layer({10, 10});

    const unsigned batch_size = 100;

    const bool should_normalize_pixel_value = true;

    const unsigned seed = -1;

    const double scale = 1;

    const unsigned epoch = 1;

    const double dx = 1e-2;

    const double learning_rate = 1e-1;

}

int main() {

    mnist::MNIST m(
                    prm::activation_function_type,
                    prm::loss_function_type,
                    prm::num_node_of_hidden_layer,
                    prm::batch_size,
                    prm::should_normalize_pixel_value,
                    prm::seed,
                    prm::scale
                  );

    m.training_(prm::epoch, prm::dx, prm::learning_rate);

    const double final_accuracy = m.testing_();
    cout << "Final Accuracy: " << final_accuracy << "(%)\n";

    cout << "\n\n#Loss Function History\n";
    const vector<double> loss_function_history = m.get_loss_function_history_();
    for (int i = 0; i < loss_function_history.size(); ++i) {
        cout << i << " " << loss_function_history[i] << "\n";
    }

    cout << "\n\n#Accuracy History for Training Data\n";
    const vector<double> accuracy_history_for_training_data = m.get_accuracy_history_for_training_data_();
    for (int i = 0; i < accuracy_history_for_training_data.size(); ++i) {
        cout << i << " " << accuracy_history_for_training_data[i] << "\n";
    }

    cout << "\n\n#Accuracy History for Testing Data\n";
    const vector<double> accuracy_history_for_testing_data = m.get_accuracy_history_for_testing_data_();
    for (int i = 0; i < accuracy_history_for_testing_data.size(); ++i) {
        cout << i << " " << accuracy_history_for_testing_data[i] << "\n";
    }

}

