using namespace std;
#include <iostream>
#include <sstream>

#define NUM_THREAD 4 //number of threads used

// #define NDEBUG

#define MNIST_DEBUG 0 //changes debug level

// #define MNIST_GRADIENT_CHECK //does gradient check periodically

//How gradient is calculated.
//0: backpropagation
//1: central difference
#define MNIST_GRADIENT_TYPE 0

//How weights are initialized.
//0: N(0, 1) multiplied by `stddev`.
//1: Xavier initialization
//2: He initialization
#define MNIST_WEIGHT_INITIALIZATION_METHOD 2

#define MNIST_ENABLE_BATCH_NORMALIZATION 1 //1 enables batch normalization and 0 disables it.

#include "header/mnist.h"

namespace prm {
    
//     mnist::MNIST::activation_function_type_ activation_function_type = mnist::MNIST::sigmoid_;
    mnist::MNIST::activation_function_type_ activation_function_type = mnist::MNIST::relu_;

    mnist::MNIST::loss_function_type_ loss_function_type = mnist::MNIST::cross_entropy_;

//     optimizer::optimizer_type opt_type = optimizer::t_gradient_descent;
//     optimizer::optimizer_type opt_type = optimizer::t_momentum;
    optimizer::optimizer_type opt_type = optimizer::t_adagrad;

    const vector<unsigned> num_node_of_hidden_layer({50, 30});

    const unsigned batch_size = 100;

    const bool should_normalize_pixel_value = true;

    const unsigned seed = 1;

    const double stddev = 0.01;

    const unsigned epoch = 16;

    const double dx = 1e-2;

    const double learning_rate = 1e-2;

}

int main(int argc, char **argv) {

    //command-line arguments {

    unsigned seed = prm::seed;
    unsigned epoch = prm::epoch;
    vector<unsigned> num_node_of_hidden_layer = prm::num_node_of_hidden_layer;

    if (argc == 1) {
        cout << "Usage: mnist.out <seed> [<epoch> [<num_node_of_hidden_layer>...] ]\n";
        return 1;
    }

    if (argc >= 2) {
        seed = atoi(argv[1]);
    }
    if (argc >= 3) {
        epoch = atoi(argv[2]);
    }
    if (argc >= 4) {
        num_node_of_hidden_layer.clear();
        for (int i = 3; i < argc; ++i) {
            num_node_of_hidden_layer.push_back(atoi(argv[i]));
        }
    }

    #if MNIST_DEBUG != 0
        cout << "seed = " << seed << "\n";
        cout << "epoch = " << epoch << "\n";
        cout << "num_node_of_hidden_layer = ";
        vector_operation::print_array(num_node_of_hidden_layer);
        cout << "\n";
    #endif

    //} command-line arguments

    mnist::MNIST m(
                    prm::activation_function_type,
                    prm::loss_function_type,
                    num_node_of_hidden_layer,
                    prm::batch_size,
                    prm::should_normalize_pixel_value,
                    seed,
                    prm::stddev
                  );

//     m.load_parameters_("result/weight_and_bias_0_3_25-45.dat");

    if (!m) {
        cout << "Some error occured.\n";
        return 1;
    }

    m.training_(epoch, prm::dx, prm::learning_rate, prm::opt_type);

    //Saves the parameters to the file.
    {
        ostringstream oss;
        oss << "result/weight_and_bias"
            << "_" << seed
            << "_" << epoch
            << "_" << num_node_of_hidden_layer[0];
        for (int i = 1; i < num_node_of_hidden_layer.size(); ++i) {
            oss << "-" << num_node_of_hidden_layer[i];
        }
        oss << ".dat";

        m.save_parameters_(oss.str().c_str());

        if (!m) {
            cout << "Some error occured.\n";
            return 1;
        }
        cout << "\nSaved the weights and the biases to [ " << oss.str() << " ].\n";
    }

    const double final_accuracy = m.testing_();
    cout << "Final Accuracy: " << final_accuracy << "(%)\n";

    #if MNIST_DEBUG != 0

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

    #endif

    cout.flush();
    cerr.flush();

}

