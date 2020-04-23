using namespace std;
#include <iostream>
#include <sstream>

clock_t clock_array[100] = {0};
string clock_label_array[100];

#define NUM_THREAD 4 //number of threads used

// #define NDEBUG

// #define MNIST_DEBUG 0 //changes debug level

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

// #define MNIST_SHOULD_ENABLE_WEIGHT_DECAY //Uncomment this to disable weight decay.

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

    #ifdef MNIST_SHOULD_ENABLE_WEIGHT_DECAY
        const double lambda_for_weight_decay = 1e-1;
    #endif

}

int main(int argc, char **argv) {

    clock_array[0] = clock();
    clock_label_array[0] = "全体";

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

    clock_array[1] = clock();
    clock_label_array[1] = "コンストラクタ";

    mnist::MNIST m(
                    prm::activation_function_type,
                    prm::loss_function_type,
                    num_node_of_hidden_layer,
                    prm::batch_size,
                    prm::should_normalize_pixel_value,
                    seed,
                    prm::stddev
                  );

    clock_array[1] = clock() - clock_array[1];

    #ifdef MNIST_SHOULD_ENABLE_WEIGHT_DECAY
        m.set_lambda_for_weight_decay_(prm::lambda_for_weight_decay);
    #endif

//     m.load_parameters_("result/weight_and_bias_0_3_25-45.dat");

    if (!m) {
        cout << "Some error occured.\n";
        return 1;
    }

    clock_array[2] = clock();
    clock_label_array[2] = "学習";

    m.training_(epoch,
                #if MNIST_GRADIENT_TYPE == 1 //central difference
                    prm::dx,
                #endif
                prm::learning_rate, prm::opt_type);

    clock_array[2] = clock() - clock_array[2];

    clock_array[3] = clock();
    clock_label_array[3] = "パラメータ保存";

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

    clock_array[3] = clock() - clock_array[3];

    clock_array[4] = clock();
    clock_label_array[4] = "推論";
    const double final_accuracy = m.testing_();
    cout << "Final Accuracy: " << final_accuracy << "(%)\n";
    clock_array[4] = clock() - clock_array[4];

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

    clock_array[0] = clock() - clock_array[0];

    cout << fixed;
    cout.precision(3);
    for (int i = 0; i < sizeof(clock_array) / sizeof(clock_t); ++i) {
        if (clock_array[i] == 0) {
            continue;
        }
        cout.width(2);
        cout << i << ": ";
        cout.width(6);
        cout << (clock_array[i] / static_cast<double>(CLOCKS_PER_SEC)) << " (";
        cout.width(3);
        cout << static_cast<unsigned>(clock_array[i] / static_cast<double>(clock_array[0]) * 100) << "(%)) (" << clock_label_array[i] << ")" << "\n";
    }

    cout.flush();
    cerr.flush();

}

