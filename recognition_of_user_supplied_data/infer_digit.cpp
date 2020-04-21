//This program reads 28*28=784 real values, interprets them as a single image, infers the true digit and exits.
//This program is intented to be internally called from `draw_digit.py`.

using namespace std;
#include <iostream>
#include <sstream>

#define NUM_THREAD 4

// #define NDEBUG
#define MNIST_DEBUG 0

#define MNIST_ENABLE_BATCH_NORMALIZATION 1

//unused macros
#define MNIST_GRADIENT_TYPE 0
#define MNIST_WEIGHT_INITIALIZATION_METHOD 0

#include "../header/mnist.h"

namespace cnst {

    constexpr unsigned num_pixel_per_image = 28 * 28;

}

namespace prm {
    
    mnist::MNIST::activation_function_type_ activation_function_type = mnist::MNIST::relu_;

    mnist::MNIST::loss_function_type_ loss_function_type = mnist::MNIST::cross_entropy_;

    const vector<unsigned> num_node_of_hidden_layer({100, 100, 100, 100, 100});

    const char *parameter_file = "../result/weight_and_bias_0_25_100-100-100-100-100.dat";

    //unused parameters
    const unsigned batch_size = 100;
    const bool should_normalize_pixel_value = true;
    const unsigned seed = 1;
    const double scale = 1;
    const unsigned epoch = 3;
    const double dx = 1e-2;
    const double learning_rate = 1e-1;
    optimizer::optimizer_type opt_type = optimizer::t_adagrad;

}

int main() {
    
    mnist::cnst::dataset_directory = "../read_mnist/data/";

    mnist::MNIST m(
                    prm::activation_function_type,
                    prm::loss_function_type,
                    prm::num_node_of_hidden_layer,
                    prm::batch_size,
                    prm::should_normalize_pixel_value,
                    prm::seed,
                    prm::scale,
                    /* should_skip_initialization = */ 0
                  );

    m.load_parameters_(prm::parameter_file);

//     //for debug (requirements: `should_skip_initialization` == 0)
//     m.training_(prm::epoch, prm::dx, prm::learning_rate, prm::opt_type);
//     const double accuracy = m.testing_();
//     cout << "Accuracy: " << accuracy << "(%)\n";

    if (!m) {
        cerr << "Some error occured.\n";
        cout << "-1\n";
        return 1;
    }

    vector<double> pixel_array;
    for (int i = 0; i < cnst::num_pixel_per_image; ++i) {
        double pixel;
        cin >> pixel;
        if (!cin) {
            break;
        }
        pixel_array.push_back(pixel);
    }
    if (!cin) {
        cerr << "Input was invalid.\n";
        cout << "Err" << "\n";
    } else {
        cout << m.infer_(vector<vector<double>>(1, pixel_array))[0] << "\n";
    }

}

