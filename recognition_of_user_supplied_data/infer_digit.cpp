//This program reads 28*28=784 real values, interprets them as a single image, infers the true digit and exits.
//This program is intented to be internally called from `draw_digit.py`.

using namespace std;
#include <iostream>
#include <sstream>

#define NUM_THREAD 4

#define NDEBUG

#include "../header/mnist.h"

namespace cnst {

    constexpr unsigned num_pixel_per_image = 28 * 28;

}

namespace prm {
    
    mnist::MNIST::activation_function_type_ activation_function_type = mnist::MNIST::sigmoid_;

    mnist::MNIST::loss_function_type_ loss_function_type = mnist::MNIST::cross_entropy_;

    const vector<unsigned> num_node_of_hidden_layer({300});

    const char *parameter_file = "../result/seed_0_epoch_50/weight_and_bias_0_50_300.dat";

    //unused parameters
    const unsigned batch_size = 100;
    const bool should_normalize_pixel_value = true;
    const unsigned seed = 1;
    const double scale = 1;
    const unsigned epoch = 50;
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
                    prm::scale,
                    /* should_skip_initialization = */ true
                  );

    m.load_weight_and_bias_(prm::parameter_file);

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

