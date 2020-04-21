//Requirements
//1. `MNIST_GRADIENT_TYPE` macro shall be defined. `0` uses backpropagation to calculate gradient and `1` uses central difference.
//2. `MNIST_DEBUG` macro is optionally defined. Then debugging information is additionally output.
//3. `MNIST_GRADIENT_CHECK` macro is optionally defined. Then gradient check is done periodically. This is only useful when you'd like to check if the implementation is correct.
//4. `MNIST_WEIGHT_INITIALIZATION_METHOD` macro is optionally defined, which determines how weights are initialized. `0` uses N(0, 1) multiplied by `stddev`, `1` performs Xavier initialization and `2` performs He initialization.
//5. `MNIST_ENABLE_BATCH_NORMALIZATION` macro shall be defined. `0` disables batch normalization and `1` enables it.

#ifndef is_mnist_included

    #define is_mnist_included

    #include <iostream>
    #include "./vector_operation.h"
    #include "./misc.h"
    #include "./layer.h"
    #include "./optimizer.h"
    #include "../read_mnist/read_mnist.h"
    #include <memory>

    namespace mnist {

        using namespace std;
        using namespace vector_operation;

        namespace cnst {

            constexpr unsigned num_input_node = 28 * 28;
            constexpr unsigned num_output_node = 10;

            constexpr double max_pixel_value = 255;

            string dataset_directory = "./read_mnist/data/";

            //The number of layers which as a whole constitute a single hidden layer.
            //For example, if each hidden layer consists of affine transformation, batch normalization and activation, then the value is 3.
            //If we omit batch normalization, then the value is 2.
            #if MNIST_ENABLE_BATCH_NORMALIZATION == 0
                constexpr unsigned num_layer_in_hidden_layer = 2;
            #else
                constexpr unsigned num_layer_in_hidden_layer = 3;
            #endif

        }

        class MNIST {

            public:

                enum activation_function_type_ {sigmoid_, relu_};
                enum loss_function_type_ {mean_squared_, cross_entropy_};

            private:

                vector<vector<double>> image_train_;
                vector<int> label_train_;
//                 vector<vector<int>> label_train_one_hot_;

                vector<vector<double>> image_test_;
//                 vector<vector<int>> label_test_one_hot_;
                vector<int> label_test_;

                const vector<unsigned> num_node_of_hidden_layer_;

                vector<vector<vector<double>>> weight_;
                vector<vector<double>> bias_;

                vector<layer::Layer *> layer_;
                layer::LastLayer *last_layer_;

                const unsigned batch_size_;

                misc::Rand rand_;

                vector<double> loss_function_history_;
                vector<double> accuracy_history_for_training_data_;
                vector<double> accuracy_history_for_testing_data_;

                bool error_flag_ = false;

            private:

                bool read_mnist_dataset_(bool should_normalize_pixel_value) {

                    const bool error_flag_old = error_flag_;
                    error_flag_ = true;

                    if (!read_mnist::read_image_train(image_train_, cnst::dataset_directory)) {
                        return false;
                    }
                    if (!read_mnist::read_label_train(label_train_, cnst::dataset_directory)) {
                        return false;
                    }
//                     if (!read_mnist::read_label_train_one_hot(label_train_one_hot_, cnst::dataset_directory)) {
//                         return false;
//                     }
                    if (!read_mnist::read_image_test(image_test_, cnst::dataset_directory)) {
                        return false;
                    }
                    if (!read_mnist::read_label_test(label_test_, cnst::dataset_directory)) {
                        return false;
                    }
//                     if (!read_mnist::read_label_test_one_hot(label_test_one_hot_, cnst::dataset_directory)) {
//                         return false;
//                     }

                    if (should_normalize_pixel_value) {
                        image_train_ /= cnst::max_pixel_value;
                        image_test_ /= cnst::max_pixel_value;
                    }

                    error_flag_ = error_flag_old;

                    return true;

                }

                void initialize_weight_and_bias_(double stddev, bool should_skip_initialization = false) {

                    //initializes `weight_` {

                    mt19937 &mt = rand_.get_rand_generator_();

                    //Generally, if a stochastic variable X follows the normal distribution N(a, b), then c*X also follows N(c*a, |c|*b) where c is an arbitrary constant.
                    //Thus we can dynamically change the value of standard deviation by setting it to unity here and later multiplying an arbitrary value to the generated values.
                    normal_distribution<double> normal_dist(0, 1);

                    const unsigned num_layer_with_weight = num_node_of_hidden_layer_.size() + 1;

                    weight_ = vector<vector<vector<double>>>(num_layer_with_weight);

                    weight_[0] = vector<vector<double>>(cnst::num_input_node, vector<double>(num_node_of_hidden_layer_[0]));
                    for (int i = 1; i < num_node_of_hidden_layer_.size(); ++i) {
                        weight_[i] = vector<vector<double>>(num_node_of_hidden_layer_[i - 1], vector<double>(num_node_of_hidden_layer_[i]));
                    }
                    weight_.back() = vector<vector<double>>(num_node_of_hidden_layer_.back(), vector<double>(cnst::num_output_node));

                    if (!should_skip_initialization) {

                        for (int i = 0; i < weight_.size(); ++i) {
                            #if MNIST_WEIGHT_INITIALIZATION_METHOD == 1 //Xavier initialization
                                stddev = sqrt(1.0 / weight_[i].size());
                            #elif MNIST_WEIGHT_INITIALIZATION_METHOD == 2 //He initialization
                                stddev = sqrt(2.0 / weight_[i].size());
                            #endif
                            for (int j = 0; j < weight_[i].size(); ++j) {
                                for (int k = 0; k < weight_[i][j].size(); ++k) {
                                    weight_[i][j][k] = normal_dist(mt) * stddev;
                                }
                            }
                        }

                    }

                    //} initializes `weight_`

                    //initializes `bias_` {

                    bias_ = vector<vector<double>>(num_layer_with_weight);
                    for (int i = 0; i < num_node_of_hidden_layer_.size(); ++i) {
                        bias_[i].resize(num_node_of_hidden_layer_[i]);
                    }
                    bias_.back().resize(cnst::num_output_node);

                    //} initializes `bias_`

                }

                void create_minibatch_(const vector<vector<double>> &image, vector<vector<double>> &minibatch, vector<int> &random_index_array) {
                    minibatch.resize(batch_size_);
                    random_index_array = rand_.create_random_number_sequence_(0, image.size(), batch_size_);
                    for (int i = 0; i < batch_size_; ++i) {
                        minibatch[i] = image[random_index_array[i]];
                    }
                }

                // Input: a minibatch
                //Output: an array of predicted labels for images in the input
                vector<int> predict_(const vector<vector<double>> &input) {
                    vector<vector<double>> output = input;
                    for (int i = 0; i < layer_.size(); ++i) {
                        output = (layer_[i] -> forward_propagation_(output));
                    }
                    vector<int> predicted_label;
                    for (int i = 0; i < output.size(); ++i) {
                        predicted_label.push_back(vector_operation::index_for_max_value(output[i]));
                    }
                    return predicted_label;
                }

                double calculate_gradient_by_central_difference_(const vector<vector<double>> &input, const vector<int> &random_index_array, double dx, vector<vector<vector<double>>> &dLdW, vector<vector<double>> &dLdB) {

                    for (int i = 0; i < weight_.size(); ++i) {

                        for (int j = 0; j < weight_[i].size(); ++j) {
                            for (int k = 0; k < weight_[i][j].size(); ++k) {

                                //f(x+h)
                                weight_[i][j][k] += dx;
                                const double E1 = forward_propagation_(input, random_index_array);

                                //f(x-h)
                                weight_[i][j][k] -= 2 * dx;
                                const double E2 = forward_propagation_(input, random_index_array);

                                weight_[i][j][k] += dx;
                                dLdW[i][j][k] = (E1 - E2) / (2 * dx); //central difference

                            }
                        }

                        for (int j = 0; j < bias_[i].size(); ++j) {
                            
                            bias_[i][j] += dx;
                            const double E1 = forward_propagation_(input, random_index_array);

                            bias_[i][j] -= 2 * dx;
                            const double E2 = forward_propagation_(input, random_index_array);

                            bias_[i][j] += dx;
                            dLdB[i][j] = (E1 - E2) / (2 * dx);

                        }
                    }

                    const double E = forward_propagation_(input, random_index_array);
                    return E;

                }

                double calculate_gradient_by_backpropagation_(const vector<vector<double>> &input, const vector<int> &random_index_array) {
                    const double E = forward_propagation_(input, random_index_array);
                    backward_propagation_();
                    return E;
                }

                //This function checks the difference between gradient calculated by central difference and one calculated by backpropagation.
                //They should have very similar value.
                void gradient_check_(const vector<vector<double>> &input, const vector<int> &random_index_array, double dx) {

                    //gradient calculated by central difference
                    vector<vector<vector<double>>> dLdW_numerical = weight_;
                    vector<vector<double>> dLdB_numerical = bias_;

                    calculate_gradient_by_central_difference_(input, random_index_array, dx, dLdW_numerical, dLdB_numerical);

                    double dLdW_diff = 0;
                    double dLdB_diff = 0;
                    unsigned num_dLdW_element = 0; //This is used to calculate an average value.
                    unsigned num_dLdB_element = 0;

                    for (int i = 0; i < weight_.size(); ++i) {

                        //gradient calculated by central difference
                        const vector<vector<double>> &dLdW_1 = dLdW_numerical[i];
                        const vector<double> &dLdB_1 = dLdB_numerical[i];

                        //gradient calculated by backpropagation
                        const vector<vector<double>> &dLdW_2 = (dynamic_cast<layer::AffineLayer *>(layer_[cnst::num_layer_in_hidden_layer * i]) -> get_dLdW_());
                        const vector<double> &dLdB_2 = (dynamic_cast<layer::AffineLayer *>(layer_[cnst::num_layer_in_hidden_layer * i]) -> get_dLdB_());

                        num_dLdW_element += weight_[i].size() * weight_[i][0].size();
                        num_dLdB_element += bias_[i].size();

                        for (int j = 0; j < weight_[i].size(); ++j) {
                            for (int k = 0; k < weight_[i][j].size(); ++k) {
                                if (dLdW_1[j][k] == 0 && dLdW_2[j][k] == 0) {
                                    dLdW_diff += 1;
                                } else if (dLdW_2[j][k] == 0) {
                                    dLdW_diff += INFINITY;
                                } else {
                                    dLdW_diff += dLdW_1[j][k] / dLdW_2[j][k];
                                }
                            }
                        }
                        for (int j = 0; j < bias_[i].size(); ++j) {
                            if (dLdB_1[j] == 0 && dLdB_2[j] == 0) {
                                dLdB_diff += 1;
                            } else if (dLdB_2[j] == 0) {
                                dLdB_diff += INFINITY;
                            } else {
                                dLdB_diff += dLdB_1[j] / dLdB_2[j];
                            }
                        }
                    }

                    cout << "=== gradient diff ===\n";
                    cout << "Average Difference of `dLdW`: " << abs((dLdW_diff / static_cast<double>(num_dLdW_element) - 1) * 100) << "(%)\n";
                    cout << "Average Difference of `dLdB`: " << abs((dLdB_diff / static_cast<double>(num_dLdB_element) - 1) * 100) << "(%)\n";
                    cout << "=====================\n";

                    //This is needed to revert the change brought up by the call of `calculate_gradient_by_central_difference_()`.
                    calculate_gradient_by_backpropagation_(input, random_index_array);

                }

                double check_accuracy_(const vector<vector<double>> &image, const vector<int> &label) {

                    vector<vector<double>> input;
                    vector<int> random_index_array;
                    create_minibatch_(image, input, random_index_array);

                    const vector<int> predicted_label = predict_(input);

                    unsigned correct_count = 0;
                    for (int i = 0; i < predicted_label.size(); ++i) {
                        if (predicted_label[i] == label[random_index_array[i]]) {
                            ++correct_count;
                        }
                    }

                    return (correct_count / static_cast<double>(batch_size_));

                }

            public:
                
                MNIST(activation_function_type_ activation_function_type, loss_function_type_ loss_function_type, const vector<unsigned> &num_node_of_hidden_layer, unsigned batch_size, bool should_normalize_pixel_value, mt19937::result_type seed, double stddev, bool should_skip_initialization = false)
                    :
                        num_node_of_hidden_layer_(num_node_of_hidden_layer),
                        layer_(cnst::num_layer_in_hidden_layer * num_node_of_hidden_layer_.size() + 1),
                        batch_size_(batch_size),
                        rand_(seed)
                {

                    //All the codes are written, assuming not a single input but a minibatch is passed.
                    //Thus, the value of `batch_size_` shall not be too small.
                    const unsigned minimum_batch_size = 5;
                    if (batch_size_ < minimum_batch_size) {
                        cout << "The value [ " << batch_size_ << " ] of `batch_size_` is too small. It shall at least be " << minimum_batch_size << " or so.\n";
                        error_flag_ = true;
                    }

                    if (num_node_of_hidden_layer_.empty()) { //when there is no hidden layer
                        error_flag_ = true;
                    }

                    if (!should_skip_initialization) {

                        read_mnist_dataset_(should_normalize_pixel_value);

                    } else { //This is useful **only when** you'd like to skip both `training_()` and `testing_()` and just call `infer_()`.
                             //This mode is used from `infer_digit.cpp` for example.
                        ;
                    }

                    initialize_weight_and_bias_(stddev, should_skip_initialization);

                    //creates layers {
                    //This is the structure:
                    // 
                    //           _________ hidden layer ________  _________ hidden layer _______          ______________________ output layer _____________________
                    //          |                               ||                              |        |                                                         |
                    // [input] ---> [affine] ---> [activation] ---> [affine] ---> [activation] ---> ... ---> [affine] ---> [last activation] ---> [loss function] ---> output
                    //          |______________________________________________________________________________________||__________________________________________|
                    // 
                    //                                          elements of `layer_`                                                    `last_layer_`
                    // 
                    //, or if `MNIST_ENABLE_BATCH_NORMALIZATION` macro is set to `1`, "[batch normalization]"s are additionally inserted just after "[affine]"s.

                    for (int i = 0; i < layer_.size(); ++i) {
                        if (i % cnst::num_layer_in_hidden_layer == 0) {
                            layer_[i] = new layer::AffineLayer(weight_[i / cnst::num_layer_in_hidden_layer], bias_[i / cnst::num_layer_in_hidden_layer]);
                        } else {

                            #if MNIST_ENABLE_BATCH_NORMALIZATION == 1
                                if (i % cnst::num_layer_in_hidden_layer == 1) {
                                    layer_[i] = new layer::BatchNormalizationLayer(batch_size_, weight_[i / cnst::num_layer_in_hidden_layer][0].size());
                                } else {
                            #endif

                                    if (activation_function_type == sigmoid_) {
                                        layer_[i] = new layer::SigmoidLayer;
                                    } else if (activation_function_type == relu_) {
                                        layer_[i] = new layer::ReluLayer;
                                    }

                            #if MNIST_ENABLE_BATCH_NORMALIZATION == 1
                                }
                            #endif

                        }
                    }

                    if (loss_function_type == cross_entropy_) {
                        last_layer_ = new layer::SoftmaxCrossEntropyLayer(label_train_, label_test_);
                    } else { //currently not supported
                        error_flag_ = true;
                    }

                    //} creates layers

                    #if MNIST_DEBUG > 1
                        cout << "weight_.size() = " << weight_.size() << "\n";
                        for (int i = 0; i < weight_.size(); ++i) {
                            cout << "weight_[" << i << "].size() = " << weight_[i].size() << "x" << weight_[i][0].size() << "\n";
                        }
                        cout << "bias_.size() = " << bias_.size() << "\n";
                        for (int i = 0; i < bias_.size(); ++i) {
                            cout << "bias_[" << i << "].size() = " << bias_[i].size() << "\n";
                        }
                        cout << "\n";
                    #endif

                }

                ~MNIST() {
                    for (int i = 0; i < layer_.size(); ++i) {
                        delete layer_[i];
                    }
                    delete last_layer_;
                }

                double forward_propagation_(const vector<vector<double>> &input, const vector<int> &random_index_array) {
                    vector<vector<double>> output = input;
                    for (int i = 0; i < layer_.size(); ++i) {
                        output = (layer_[i] -> forward_propagation_(output));
                    }
                    const double E = (last_layer_ -> forward_propagation_(output, random_index_array));
                    return E;
                }

                void backward_propagation_() {
                    vector<vector<double>> output;
                    output = (last_layer_ -> backward_propagation_());
                    for (int i = layer_.size() - 1; i >= 0; --i) {
                        output = (layer_[i] -> backward_propagation_(output));
                    }
                }

                void training_(unsigned epoch, double dx, double learning_rate, optimizer::optimizer_type opt_type) {

                    last_layer_ -> change_calculation_mode_(layer::LastLayer::training_);

                    loss_function_history_.clear();
                    accuracy_history_for_training_data_.clear();
                    accuracy_history_for_testing_data_.clear();

                    unique_ptr<optimizer::Optimizer> optimizer = optimizer::create_optimizer(learning_rate, weight_, bias_, opt_type);
                    
                    const unsigned num_loop_per_epoch = image_train_.size() / batch_size_;
                    const unsigned num_loop = num_loop_per_epoch * epoch;

                    for (int iter = 0; iter < num_loop; ++iter) {

                        vector<vector<double>> input;
                        vector<int> random_index_array;
                        create_minibatch_(image_train_, input, random_index_array);

                        #if MNIST_GRADIENT_TYPE == 0 //backpropagation (efficient)

                            const double E = calculate_gradient_by_backpropagation_(input, random_index_array);

                        #elif MNIST_GRADIENT_TYPE == 1 //central difference (inefficient)

                            vector<vector<vector<double>>> dLdW_array = weight_;
                            vector<vector<double>> dLdB_array = bias_;
                            const double E = calculate_gradient_by_central_difference_(input, random_index_array, dx, dLdW_array, dLdB_array);

                        #endif

                        loss_function_history_.push_back(E);

                        #ifdef MNIST_GRADIENT_CHECK
                            if (iter % num_loop_per_epoch == 0) {
                                gradient_check_(input, random_index_array, dx);
                            }
                        #endif

                        //modifies parameters {

                        for (int i = 0; i < weight_.size(); ++i) {

                            #if MNIST_GRADIENT_TYPE == 0
                                const vector<vector<double>> &dLdW = (dynamic_cast<layer::AffineLayer *>(layer_[cnst::num_layer_in_hidden_layer * i]) -> get_dLdW_());
                                const vector<double> &dLdB = (dynamic_cast<layer::AffineLayer *>(layer_[cnst::num_layer_in_hidden_layer * i]) -> get_dLdB_());
                            #elif MNIST_GRADIENT_TYPE == 1
                                const vector<vector<double>> &dLdW = dLdW_array[i];
                                const vector<double> &dLdB = dLdB_array[i];
                            #endif

                            optimizer -> optimize_(i, dLdW, dLdB);

                        }

                        #if MNIST_ENABLE_BATCH_NORMALIZATION == 1
                            for (int i = 0; i < layer_.size(); ++i) {
                                if (i % cnst::num_layer_in_hidden_layer == 1) {
                                    dynamic_cast<layer::BatchNormalizationLayer *>(layer_[i]) -> optimize_gamma_and_beta_(learning_rate);
                                }
                            }
                        #endif

                        //} modifies parameters

                        //checks accuracy {

                        const double accuracy_for_training_data = check_accuracy_(image_train_, label_train_) * 100;
                        const double accuracy_for_testing_data = check_accuracy_(image_test_, label_test_) * 100;

                        accuracy_history_for_training_data_.push_back(accuracy_for_training_data);
                        accuracy_history_for_testing_data_.push_back(accuracy_for_testing_data);

                        #ifdef MNIST_DEBUG
                            if (iter % num_loop_per_epoch == 0) {
                                cout << "=== accuracy ===\n";
                                cout << "Training Data: " << accuracy_for_training_data<< "(%)\n";
                                cout << " Testing Data: " << accuracy_for_testing_data << "(%)\n";
                                cout << "================\n";
                            }
                        #endif

                        #if MNIST_DEBUG > 2
                            cout << "(" << iter << "/" << num_loop << ") " << E << " " << accuracy_for_training_data << " " << accuracy_for_testing_data << "\n";
                        #endif

                        //} checks accuracy

                    }

                }
                
                double testing_() {

                    last_layer_ -> change_calculation_mode_(layer::LastLayer::testing_);

                    const unsigned num_loop = image_test_.size() / batch_size_;
                    unsigned correct_count = 0;

                    for (int iter = 0; iter < num_loop; ++iter) {

                        vector<int> index_mask(batch_size_);
                        vector<vector<double>> minibatch(batch_size_);
                        for (int i = 0; i < batch_size_; ++i) {
                            index_mask[i] = iter * batch_size_ + i;
                            minibatch[i] = image_test_[index_mask[i]];
                        }

                        const vector<int> predicted_label = predict_(minibatch);

                        for (int i = 0; i < predicted_label.size(); ++i) {
                            if (predicted_label[i] == label_test_[index_mask[i]]) {
                                ++correct_count;
                            }
                        }

                    }

                    return (correct_count / static_cast<double>(num_loop * batch_size_) * 100);

                }

                vector<int> infer_(vector<vector<double>> minibatch) {

                    last_layer_ -> change_calculation_mode_(layer::LastLayer::testing_);

                    //appends dummy data to make the size be `batch_size_`
                    for (int i = 1; i < batch_size_; ++i) {

                        if (image_test_.empty()) {

                            vector<double> dummy(cnst::num_input_node);
                            for (int j = 0; j < cnst::num_input_node; ++j) {
                                dummy.push_back(rand_.bool_rand_());
                            }
                            minibatch.push_back(dummy);

                        } else {

                            minibatch.push_back(image_test_[rand_.int_rand_(0, image_test_.size() - 1)]);

                        }

                    }

                    const vector<int> predicted_label = predict_(minibatch);
                    return predicted_label;

                }

                void save_parameters_(const string &filename) {

                    ofstream ofs(filename.c_str(), ios_base::binary);
                    if (!ofs) {
                        cout << "Couldn't open the file [ " << filename << " ] for write.\n";
                        error_flag_ = true;
                        return;
                    }

                    #if MNIST_DEBUG
                        cout << "\n----- weight -----\n";
                    #endif

                    for (int i = 0; i < weight_.size(); ++i) {
                        for (int j = 0; j < weight_[i].size(); ++j) {
                            for (int k = 0; k < weight_[i][j].size(); ++k) {

                                ofs.write(reinterpret_cast<char *>(&weight_[i][j][k]), sizeof(double));

                                #if MNIST_DEBUG
                                    cout << weight_[i][j][k] << " ";
                                #endif

                            }
                        }
                    }

                    #if MNIST_DEBUG
                        cout << "\n------------------\n\n----- bias -----\n";
                    #endif

                    for (int i = 0; i < bias_.size(); ++i) {
                        for (int j = 0; j < bias_[i].size(); ++j) {

                            ofs.write(reinterpret_cast<char *>(&bias_[i][j]), sizeof(double));

                            #if MNIST_DEBUG
                                cout << bias_[i][j] << " ";
                            #endif

                        }
                    }

                    #if MNIST_DEBUG
                        cout << "\n----------------\n\n";
                    #endif

                    #if MNIST_ENABLE_BATCH_NORMALIZATION == 1

                        for (int i = 0; i < layer_.size(); ++i) {
                            if (i % cnst::num_layer_in_hidden_layer == 1) {

                                const vector<double> &gamma = dynamic_cast<layer::BatchNormalizationLayer *>(layer_[i]) -> get_gamma_();
                                for (int j = 0; j < gamma.size(); ++j) {
                                    ofs.write(reinterpret_cast<const char *>(&gamma[j]), sizeof(double));
                                }

                                const vector<double> &beta = dynamic_cast<layer::BatchNormalizationLayer *>(layer_[i]) -> get_beta_();
                                for (int j = 0; j < beta.size(); ++j) {
                                    ofs.write(reinterpret_cast<const char *>(&beta[j]), sizeof(double));
                                }

                            }
                        }

                    #endif

                    ofs.close();

                }

                void load_parameters_(const string &filename) {

                    ifstream ifs(filename.c_str(), ios_base::binary);
                    if (!ifs) {
                        cout << "Couldn't open the file [ " << filename << " ] for read.\n";
                        error_flag_ = true;
                        return;
                    }

                    #if MNIST_DEBUG
                        cout << "\n----- weight -----\n";
                    #endif

                    for (int i = 0; i < weight_.size(); ++i) {
                        for (int j = 0; j < weight_[i].size(); ++j) {
                            for (int k = 0; k < weight_[i][j].size(); ++k) {

                                ifs.read(reinterpret_cast<char *>(&weight_[i][j][k]), sizeof(double));

                                #if MNIST_DEBUG
                                    cout << weight_[i][j][k] << " ";
                                #endif

                            }
                        }
                    }

                    #if MNIST_DEBUG
                        cout << "\n------------------\n\n----- bias -----\n";
                    #endif

                    for (int i = 0; i < bias_.size(); ++i) {
                        for (int j = 0; j < bias_[i].size(); ++j) {

                            ifs.read(reinterpret_cast<char *>(&bias_[i][j]), sizeof(double));
                            #if MNIST_DEBUG
                                cout << bias_[i][j] << " ";
                            #endif

                        }
                    }

                    #if MNIST_DEBUG
                        cout << "\n----------------\n\n";
                    #endif

                    #if MNIST_ENABLE_BATCH_NORMALIZATION == 1

                        for (int i = 0; i < layer_.size(); ++i) {
                            if (i % cnst::num_layer_in_hidden_layer == 1) {

                                vector<double> gamma(dynamic_cast<layer::BatchNormalizationLayer *>(layer_[i]) -> get_gamma_().size());
                                for (int j = 0; j < gamma.size(); ++j) {
                                    ifs.read(reinterpret_cast<char *>(&gamma[j]), sizeof(double));
                                }
                                dynamic_cast<layer::BatchNormalizationLayer *>(layer_[i]) -> set_gamma_(gamma);

                                vector<double> beta(dynamic_cast<layer::BatchNormalizationLayer *>(layer_[i]) -> get_beta_().size());
                                for (int j = 0; j < beta.size(); ++j) {
                                    ifs.read(reinterpret_cast<char *>(&beta[j]), sizeof(double));
                                }
                                dynamic_cast<layer::BatchNormalizationLayer *>(layer_[i]) -> set_beta_(beta);

                            }
                        }

                    #endif

                    //Now we shall be at the very end of the file.
                    if (!ifs) {
                        cout << "An error occurred while reading the file [ " << filename << " ]. Perhaps the network size is different.\n";
                        error_flag_ = true;
                        return;
                    } else {
                        double trash;
                        ifs.read(reinterpret_cast<char *>(&trash), sizeof(double));
                        if (ifs) {
                            cout << "The file [ " << filename << " ] seems to have more than enough information. Perhaps the network size is different.\n";
                            error_flag_ = true;
                            return;
                        }
                    }

                    ifs.close();

                }

                vector<double> get_loss_function_history_() const {
                    return loss_function_history_;
                }

                vector<double> get_accuracy_history_for_training_data_() const {
                    return accuracy_history_for_training_data_;
                }

                vector<double> get_accuracy_history_for_testing_data_() const {
                    return accuracy_history_for_testing_data_;
                }

                bool error_() const {
                    return error_flag_;
                }

                operator bool () const {
                    return !error_flag_;
                }

        };

    }

#endif

// vim: spell

