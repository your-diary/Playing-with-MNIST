#ifndef is_layer_included

    #define is_layer_included

    #include <cmath>
    #include <vector>
    #include "./vector_operation.h"

    namespace layer {

        using namespace std;
        using namespace vector_operation;

        //ABC(abstract base class)
        class Layer {

            public:

                virtual ~Layer() { }

                virtual vector<vector<double>> forward_propagation_(const vector<vector<double>> &forward_input) = 0;

                virtual vector<vector<double>> backward_propagation_(const vector<vector<double>> &backward_input) = 0;

        };

        class LastLayer {

            public:

                enum calculation_mode_type_ {training_, testing_};

            public:

                virtual ~LastLayer() { }

                virtual double forward_propagation_(const vector<vector<double>> &network_output, const vector<int> &label_mask) = 0;

                virtual vector<vector<double>> backward_propagation_() = 0;

                virtual void change_calculation_mode_(calculation_mode_type_ calculation_mode) = 0;

        };

        //This layer calculates `I * W + B` (affine transformation) where `I` is an input, `W` the weight and `B` the bias.
        class AffineLayer : public Layer {

            private:
                
                const vector<vector<double>> &weight_;
                const vector<double> &bias_;

                vector<vector<double>> forward_input_;

                vector<vector<double>> dLdW_;
                vector<double> dLdB_;

            public:

                AffineLayer(const vector<vector<double>> &weight, const vector<double> &bias)
                    :
                    weight_(weight),
                    bias_(bias),
                    dLdB_(bias_.size())
                { }

                vector<vector<double>> forward_propagation_(const vector<vector<double>> &forward_input) {
                    forward_input_ = forward_input;
                    return (forward_input * weight_ + bias_);
                }

                vector<vector<double>> backward_propagation_(const vector<vector<double>> &backward_input) {

                    dLdW_ = vector_operation::transpose(forward_input_) * backward_input;
                    assert(dLdW_.size() == weight_.size());
                    assert(dLdW_[0].size() == weight_[0].size());

                    for (int i = 0; i < dLdB_.size(); ++i) {
                        dLdB_[i] = 0;
                    }
                    for (int i = 0; i < backward_input.size(); ++i) { //This implementation may seem awkward but is faster than the naive implementation in terms of cache.
                        for (int j = 0; j < backward_input[i].size(); ++j) {
                            dLdB_[j] += backward_input[i][j];
                        }
                    }
                    assert(dLdB_.size() == bias_.size());

                    return (backward_input * vector_operation::transpose(weight_));

                }

                const vector<vector<double>> & get_dLdW_() const {
                    return dLdW_;
                }

                const vector<double> & get_dLdB_() const {
                    return dLdB_;
                }

        };

        //This layer performs batch normalization.
        //This layer is to be inserted between `AffineLayer` and its activation function layer.
        class BatchNormalizationLayer : public Layer {

            private:

                static constexpr double epsilon_ = 1e-7;

                const unsigned batch_size_;
                const unsigned input_size_;

                //parameters
                vector<double> gamma_;
                vector<double> beta_;

                vector<double> dLdGamma_;
                vector<double> dLdBeta_;

                //used in backpropagation
                vector<double> stddev_; //\sqrt{\sigma ^2 + \eps}
                vector<vector<double>> X_hat_;
                vector<vector<double>> X_minus_mu_;

                //used inside `optimize_gamma_and_beta_()`
                static constexpr double delta_ = 1e-7; //to avoid zero-division
                vector<double> h_for_gamma_;
                vector<double> h_for_beta_;

            public:

                BatchNormalizationLayer(unsigned batch_size, unsigned input_size)
                    :
                    batch_size_(batch_size),
                    input_size_(input_size),
                    gamma_(input_size_, 1),
                    beta_(input_size_, 0),
                    dLdGamma_(input_size_),
                    dLdBeta_(input_size_),
                    stddev_(input_size_),
                    X_hat_(batch_size_, vector<double>(input_size_)),
                    X_minus_mu_(batch_size_, vector<double>(input_size_)),
                    h_for_gamma_(input_size_, delta_),
                    h_for_beta_(input_size_, delta_)
                { }

                vector<vector<double>> forward_propagation_(const vector<vector<double>> &forward_input) {

                    assert(forward_input.size() == batch_size_);
                    assert(forward_input[0].size() == input_size_);

                    vector<double> mu(input_size_, 0);
                    for (int j = 0; j < input_size_; ++j) {
                        for (int i = 0; i < batch_size_; ++i) {
                            mu[j] += forward_input[i][j];
                        }
                        mu[j] /= batch_size_;
                    }

                    for (int i = 0; i < batch_size_; ++i) {
                        for (int j = 0; j < input_size_; ++j) {
                            X_minus_mu_[i][j] = forward_input[i][j] - mu[j];
                        }
                    }

                    vector<double> sigma_squared(input_size_, 0);
                    for (int j = 0; j < input_size_; ++j) {
                        for (int i = 0; i < batch_size_; ++i) {
                            sigma_squared[j] += pow(X_minus_mu_[i][j], 2);
                        }
                        sigma_squared[j] /= batch_size_;
                    }

                    for (int j = 0; j < input_size_; ++j) {
                        stddev_[j] = sqrt(sigma_squared[j] + epsilon_);
                    }

                    for (int i = 0; i < batch_size_; ++i) {
                        for (int j = 0; j < input_size_; ++j) {
                            X_hat_[i][j] = X_minus_mu_[i][j] / stddev_[j];
                        }
                    }

                    return (gamma_ * X_hat_ + beta_);

                }

                vector<vector<double>> backward_propagation_(const vector<vector<double>> &backward_input) {

                    for (int j = 0; j < input_size_; ++j) {
                        dLdGamma_[j] = 0;
                        dLdBeta_[j] = 0;
                        for (int i = 0; i < batch_size_; ++i) {
                            dLdGamma_[j] += backward_input[i][j] * X_hat_[i][j];
                            dLdBeta_[j] += backward_input[i][j];
                        }
                    }

                    vector<vector<double>> backward_output(batch_size_, vector<double>(input_size_));

                    for (int j = 0; j < input_size_; ++j) {

                        const double A = gamma_[j] / stddev_[j];
                        const double B = A / batch_size_;
                        const double C = B / pow(stddev_[j], 2);

                        double D = 0;
                        double E = 0;
                        for (int k = 0; k < batch_size_; ++k) {
                            D += backward_input[k][j];
                            E += backward_input[k][j] * X_minus_mu_[k][j];
                        }

                        for (int i = 0; i < batch_size_; ++i) {
                            backward_output[i][j] = A * backward_input[i][j] - B * D - C * E * X_minus_mu_[i][j];
                        }

                    }

                    return backward_output;

                }

//                 const vector<double> & get_dLdGamma_() const {
//                     return dLdGamma_;
//                 }
// 
//                 const vector<double> & get_dLdBeta_() const {
//                     return dLdBeta_;
//                 }

                //This function optimizes the values of `gamma_` and `beta_`, using AdaGrad.
                //Ideally this function should instead be included in the classes defined in "./optimizer.h".
                void optimize_gamma_and_beta_(double learning_rate) {

                    h_for_gamma_ = h_for_gamma_ + element_wise_multiplication(dLdGamma_, dLdGamma_);
                    h_for_beta_ = h_for_beta_ + element_wise_multiplication(dLdBeta_, dLdBeta_);

                    for (int j = 0; j < input_size_; ++j) {
                        gamma_[j] -= learning_rate * dLdGamma_[j] / h_for_gamma_[j];
                        beta_[j] -= learning_rate * dLdBeta_[j] / h_for_beta_[j];
                    }

                }

        };

        class SigmoidLayer : public Layer {

            private:

                vector<vector<double>> forward_output_;

            public:

                SigmoidLayer() { }

                vector<vector<double>> forward_propagation_(const vector<vector<double>> &forward_input) {
                    forward_output_ = forward_input;
                    for (int i = 0; i < forward_output_.size(); ++i) {
                        for (int j = 0; j < forward_output_[i].size(); ++j) {
                            forward_output_[i][j] = 1 / (1 + exp(-forward_output_[i][j]));
                        }
                    }
                    return forward_output_;
                }

                vector<vector<double>> backward_propagation_(const vector<vector<double>> &backward_input) {
                    vector<vector<double>> ret(forward_output_.size(), vector<double>(forward_output_[0].size()));
                    for (int i = 0; i < ret.size(); ++i) {
                        for (int j = 0; j < ret[i].size(); ++j) {
                            ret[i][j] = backward_input[i][j] * forward_output_[i][j] * (1 - forward_output_[i][j]);
                        }
                    }
                    assert(ret.size() == backward_input.size());
                    assert(ret[0].size() == backward_input[0].size());
                    return ret;
                }

        };

        class ReluLayer : public Layer {

            private:

                vector<vector<double>> forward_output_;

            public:

                ReluLayer() { }

                vector<vector<double>> forward_propagation_(const vector<vector<double>> &forward_input) {
                    forward_output_ = forward_input;
                    for (int i = 0; i < forward_output_.size(); ++i) {
                        for (int j = 0; j < forward_output_[i].size(); ++j) {
                            if (forward_output_[i][j] < 0) {
                                forward_output_[i][j] = 0;
                            }
                        }
                    }
                    return forward_output_;
                }

                vector<vector<double>> backward_propagation_(const vector<vector<double>> &backward_input) {
                    vector<vector<double>> ret(forward_output_.size(), vector<double>(forward_output_[0].size()));
                    for (int i = 0; i < ret.size(); ++i) {
                        for (int j = 0; j < ret[i].size(); ++j) {
                            if (forward_output_[i][j]) {
                                ret[i][j] = backward_input[i][j];
                            }
                        }
                    }
                    assert(ret.size() == backward_input.size());
                    assert(ret[0].size() == backward_input[0].size());
                    return ret;
                }

        };

        class SoftmaxCrossEntropyLayer : public LastLayer {

            private:

                const vector<int> &true_label_for_training_;
                const vector<int> &true_label_for_testing_;

                const vector<int> *p_true_label_;

                vector<int> forward_input_; //This is a backup of `label_mask` though `network_output` is also an input to this layer.
                vector<vector<double>> forward_output_; //This is a backup of the output of softmax function though the final output of this layer is created additionally by applying cross entropy error.

            public:

                SoftmaxCrossEntropyLayer(const vector<int> &true_label_for_training, const vector<int> &true_label_for_testing, calculation_mode_type_ calculation_mode = training_)
                    :
                    true_label_for_training_(true_label_for_training),
                    true_label_for_testing_(true_label_for_testing)
                {
                    change_calculation_mode_(calculation_mode);
                }

                //This is the structure of `forward_propagation_()`.
                //
                //                               (forward_output_)
                //    network_output              softmax_output                    final_output
                // o ----------------> (softmax) ----------------> (cross entropy) -------------->
                //                                              ->
                //                                             /
                //                                 ------------
                //                                  label_mask
                //                               (forward_input_)
                //
                //`label_mask` has the same size as that of the minibatch's and is an array of indices for `true_label_`.
                double forward_propagation_(const vector<vector<double>> &network_output, const vector<int> &label_mask) {

                    //softmax {

                    forward_output_ = network_output;

                    vector<vector<double>> &softmax_output = forward_output_;

                    for (int i = 0; i < softmax_output.size(); ++i) {

                        const double max_value = vector_operation::max(softmax_output[i]); //avoids overflow
                        double sum = 0;
                        for (int j = 0; j < softmax_output[i].size(); ++j) {
                            sum += exp(softmax_output[i][j] - max_value);
                        }
                        sum = log(sum) + max_value; //Rather than implementing naively (i.e. `exp(x - max_value) / sum`), we convert the division to subtraction inside `exp()` by applying `log()` to and adding `max_value` to `sum` in advance, which may improve the performance.
                        for (int j = 0; j < softmax_output[i].size(); ++j) {
                            softmax_output[i][j] = exp(softmax_output[i][j] - sum);
                        }

                    }

                    //} softmax

                    //cross entropy {

                    forward_input_ = label_mask;
                    
                    double final_output = 0;
                    const unsigned N = softmax_output.size();

                    static const double delta = 1e-7; //avoids overflow

                    for (int i = 0; i < N; ++i) {
                        const int true_label = (p_true_label_ -> operator[](label_mask[i]));
                        final_output -= log(softmax_output[i][true_label] + delta);
                    }
                    final_output /= N;

                    assert(final_output >= 0);

                    //} cross entropy

                    return final_output;

                }

                vector<vector<double>> backward_propagation_() {

                    vector<vector<double>> ret = forward_output_;
                    for (int i = 0; i < ret.size(); ++i) {
                        for (int j = 0; j < ret[i].size(); ++j) {
                            if (j == (p_true_label_ -> operator[](forward_input_[i]))) {
                                ret[i][j] -= 1;
                                break;
                            }
                        }
                    }
                    ret /= ret.size();

                    return ret;

                }

                void change_calculation_mode_(calculation_mode_type_ calculation_mode) {
                    if (calculation_mode == training_) {
                        p_true_label_ = &true_label_for_training_;
                    } else if (calculation_mode == testing_) {
                        p_true_label_ = &true_label_for_testing_;
                    }
                }

        };

    }

#endif

// vim: spell

