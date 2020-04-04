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

                virtual vector<vector<double>> forward_propagation_(const vector<vector<double>> &forward_input) = 0;

                virtual vector<vector<double>> backward_propagation_(const vector<vector<double>> &backward_input) = 0;

        };

        class LastLayer {

            public:

                enum calculation_mode_type_ {training_, testing_};

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
                    
                    double final_output;
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

