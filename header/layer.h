#ifndef is_layer_included

    #define is_layer_included

    #include <vector>
    #include "./vector_operation.h"

    namespace layer {

        using namespace std;
        using namespace vector_operation;

        //ABC(abstract base class)
        class Layer {

            public:

                virtual vector<vector<double>> forward_propagation_(const vector<vector<double>> &input);

                virtual vector<vector<double>> backward_propagation_(const vector<vector<double>> &input);

        };

        //This layer calculates `I * W + B` (affine transformation) where `I` is an input, `W` the weight and `B` the bias.
        class AffineLayer : public Layer {

            private:
                
                const vector<vector<double>> &weight_;
                const vector<vector<double>> &weight_transposed_;
                const vector<double> &bias_;

                vector<vector<double>> forward_input_;

                vector<vector<double>> dLdW_;
                vector<double> dLdB_;

            public:

                AffineLayer(const vector<vector<double>> &weight, const vector<vector<double>> &weight_transposed, const vector<double> &bias)
                    :
                    weight_(weight),
                    weight_transposed_(weight_transposed),
                    bias_(bias),
                    dLdB_(bias_.size())
                { }

                vector<vector<double>> forward_propagation_(const vector<vector<double>> &forward_input) {
                    forward_input_ = forward_input;
                    return (forward_input * weight_ + bias_);
                }

                vector<vector<double>> backward_propagation_(const vector<vector<double>> &backward_input) {

                    dLdW_ = vector_operation::transpose(forward_input_) * backward_input;

                    for (int i = 0; i < dLdB_.size(); ++i) {
                        dLdB_[i] = 0;
                    }
                    for (int i = 0; i < backward_input.size(); ++i) { //This implementation may seem poor but is faster than the naive implementation in terms of cache.
                        for (int j = 0; j < backward_input[i].size(); ++j) {
                            dLdB_[j] += backward_input[i][j];
                        }
                    }

                    return (backward_input * weight_transposed_);

                }

        };

        class SigmoidLayer : public Layer {






        }

    }

#endif

// vim: spell

