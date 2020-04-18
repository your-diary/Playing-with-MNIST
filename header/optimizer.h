#ifndef is_optimizer_included

    #define is_optimizer_included

    #include <cmath>
    #include <vector>
    #include <memory>
    #include <cassert>
    #include "./vector_operation.h"

    namespace optimizer {

        using namespace std;
        using namespace vector_operation;

        enum optimizer_type { t_gradient_descent, t_momentum, t_adagrad };

        class Optimizer {

            protected:

                vector<vector<vector<double>>> &weight_;
                vector<vector<double>> &bias_;

                vector<vector<vector<double>>> learning_rate_for_weight_;
                vector<vector<double>> learning_rate_for_bias_;

            public:

                Optimizer(double learning_rate, vector<vector<vector<double>>> &weight, vector<vector<double>> &bias)
                    :
                    weight_(weight),
                    bias_(bias),
                    learning_rate_for_weight_(weight_),
                    learning_rate_for_bias_(bias_)
                {
                    for (int i = 0; i < learning_rate_for_weight_.size(); ++i) {
                        vector_operation::element_wise_assignment(learning_rate_for_weight_[i], learning_rate);
                    }
                    vector_operation::element_wise_assignment(learning_rate_for_bias_, learning_rate);
                }

                virtual ~Optimizer() { }

                virtual void optimize_(unsigned index, const vector<vector<double>> &dLdW, const vector<double> &dLdB) = 0;

        };

        class GradientDescent : public Optimizer {

            public:

                GradientDescent(double learning_rate, vector<vector<vector<double>>> &weight, vector<vector<double>> &bias)
                    :
                    Optimizer(learning_rate, weight, bias)
                { }

                void optimize_(unsigned i, const vector<vector<double>> &dLdW, const vector<double> &dLdB) {

                    for (int j = 0; j < weight_[i].size(); ++j) {
                        for (int k = 0; k < weight_[i][j].size(); ++k) {
                            weight_[i][j][k] -= learning_rate_for_weight_[0][0][0] * dLdW[j][k];
                        }
                    }

                    for (int j = 0; j < bias_[i].size(); ++j) {
                        bias_[i][j] -= learning_rate_for_bias_[0][0] * dLdB[j];
                    }

                }

        };

        class Momentum : public Optimizer {

            private:
                
                vector<vector<vector<double>>> v_for_weight_;
                vector<vector<double>> v_for_bias_;

                const double alpha = 0.9; //The coefficient of `v_*`. Ideally this should be a parameter.

            public:

                Momentum(double learning_rate, vector<vector<vector<double>>> &weight, vector<vector<double>> &bias)
                    :
                    Optimizer(learning_rate, weight, bias),
                    v_for_weight_(weight),
                    v_for_bias_(bias)
                {
                    for (int i = 0; i < v_for_weight_.size(); ++i) {
                        vector_operation::element_wise_assignment(v_for_weight_[i], 0);
                    }
                    vector_operation::element_wise_assignment(v_for_bias_, 0);
                }

                void optimize_(unsigned i, const vector<vector<double>> &dLdW, const vector<double> &dLdB) {

                    v_for_weight_[i] = alpha * v_for_weight_[i] - learning_rate_for_weight_[0][0][0] * dLdW;
                    v_for_bias_[i] = alpha * v_for_bias_[i] - learning_rate_for_bias_[0][0] * dLdB;

                    for (int j = 0; j < weight_[i].size(); ++j) {
                        for (int k = 0; k < weight_[i][j].size(); ++k) {
                            weight_[i][j][k] += v_for_weight_[i][j][k];
                        }
                    }

                    for (int j = 0; j < bias_[i].size(); ++j) {
                        bias_[i][j] += v_for_bias_[i][j];
                    }

                }

        };

        class AdaGrad : public Optimizer {

            private:
                
                vector<vector<vector<double>>> h_for_weight_;
                vector<vector<double>> h_for_bias_;

            public:

                AdaGrad(double learning_rate, vector<vector<vector<double>>> &weight, vector<vector<double>> &bias)
                    :
                    Optimizer(learning_rate, weight, bias),
                    h_for_weight_(weight),
                    h_for_bias_(bias)
                {
                    const double delta = 1e-7; //Theoretically `h_*` are to be initialized to zero, but we use non-zero `delta` as the initial value to avoid zero-division.
                    for (int i = 0; i < h_for_weight_.size(); ++i) {
                        vector_operation::element_wise_assignment(h_for_weight_[i], delta);
                    }
                    vector_operation::element_wise_assignment(h_for_bias_, delta);
                }

                void optimize_(unsigned i, const vector<vector<double>> &dLdW, const vector<double> &dLdB) {

                    h_for_weight_[i] = h_for_weight_[i] + element_wise_multiplication(dLdW, dLdW);
                    h_for_bias_[i] = h_for_bias_[i] + element_wise_multiplication(dLdB, dLdB);

                    for (int j = 0; j < weight_[i].size(); ++j) {
                        for (int k = 0; k < weight_[i][j].size(); ++k) {
                            weight_[i][j][k] -= learning_rate_for_weight_[0][0][0] * dLdW[j][k] / sqrt(h_for_weight_[i][j][k]);
                        }
                    }

                    for (int j = 0; j < bias_[i].size(); ++j) {
                        bias_[i][j] -= learning_rate_for_bias_[0][0] * dLdB[j] / h_for_bias_[i][j];
                    }

                }

        };

        unique_ptr<Optimizer> create_optimizer(double learning_rate, vector<vector<vector<double>>> &weight, vector<vector<double>> &bias, optimizer::optimizer_type opt_type) {
            switch (opt_type) {
                case t_gradient_descent: return unique_ptr<Optimizer>(new GradientDescent(learning_rate, weight, bias));
                case t_momentum:         return unique_ptr<Optimizer>(new Momentum(learning_rate, weight, bias));
                case t_adagrad:          return unique_ptr<Optimizer>(new AdaGrad(learning_rate, weight, bias));
            }
            assert(0);
            return unique_ptr<Optimizer>();
        }

    }

#endif

