//Requirements
//1. `-pthread` flag is specified.
//2. `NUM_THREAD` macro is defined. If not, at most four threads are used.

#ifndef is_vector_operation_included

    #define is_vector_operation_included

    #ifndef NUM_THREAD
        #define NUM_THREAD 4
    #endif

    #include <iostream>
    #include <vector>
    #include <thread>
    #include <cassert>

    namespace vector_operation {

        using namespace std;

        namespace cnst {
            constexpr unsigned num_thread = NUM_THREAD;
        }

//addition {

        namespace __internal {

            template <typename T1, typename T2>
            void matrix_plus_array(vector<vector<T1>> &A, const T2 *B, unsigned row_start, unsigned /* exclusive */ row_end) {

                static const unsigned num_unroll = 5;

                const unsigned j_max = A[0].size();
                const unsigned j_max_for_unrolled_loop = j_max / num_unroll * num_unroll;

                for (int i = row_start; i < row_end; ++i) {
                    T1 *p_A_i = A[i].data();
                    const T2 *p_B = B;
                    for (int j = 0; j < j_max_for_unrolled_loop; j += num_unroll) {
                        *p_A_i++ += *p_B++;
                        *p_A_i++ += *p_B++;
                        *p_A_i++ += *p_B++;
                        *p_A_i++ += *p_B++;
                        *p_A_i++ += *p_B++;
                    }
                    for (int j = j_max_for_unrolled_loop; j < j_max; ++j) {
                        *p_A_i++ += *p_B++;
                    }
                }

            }

        }

        //(matrix)+(array)
        //This function interprets `rhs` as a row vector, repeats it vertically to create a matrix whose dimensions are same as `lhs`'s and finally calculates the sum of two matrices.
        template <typename T1, typename T2>
        vector<vector<decltype(declval<T1>() + declval<T2>())>> operator + (vector<vector<T1>> lhs, const vector<T2> &rhs) {

            assert(lhs[0].size() == rhs.size());

            #define __MATRIX_PLUS_ARRAY_METHOD 1

            #if __MATRIX_PLUS_ARRAY_METHOD == 0

                for (int i = 0; i < lhs.size(); ++i) {
                    for (int j = 0; j < lhs[i].size(); ++j) {
                        lhs[i][j] += rhs[j];
                    }
                }

            #elif __MATRIX_PLUS_ARRAY_METHOD == 1
                //same as above, but with partial unrolling

                static const unsigned num_unroll = 5;

                const unsigned i_max = lhs.size();
                const unsigned j_max = lhs[0].size();
                const unsigned j_max_for_unrolled_loop = j_max / num_unroll * num_unroll;

                for (int i = 0; i < i_max; ++i) {
                    T1 *p_A_i = lhs[i].data();
                    const T2 *p_B = rhs.data();
                    for (int j = 0; j < j_max_for_unrolled_loop; j += num_unroll) {
                        *p_A_i++ += *p_B++;
                        *p_A_i++ += *p_B++;
                        *p_A_i++ += *p_B++;
                        *p_A_i++ += *p_B++;
                        *p_A_i++ += *p_B++;
                    }
                    for (int j = j_max_for_unrolled_loop; j < j_max; ++j) {
                        *p_A_i++ += *p_B++;
                    }
                }

            #elif __MATRIX_PLUS_ARRAY_METHOD == 2
                //multithreaded version
                //This should be faster for larger cases, but worsened the performance for middle cases we'd like to handle.

                vector<thread> thread_array;
                for (int i = 0; i < cnst::num_thread; ++i) {
                    const unsigned row_start = i * (lhs.size() / cnst::num_thread);
                    const unsigned row_end = (i == cnst::num_thread - 1 ? lhs.size() : (i + 1) * (lhs.size() / cnst::num_thread));
                    thread_array.push_back(thread(
                                                    __internal::matrix_plus_array<T1, T2>,
                                                    ref(lhs),
                                                    rhs.data(),
                                                    row_start,
                                                    row_end
                                                  ));
                }
                for (int i = 0; i < cnst::num_thread; ++i) {
                    thread_array[i].join();
                }

            #endif

            return lhs;

        }

        //(array)+(array)
        template <typename T1, typename T2>
        vector<T1> operator + (vector<T1> lhs, const vector<T2> &rhs) {
            assert(lhs.size() == rhs.size());
            for (int i = 0; i < lhs.size(); ++i) {
                lhs[i] += rhs[i];
            }
            return lhs;
        }

        //(matrix)+(matrix)
        template <typename T1, typename T2>
        vector<vector<T1>> operator + (vector<vector<T1>> lhs, const vector<vector<T2>> &rhs) {
            assert(lhs.size() == rhs.size());
            for (int i = 0; i < lhs.size(); ++i) {
                for (int j = 0; j < lhs[i].size(); ++j) {
                    lhs[i][j] += rhs[i][j];
                }
            }
            return lhs;
        }

//} addition

//subtraction {

        //(array)-(array)
        template <typename T1, typename T2>
        vector<T1> operator - (vector<T1> lhs, const vector<T2> &rhs) {
            assert(lhs.size() == rhs.size());
            for (int i = 0; i < lhs.size(); ++i) {
                lhs[i] -= rhs[i];
            }
            return lhs;
        }

        //(matrix)-(matrix)
        template <typename T1, typename T2>
        vector<vector<T1>> operator - (vector<vector<T1>> lhs, const vector<vector<T2>> &rhs) {
            assert(lhs.size() == rhs.size());
            for (int i = 0; i < lhs.size(); ++i) {
                for (int j = 0; j < lhs[i].size(); ++j) {
                    lhs[i][j] -= rhs[i][j];
                }
            }
            return lhs;
        }

//} subtraction

//multiplication {

        namespace __internal {

            template <typename T1, typename T2, typename T3>
            void matrix_multiplication(const vector<vector<T1>> &A, const vector<vector<T2>> &B, vector<vector<T3>> &C, unsigned row_start, unsigned /* exclusive */ row_end) {

                static const unsigned num_unroll = 5;

                const unsigned j_max = B[0].size();
                const unsigned k_max_for_unrolled_loop = B.size() / num_unroll * num_unroll;
                const unsigned k_max = B.size();

                //We employ the optimizations below.
                //These were discussed in |https://stackoverflow.com/questions/60973065/matrix-multiplication-via-stdvector-is-10-times-slower-than-numpy|.
                //1
                //By not using `i,j,k` order but using `i,k,j` order, we reduce operations which stretch over different rows.
                //Generally, when calculating `C = A * B` with the `i,j,k` order, operations are done along with a column of `B`, which is inefficient in terms of cache.
                //Note, however, changing the order to `i,k,j` makes the calculation faster only when, as in the case of the C++ builtin array, the memory is contiguous in the direction of rows.
                //Fortunately this condition is satisfied also for `std::vector`.
                //Each row is not guaranteed to be aligned continuously when nested `std::vector<std::vector>` is used to implement matrices, though.
                //2
                //By partially unrolling the inner-most loop, we speed it up.
                //The size of an input matrix is **not** needed to be known in advance to do partial unrolling.
                //For example, to unroll five iterations, the size doesn't have to be a multiple of five.
                //It is because, in our implementation, most elements are process in the unrolled loop and remaining elements are touched in the normal loop.
                //3
                //By introducing the variables `sum`, `p` and `A_ik`, we reduce the call of `operator []`.
                //In particular, since `p` is a pointer to an internal raw array, it should be faster than an iterator.
                for (int i = row_start; i < row_end; ++i) {
                    for (int k = 0; k < k_max_for_unrolled_loop; k += num_unroll) {
                        for (int j = 0; j < j_max; ++j) {
                            const double *p = A[i].data() + k; //a pointer to `A[i][k]`
                            double sum;
                            sum =  *p++ * B[k][j];
                            sum += *p++ * B[k+1][j];
                            sum += *p++ * B[k+2][j];
                            sum += *p++ * B[k+3][j];
                            sum += *p++ * B[k+4][j];
                            C[i][j] += sum;
                        }
                    }
                    for (int k = k_max_for_unrolled_loop; k < k_max; ++k) {
                        const double A_ik = A[i][k];
                        for (int j = 0; j < j_max; ++j) {
                            C[i][j] += A_ik * B[k][j];
                        }
                    }
                }

            }

        }

        //(matrix)*(matrix)
        //This function multiplies two matrices according to the mathematical definition.
        //Note this is **not** element-wise multiplications.
        template <typename T1, typename T2>
        vector<vector<decltype(declval<T1>() * declval<T2>())>> operator * (const vector<vector<T1>> &lhs, const vector<vector<T2>> &rhs) {

            assert(lhs[0].size() == rhs.size());

            vector<vector<decltype(declval<T1>() * declval<T2>())>> ret(lhs.size(), vector<decltype(declval<T1>() * declval<T2>())>(rhs[0].size()));

            vector<thread> thread_array;
            for (int i = 0; i < cnst::num_thread; ++i) {
                const unsigned row_start = i * (ret.size() / cnst::num_thread);
                const unsigned row_end = (i == cnst::num_thread - 1 ? ret.size() : (i + 1) * (ret.size() / cnst::num_thread));
                thread_array.push_back(thread(
                                                __internal::matrix_multiplication<T1, T2, decltype(declval<T1>() * declval<T2>())>,
                                                cref(lhs),
                                                cref(rhs),
                                                ref(ret),
                                                row_start,
                                                row_end
                                              ));
            }

            for (int i = 0; i < cnst::num_thread; ++i) {
                thread_array[i].join();
            }

            return ret;

        }

        //(matrix)*(matrix)
        //element-wise multiplication
        template <typename T1, typename T2>
        vector<vector<T1>> element_wise_multiplication(vector<vector<T1>> lhs, const vector<vector<T2>> &rhs) {
            assert(lhs.size() == rhs.size());
            for (int i = 0; i < lhs.size(); ++i) {
                for (int j = 0; j < lhs[i].size(); ++j) {
                    lhs[i][j] *= rhs[i][j];
                }
            }
            return lhs;
        }

        //(array)*(array)
        //element-wise multiplication
        template <typename T1, typename T2>
        vector<T1> element_wise_multiplication(vector<T1> lhs, const vector<T2> &rhs) {
            assert(lhs.size() == rhs.size());
            for (int i = 0; i < lhs.size(); ++i) {
                lhs[i] *= rhs[i];
            }
            return lhs;
        }

        //(array)*(matrix)
        //element-wise multiplication (The lhs is replicated vertically to have the same dimensions as the rhs.)
        template <typename T1, typename T2>
        vector<vector<T2>> operator * (const vector<T1> &lhs, vector<vector<T2>> rhs) {
            assert(lhs.size() == rhs[0].size());
            for (int i = 0; i < rhs.size(); ++i) {
                for (int j = 0; j < rhs[i].size(); ++j) {
                    rhs[i][j] *= lhs[j];
                }
            }
            return rhs;
        }

        //(scalar)*(array)
        template <typename T1, typename T2>
        vector<T2> operator * (T1 lhs, vector<T2> rhs) {
            for (int i = 0; i < rhs.size(); ++i) {
                rhs[i] *= lhs;
            }
            return rhs;
        }

        //(scalar)*(matrix)
        template <typename T1, typename T2>
        vector<vector<T2>> operator * (T1 lhs, vector<vector<T2>> rhs) {
            for (int i = 0; i < rhs.size(); ++i) {
                for (int j = 0; j < rhs[i].size(); ++j) {
                    rhs[i][j] *= lhs;
                }
            }
            return rhs;
        }

//} multiplication

//division {

        //(matrix)/=(scalar)
        //element-wise division
        template <typename T1, typename T2>
        vector<vector<T1>> & operator /= (vector<vector<T1>> &lhs, T2 rhs) {
            for (int i = 0; i < lhs.size(); ++i) {
                for (int j = 0; j < lhs[i].size(); ++j) {
                    lhs[i][j] /= rhs;
                }
            }
            return lhs;
        }

//} division

//assignment {

        //(matrix)=(scalar)
        //element-wise assignment
        template <typename T1, typename T2>
        vector<vector<T1>> & element_wise_assignment(vector<vector<T1>> &lhs, T2 rhs) {
            for (int i = 0; i < lhs.size(); ++i) {
                for (int j = 0; j < lhs[i].size(); ++j) {
                    lhs[i][j] = rhs;
                }
            }
            return lhs;
        }

//} assignment

        //This function returns the transposed version of an input matrix.
        //Note transposing is an expensive operation and it can sometimes be avoided just by accessing a matrix with reversed indices (e.g. `A[j][i]` instead of `A[i][j]`).
        template <typename T>
        vector<vector<T>> transpose(const vector<vector<T>> &v) {
            vector<vector<T>> ret(v[0].size(), vector<T>(v.size()));
            for (int i = 0; i < ret.size(); ++i) {
                for (int j = 0; j < ret[i].size(); ++j) {
                    ret[i][j] = v[j][i];
                }
            }
            return ret;
        }

        //This function returns the element of an array who has the maximum value.
        template <typename T>
        T max(const vector<T> &v) {
            T max_value = v[0];
            for (int i = 1; i < v.size(); ++i) {
                if (v[i] > max_value) {
                    max_value = v[i];
                }
            }
            return max_value;
        }

        //This function returns the index of the maximum element in an array.
        template <typename T>
        int index_for_max_value(const vector<T> &v) {
            T max_value = v[0];
            int ret = 0;
            for (int i = 1; i < v.size(); ++i) {
                if (v[i] > max_value) {
                    max_value = v[i];
                    ret = i;
                }
            }
            return ret;
        }

        template <typename T>
        void print_array(const vector<T> &v) {
            if (v.empty()) {
                cout << "[]";
                return;
            }
            cout << "[" << v[0];
            for (int i = 1; i < v.size(); ++i) {
                cout << ", " << v[i];
            }
            cout << "]\n";
        }

    }

#endif

// vim: spell

