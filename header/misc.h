#ifndef is_misc_included

    #define is_misc_included

    #include <random>
    #include <vector>

    namespace misc {

        using namespace std;

            class Rand {

                private:

                    mt19937 rand_generator_;
                    /* const */ unsigned rand_max_;
                    uniform_real_distribution<double> probability_dist_;

                public:

                    Rand(unsigned seed = -1) : probability_dist_(0, 1) {

                        if (seed == -1) {
                            rand_generator_ = mt19937(random_device()());
                        } else {
                            rand_generator_ = mt19937(seed);
                        }

                        rand_max_ = rand_generator_.max();
                        
                    }

                    inline double operator() () {
                        return probability_dist_(rand_generator_);
                    }

                    inline double rand_() {
                        return probability_dist_(rand_generator_);
                    }

                    inline int int_rand_(int min, int max) {
                        return rand_generator_() % (max - min + 1) + min;
                    }

                    inline bool bool_rand_() {
                        return rand_generator_() % 2;
                    }

                    inline double real_rand_(double min, double max) {
                        return rand_generator_() / static_cast<double>(rand_max_) * (max - min) + min;
                    }

                    mt19937 & get_rand_generator_() {
                        return rand_generator_;
                    }

                    template <class RAI>
                    void fisher_yates_shuffle_(RAI first, RAI last, unsigned num_swap = 0) {

                        const typename iterator_traits<RAI>::difference_type diff = (num_swap == 0 ? 1 : last - first - num_swap);

                        while (last - first > diff) {
                            --last;
                            swap(*last, *(first + int_rand_(0, last - first)));
                        }

                    }

                    //This function returns an array whose size is `length` and whose elements are randomly picked from the range `[min, max)` without duplication.
                    vector<int> create_random_number_sequence_(int min, int max, int length) {

                        vector<int> random_number_sequence;

                        if (length > (max - min)) { //Since no duplication is allowed, `length` has upper limit.
                            return vector<int>();
                        }

                        vector<int> array;
                        for (int i = 0; i < (max - min); ++i) {
                            array.push_back(min + i);
                        }

                        fisher_yates_shuffle_(array.begin(), array.end(), length);

                        for (int i = 0; i < length; ++i) {
                            random_number_sequence.push_back(*(array.end() - 1 - i));
                        }

                        return random_number_sequence;

                    }

            };

    }

#endif

