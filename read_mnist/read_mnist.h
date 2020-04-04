#ifndef is_read_mnist_included

    #define is_read_mnist_included

    #include <iostream>
    #include <fstream>
    #include <vector>

    namespace read_mnist {

        using namespace std;

        const char *image_train_name = "image_train.dat";
        const char *label_train_name = "label_train.dat";
        const char *image_test_name = "image_test.dat";
        const char *label_test_name = "label_test.dat";

        const unsigned image_train_size = 60000;
        const unsigned label_train_size = 60000;
        const unsigned image_test_size = 10000;
        const unsigned label_test_size = 10000;

        const unsigned num_pixel_per_image = 28 * 28;

        const unsigned num_digit = 10; //0,1,...,9

        //loads images {

        template <typename T>
        bool read_image(vector<vector<T>> &v, const string &path, unsigned size) {

            ifstream ifs(path, ios_base::binary | ios_base::in);
            if (!ifs) {
                cout << __func__ << ": Couldn't open the file [ " << path << " ].\n";
                return false;
            }

            unsigned char *buf = new unsigned char [size * read_mnist::num_pixel_per_image];
            ifs.read(reinterpret_cast<char *>(buf), sizeof(unsigned char) * size * read_mnist::num_pixel_per_image);
            if (!ifs) {
                cout << __func__ << ": Format of the file [ " << path << " ] is invalid.\n";
                return false;
            }

            v = vector<vector<T>>(size);
            for (int i = 0; i < size; ++i) {
                v[i].assign(buf + i * read_mnist::num_pixel_per_image, buf + i * read_mnist::num_pixel_per_image + read_mnist::num_pixel_per_image);
            }

            delete[] buf;

            return true;

        }

        template <typename T>
        bool read_image_train(vector<vector<T>> &v, const string &parent_directory) {
            return read_mnist::read_image(v, parent_directory + read_mnist::image_train_name, read_mnist::image_train_size);
        }

        template <typename T>
        bool read_image_test(vector<vector<T>> &v, const string &parent_directory) {
            return read_mnist::read_image(v, parent_directory + read_mnist::image_test_name, read_mnist::image_test_size);
        }

        //} loads images

        //loads labels {

        template <typename T>
        bool read_label(vector<T> &v, const string &path, unsigned size) {

            ifstream ifs(path, ios_base::binary | ios_base::in);
            if (!ifs) {
                cout << __func__ << ": Couldn't open the file [ " << path << " ].\n";
                return false;
            }

            unsigned char *buf = new unsigned char [size];
            ifs.read(reinterpret_cast<char *>(buf), sizeof(unsigned char) * size);
            if (!ifs) {
                cout << __func__ << ": Format of the file [ " << path << " ] is invalid.\n";
                return false;
            }

            v.assign(buf, buf + size);

            delete[] buf;

            return true;

        }

        template <typename T>
        bool read_label_train(vector<T> &v, const string &parent_directory) {
            return read_mnist::read_label(v, parent_directory + read_mnist::label_train_name, read_mnist::label_train_size);
        }

        template <typename T>
        bool read_label_test(vector<T> &v, const string &parent_directory) {
            return read_mnist::read_label(v, parent_directory + read_mnist::label_test_name, read_mnist::label_test_size);
        }

        //} loads labels

        //loads labels (one-hot representation) {

        //This is same as `read_label()` but stores in one-hot representation.
        template <typename T>
        bool read_label_one_hot(vector<vector<T>> &v, const string &path, unsigned size) {

            ifstream ifs(path, ios_base::binary | ios_base::in);
            if (!ifs) {
                cout << __func__ << ": Couldn't open the file [ " << path << " ].\n";
                return false;
            }

            unsigned char *buf = new unsigned char [size];
            ifs.read(reinterpret_cast<char *>(buf), sizeof(unsigned char) * size);
            if (!ifs) {
                cout << __func__ << ": Format of the file [ " << path << " ] is invalid.\n";
                return false;
            }

            v = vector<vector<T>>(size, vector<T>(read_mnist::num_digit, 0));
            for (int i = 0; i < size; ++i) {
                v[i][buf[i]] = 1;
            }

            delete[] buf;

            return true;

        }

        template <typename T>
        bool read_label_train_one_hot(vector<vector<T>> &v, const string &parent_directory) {
            return read_mnist::read_label_one_hot(v, parent_directory + read_mnist::label_train_name, read_mnist::label_train_size);
        }

        template <typename T>
        bool read_label_test_one_hot(vector<vector<T>> &v, const string &parent_directory) {
            return read_mnist::read_label_one_hot(v, parent_directory + read_mnist::label_test_name, read_mnist::label_test_size);
        }

        //} loads labels (one-hot representation)

    }

#endif

// vim: spell

