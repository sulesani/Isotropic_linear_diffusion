#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <fstream>
#include <sstream>
#include "opencv2/core/ocl.hpp"

using namespace cv;
using namespace std;

// this class is to develop the data_set according to the questions
class Data {
    string image_name;
    double contrast_parameter;
    string question;
    int diffusion_time;

public:
    void set_data(string imagename, double contrast_p, string question_no, int diff_time) {
        image_name = imagename;
        contrast_parameter = contrast_p;
        question = question_no;
        diffusion_time = diff_time;
    }

    string get_image_name() { return image_name; }
    double get_contrast_parameter() { return contrast_parameter; }
    string get_question() { return question; }
    int get_diffusion_time() { return diffusion_time; }
};

// x ==> image to be diffused
// xc ==> diffused image
// iter ==> diffusion time
// k ==> λ (contrast parameter)

int main(int argc, char** argv) {

    //  this array according to the data provided in the questions
    Data data_array[11];
    data_array[0].set_data("office_noisy.png", 0.5, "Question_4_t=1_k=0.5", 1);
    data_array[1].set_data("office_noisy.png", 0.5, "Question_4_t=5_k=0.5", 5);
    data_array[2].set_data("office_noisy.png", 0.5, "Question_4_t=10_k=0.5", 10);
    data_array[3].set_data("office_noisy.png", 0.5, "Question_4_t=30_k=0.5", 30);
    data_array[4].set_data("office_noisy.png", 0.5, "Question_4_t=100_k=0.5", 100);
    data_array[5].set_data("office_noisy.png", 0.5, "Question_5_t=10_k=0.5", 10);
    data_array[6].set_data("office_noisy.png", 1, "Question_5_t=10_k=1", 10);
    data_array[7].set_data("office_noisy.png", 2, "Question_5_t=10_k=2", 10);
    data_array[8].set_data("office_noisy.png", 5, "Question_5_t=10_k=5", 10);
    data_array[9].set_data("office_noisy.png", 10, "Question_5_t=10_k=10", 10);
    data_array[10].set_data("office.png", 0.5, "Question_4_t=10_k=0.5_without_noise", 10);

    //  looping through the data_array
    for (int i = 0; i < sizeof(data_array); i++) {

        // put your image folder path in path_image_folder
        string path_image_folder = "/Users/sami/Desktop/Canada/Assignment/";
        Mat1b x = imread(path_image_folder + data_array[i].get_image_name(), IMREAD_GRAYSCALE);
        Mat x0;
        x.convertTo(x0, CV_32FC1);

        double t = 0;
        double lambda = 0.25;

        // k ==> λ (contrast parameter)
        double K = data_array[i].get_contrast_parameter();

        // iter ==> diffusion time
        int iter = data_array[i].get_diffusion_time();

        Mat x1, xc;

        while (t < iter) {
            Mat D;
            Mat gradxX, gradyX;
            Sobel(x0, gradxX, CV_32F, 1, 0, 3);
            Sobel(x0, gradyX, CV_32F, 0, 1, 3);
            D = Mat::zeros(x0.size(), CV_32F);
            for (int i = 0; i < x0.rows; i++)
                for (int j = 0; j < x0.cols; j++) {
                    float gx = gradxX.at<float>(i, j), gy = gradyX.at<float>(i, j);
                    float d;
                    if (i == 0 || i == x0.rows - 1 || j == 0 || j == x0.cols - 1) // conduction coefficient set to 1
                        d = 1;
                    else
                        d = 1.0 / (1.0 + abs((gx * gx + gy * gy)) / (K * K)); // expression of g(gradient(I))
                        //d =-exp(-(gx*gx + gy*gy)/(K*K)); // expression of g(gradient(I))
                    D.at<float>(i, j) = d;
                }
            x1 = Mat::zeros(x0.size(), CV_32F);
            for (int i = 1; i < x0.rows - 1; i++) {
                float* u1 = (float*)x1.ptr(i);
                u1++;
                for (int j = 1; j < x0.cols - 1; j++, u1++) {
                    // Value of I at (i+1,j),(i,j+1)...(i,j)
                    float ip10 = x0.at<float>(i + 1, j), i0p1 = x0.at<float>(i, j + 1);
                    float im10 = x0.at<float>(i - 1, j), i0m1 = x0.at<float>(i, j - 1), i00 = x0.at<float>(i, j);
                    // Value of D at at (i+1,j),(i,j+1)...(i,j)
                    float cp10 = D.at<float>(i + 1, j), c0p1 = D.at<float>(i, j + 1);
                    float cm10 = D.at<float>(i - 1, j), c0m1 = D.at<float>(i, j - 1), c00 = D.at<float>(i, j);
                    // Equation (7) p632
                    *u1 = i00 + lambda / 4 * ((cp10 + c00) * (ip10 - i00) + (c0p1 + c00) * (i0p1 - i00) + (cm10 + c00) * (im10 - i00) + (c0m1 + c00) * (i0m1 - i00));
                    // equation (9)
                }
            }
            x1.copyTo(x0);
            x0.convertTo(xc, CV_8U);

            // it will show diffused image according to the data.
            // after running code keep pressing enter to show new diffused image accroding to the data
            // provided through looping.
            imshow(data_array[i].get_question(), xc);
            waitKey(5);
            t = t + lambda;
        }
        waitKey();
    }

    return 0;
}
