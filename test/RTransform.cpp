#include "RTransform.h"
#include <vector>
#include <iostream>
#include <cmath>

#define ROUNDING_FACTOR(x) (((x) >= 0) ? 0.5 : -0.5)
typedef unsigned char uchar;

std::vector <double> RTransformer::RTransform(std::vector< std::vector<unsigned char> > im, int cols, int rows, int N) {
    int image_width = cols;
    int image_height = rows;

    //calc offsets to center the image
    float xofftemp = image_width/2.0f - 1;
    float yofftemp = image_height/2.0f - 1;
    int xoffset = (int)std::floor(xofftemp + ROUNDING_FACTOR(xofftemp));
    int yoffset = (int)std::floor(yofftemp + ROUNDING_FACTOR(yofftemp));
    float dtemp = (float)std::sqrt((double)(xoffset * xoffset + yoffset * yoffset));
    int D = (int)floor(dtemp + ROUNDING_FACTOR(dtemp));

    std::vector <double> rNorm(N, 0.0);
    std::vector <unsigned long long> rT (N, 0);

    /*
    for (int i = 0; i < N; ++i) {
        rNorm[i] = 0.0;
        rT[i] = 0;
    }
    */

    std::vector < std::vector <unsigned long long> > imRadon(N);
    for(int row = 0; row < N; row++) {
        imRadon[row].resize(D);
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D; ++j) {
            imRadon[i][j] = 0;
        }
    }

    //for each angle k to consider
    for (int k= 0 ; k < N; k++) {
        //only consider from PI/8 to 3PI/8 and 5PI/8 to 7PI/8
        //to avoid computational complexity of a steep angle
        if (k == 0){
            k = N/8;
            continue;
        } else if (k == (3 * N / 8 + 1)) {
            k = 5*N/8;
            continue;
        } else if (k == 7 * N / 8 + 1) {
            k = N;
            continue;
        }

        //for each rho length, determine linear equation and sum the line
        //sum is to sum the values along the line at angle k2pi/N
        //sum2 is to sum the values along the line at angle k2pi/N + N/4
        //The sum2 is performed merely by swapping the x,y axis as if the image were rotated 90 degrees.
        for (int d = 0; d < D; d++) {
            double theta = 2*k*3.14/N;//calculate actual theta
            double alpha = std::tan(theta + 3.14/2);//calculate the slope
            double beta_temp = -alpha*d*std::cos(theta) + d*std::sin(theta);//y-axis intercept for the line
            int beta = (int)std::floor(beta_temp + ROUNDING_FACTOR(beta_temp));

            //for each value of m along x-axis, calculate y
            //if the x,y location is within the boundary for the respective image orientations, add to the sum
            unsigned int sum1 = 0, sum2 = 0;
            int M = (image_width >= image_height) ? image_width : image_height;

            for (int m=0;m < M; m++) {
                //interpolate in-between values using nearest-neighbor approximation
                //using m,n as x,y indices into image
                double n_temp = alpha*(m - xoffset) + beta;
                int n = (int)std::floor(n_temp + ROUNDING_FACTOR(n_temp));

                if ((m < image_width) && (n + yoffset >= 0) && (n + yoffset < image_height)){
                    sum1 += im[m][n + yoffset];
                }

                n_temp = alpha*(m - yoffset) + beta;
                n = (int)std::floor(n_temp + ROUNDING_FACTOR(n_temp));

                if ((m < image_height)&&(n + xoffset >= 0)&&(n + xoffset < image_width)){
                    sum2 += im[-(n + xoffset) + image_width - 1][m];
                }
            }
            //assign the sums into the result matrix
            imRadon[k][d] = (float) sum1;
            //assign sum2 to angle position for theta + PI/4
            imRadon[((k + N / 4) % N)][d] = (float) sum2;
        }
    }

    unsigned long long maxVal = 0;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < D; ++j) {
            rT[i] += imRadon[i][j] * imRadon[i][j];
        }

        if (rT[i] > maxVal) {
            maxVal = rT[i];
        }
    }

    for (int i = 0; i < N; ++i) {
        rNorm[i] = (float) rT[i] / maxVal;
    }
    return rNorm;
}

RTransformer::RTransformer()
{

}

RTransformer::~RTransformer()
{

}
