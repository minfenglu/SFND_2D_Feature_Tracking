# SFND 2D Feature Tracking

<img src="images/keypoints.png" width="820" height="248" />

The idea of the camera course is to build a collision detection system - that's the overall goal for the Final Project. As a preparation for this, you will now build the feature tracking part and test various detector / descriptor combinations to see which ones perform best. This mid-term project consists of four parts:

* First, you will focus on loading images, setting up data structures and putting everything into a ring buffer to optimize memory load.
* Then, you will integrate several keypoint detectors such as HARRIS, FAST, BRISK and SIFT and compare them with regard to number of keypoints and speed.
* In the next part, you will then focus on descriptor extraction and matching using brute force and also the FLANN approach we discussed in the previous lesson.
* In the last part, once the code framework is complete, you will test the various algorithms in different combinations and compare them with regard to some performance measures.

See the classroom instruction and code comments for more details on each of these parts. Once you are finished with this project, the keypoint matching part will be set up and you can proceed to the next lesson, where the focus is on integrating Lidar points and on object detection using deep-learning.

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./2D_feature_tracking`.


##  Detectors & Descriptors
* **Detectors**: An interest point detector is an algorithm that chooses points from an image based on some criterion. Typically, an interest point is a local maximum of some function, such as a "cornerness" metric.

* **Descriptors**: A descriptor is a vector of values that describe the image patch around an interest point, like raw pixel values, histogram of gradients (HoG), etc.


##  Haris Corner Detector

<a href="http://www.bmva.org/bmvc/1988/avc-88-023.pdf">Original Paper</a>


<img src="formulas/Harris/Harris_1.png" width="400" height="50" />

Taylor Series:   

<img src="formulas/Harris/Taylor_1.png" width="400" height="25" />


Rewriting the shifted intensity using the above formula:

<img src="formulas/Harris/Taylor_2.png" width="400" height="50" />


<img src="formulas/Harris/Taylor_3.png" width="250" height="50" />

(Image derivatives in the X and Y direction respectively)

<img src="formulas/Harris/Taylor_4.png" width="500" height="50" />

<img src="formulas/Harris/Taylor_5.png" width="250" height="50" />

<img src="formulas/Harris/Taylor_6.png" width="400" height="50" />

<img src="formulas/Harris/Harris_2.png" width="200" height="50" />


<img src="formulas/Harris/Harris_3.png" width="220" height="50" />

<img src="formulas/Harris/Harris_4.png" width="200" height="20" />

Where

<img src="formulas/Harris/Harris_5.png" width="150" height="20" />

<img src="formulas/Harris/Harris_6.png" width="150" height="20" />

<img src="formulas/Harris/lambda1.png" width="20" height="20" />
and <img src="formulas/Harris/lambda2.png" width="20" height="20" /> are eigenvalues of M, k is an empirical constant with value between 0.04-0.06

### C++ Code
```c++
// load image from file
cv::Mat img;
img = cv::imread("../images/img1.png");
cv::cvtColor(img, img, cv::COLOR_BGR2GRAY); // convert to grayscale

// Detector parameters
int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
double k = 0.04;       // Harris parameter

// Detect Harris corners and normalize output
cv::Mat dst, dst_norm, dst_norm_scaled;
dst = cv::Mat::zeros(img.size(), CV_32FC1);
cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
cv::convertScaleAbs(dst_norm, dst_norm_scaled);

// visualize results
string windowName = "Harris Corner Detector Response Matrix";
cv::namedWindow(windowName, 4);
cv::imshow(windowName, dst_norm_scaled);
cv::waitKey(0);

// Look for prominent corners and instantiate keypoints
vector<cv::KeyPoint> keypoints;
double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
for (size_t j = 0; j < dst_norm.rows; j++)
{
    for (size_t i = 0; i < dst_norm.cols; i++)
    {
        int response = (int)dst_norm.at<float>(j, i);
        if (response > minResponse)
        { // only store points above a threshold

            cv::KeyPoint newKeyPoint;
            newKeyPoint.pt = cv::Point2f(i, j);
            newKeyPoint.size = 2 * apertureSize;
            newKeyPoint.response = response;

            // perform non-maximum suppression (NMS) in local neighbourhood around new key point
            bool bOverlap = false;
            for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
            {
                double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                if (kptOverlap > maxOverlap)
                {
                    bOverlap = true;
                    if (newKeyPoint.response > (*it).response)
                    {                      // if overlap is >t AND response is higher for new kpt
                        *it = newKeyPoint; // replace old key point with new one
                        break;             // quit loop over keypoints
                    }
                }
            }
            if (!bOverlap)
            {                                     // only add new key point if no overlap has been found in previous NMS
                keypoints.push_back(newKeyPoint); // store new keypoint in dynamic list
            }
        }
    } // eof loop over cols
}     // eof loop over rows
```




grayscale image

<img src="output_images/Harris/gray.png" width="820" height="248" />

after applying Harris

<img src="output_images/Harris/normalized.png" width="820" height="248" />

after applying non-maximum suppression

<img src="output_images/Harris/keypoints.png" width="820" height="248" />




##  Shi-Tomasi Corner Detector
<a href="http://www.ai.mit.edu/courses/6.891/handouts/shi94good.pdf">Original Paper</a>

The scoring function in Harris Corner Detector is:

<img src="formulas/ShiTomasi/ST_1.png" width="200" height="20" />

The scoring function in Shi-Tomasi Corner Detector is:

<img src="formulas/ShiTomasi/ST_2.png" width="120" height="20" />

### C++ Code


```c++
cv::Mat imgGray;
cv::Mat img = cv::imread("../images/img1.png");
cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

// Shi-Tomasi detector
int blockSize = 6;       //  size of a block for computing a derivative covariation matrix over each pixel neighborhood
double maxOverlap = 0.0; // max. permissible overlap between two features in %
double minDistance = (1.0 - maxOverlap) * blockSize;
int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints
double qualityLevel = 0.01;                                   // minimal accepted quality of image corners
double k = 0.04;
bool useHarris = false;

vector<cv::KeyPoint> kptsShiTomasi;
vector<cv::Point2f> corners;
double t = (double)cv::getTickCount();
cv::goodFeaturesToTrack(imgGray, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, useHarris, k);

for (auto it = corners.begin(); it != corners.end(); ++it)
{ // add corners to result vector
    cv::KeyPoint newKeyPoint;
    newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
    newKeyPoint.size = blockSize;
    kptsShiTomasi.push_back(newKeyPoint);
}

// visualize results
cv::Mat visImage = img.clone();
cv::drawKeypoints(img, kptsShiTomasi, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
string windowName = "Shi-Tomasi Results";
cv::namedWindow(windowName, 1);
imshow(windowName, visImage);
cv::waitKey(0);
```
