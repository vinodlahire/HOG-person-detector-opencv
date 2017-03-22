#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <vector>


using namespace cv;
using namespace cv::ml;
using namespace std;

void draw_locations(Mat & img, const vector< Rect > & locations, const Scalar & color);
void draw_locations(Mat & img, const vector< Rect > & locations, const Scalar & color)
{
	if (!locations.empty())
	{
		vector< Rect >::const_iterator loc = locations.begin();
		vector< Rect >::const_iterator end = locations.end();
		for (; loc != end; ++loc)
		{
			rectangle(img, *loc, color, 2);
		}
	}
}


int main() {

	Mat img, gray,draw;
	string filename = "C://sample video1.avi";
	VideoCapture capture(filename);
	Scalar reference(0, 255, 0);
	HOGDescriptor hog;
	vector< Rect > locations;
	hog.setSVMDetector(hog.getDefaultPeopleDetector());
	for (;;) {
		capture >> img;
		//resize(img, img, Size(512, 256));
		cvtColor(img, gray, CV_BGR2GRAY);
		vector< float > hog_detector;
		// Set the people detector.

		draw = img.clone();
		hog.detectMultiScale(gray, locations);
		draw_locations(draw, locations, reference);
		imshow("Video", draw);
		waitKey(1);
	}
}


