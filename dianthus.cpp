#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>

int main(int argc,char** argv)
{
	cv::Mat ground_truth( cv::imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE ) );
	cv::namedWindow("ground_truth");
	cv::imshow("ground_truth",ground_truth);

	cv::Mat horizontal_sampling = cv::Mat::zeros(ground_truth.rows,ground_truth.cols,CV_8U);
	for ( size_t r = 0; r < horizontal_sampling.rows; r+=2 ) {
		cv::line(horizontal_sampling, cv::Point(0,r), cv::Point(horizontal_sampling.cols-1,r), 255);
	}
	cv::namedWindow("horizontal_sampling");
	cv::imshow("horizontal_sampling",horizontal_sampling); 	

	cv::Mat vertical_sampling = cv::Mat::zeros(ground_truth.rows,ground_truth.cols,CV_8U);
	for ( size_t c = 0; c < vertical_sampling.cols; c+=1 ) {
		cv::line(vertical_sampling, cv::Point(c,0), cv::Point(c,vertical_sampling.rows-1), 255);
	}
	cv::namedWindow("vertical_sampling");
	cv::imshow("vertical_sampling",vertical_sampling);

	cv::Mat sampling;
	cv::bitwise_and(horizontal_sampling,vertical_sampling,sampling);
	cv::namedWindow("sampling");
	cv::imshow("sampling",sampling);


	cv::Mat binary;
 	cv::threshold( ground_truth, binary, 0, 255, cv::THRESH_OTSU );

	cv::namedWindow("binary");
	cv::imshow("binary",binary); 

	cv::Mat points;
	cv::bitwise_and(sampling,binary,points);
	cv::namedWindow("points");
	cv::imshow("points",points);	

	cv::waitKey(0);

	return 0;
}