#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <vector>

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

	cv::Mat img_points;
	cv::bitwise_and(sampling,binary,img_points);
	cv::namedWindow("img_points");
	cv::imshow("img_points",img_points);

	std::vector< cv::Point > points;

	for ( size_t c = 0; c < img_points.cols; c++ ) {
		for ( size_t r = 0; r < img_points.rows; r++ ) {
			if (img_points.at<unsigned char>(r,c)) {
				points.push_back(cv::Point(c,r));
			}
		}
	}

	cv::Mat samples(points.size(),2,CV_32F);
	for ( size_t i = 0; i < points.size(); i++ ) {
		samples.at<float>(i,0)=points[i].x;
		samples.at<float>(i,1)=points[i].y;
	}



	std::vector<int> labels;

	int attempts = 5;
  	cv::Mat centers;
	cv::kmeans(samples,26+10,labels,cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001),attempts, cv::KMEANS_PP_CENTERS, centers);

	

	cv::Mat clusters = cv::Mat::zeros( ground_truth.rows, ground_truth.cols, CV_8UC3 );


	cv::RNG rng(12345);
	std::vector< cv::Vec3b > colors;
	for ( size_t i = 0; i < 26+10; i++ ) {
		colors.push_back(cv::Vec3b(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255)));
	}


	for ( size_t i = 0; i < labels.size(); i++ ) {
		clusters.at<cv::Vec3b>(points[i]) = colors[labels[i]];
	}
	cv::namedWindow("clusters");
	cv::imshow("clusters",clusters);	

	cv::waitKey(0);

	return 0;
}