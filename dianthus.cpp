#define _USE_MATH_DEFINES 

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/ml/ml.hpp>

#include <iostream>
#include <vector>
#include <map>
#include <numeric>
#include <random>
#include <functional>
#include <set>

cv::Point2f centroid( const std::vector< cv::Point2f >& p )
{
   cv::Point2f r( std::accumulate( p.begin(), p.end(), cv::Point2f(0,0) ) );
   r.x /= p.size();
   r.y /= p.size();
   return r;
}

cv::Mat point_to_homogeneous( const cv::Point2f& p )
{
   float v[]={p.x, p.y, 1};
   return cv::Mat(3,1,CV_32FC1,v).clone();
}

cv::Point2f homogeneous_to_point( const cv::Mat& m )
{
   cv::Mat r = m/m.at<float>(2,0);
   return cv::Point2f(r.at<float>(0,0),r.at<float>(1,0));
}

std::vector< cv::Point2f > affine_transformation( const std::vector< cv::Point2f >& p, float m11, float m12, float m13, float m21, float m22, float m23 )
{
   float to_origin_raw[]={1, 0, -centroid(p).x,
                          0, 1, -centroid(p).y,
                          0, 0,                   1};
   cv::Mat to_origin(3,3,CV_32FC1,to_origin_raw);
   std::vector< cv::Point2f > dst( p.size() );
   for ( size_t i = 0; i < p.size(); i++ ) {
      cv::Mat tmp1 = point_to_homogeneous(p[i]);
      cv::Mat tmp = to_origin*tmp1;
      dst[i] = homogeneous_to_point(tmp);
   }

   float shear_raw[]={m11, m12, m13,
                      m21, m22, m23,
                        0,   0,   1};

   cv::Mat shear(3,3,CV_32FC1,shear_raw);
   for ( size_t i = 0; i < dst.size(); i++ ) {
      cv::Mat tmp1 = point_to_homogeneous(dst[i]);
      cv::Mat tmp = shear*tmp1;
      dst[i] = homogeneous_to_point(tmp);
   }

   float to_centroid_raw[]={1, 0, centroid(p).x,
                          0, 1, centroid(p).y,
                          0, 0,                  1};
   cv::Mat to_centroid(3,3,CV_32FC1,to_centroid_raw);
   for ( size_t i = 0; i < dst.size(); i++ ) {
      cv::Mat tmp1 = point_to_homogeneous(dst[i]);
      cv::Mat tmp = to_centroid*tmp1;
      dst[i] = homogeneous_to_point(tmp);
   }

   return dst;
}

void save_point_set( const char* name, int rows, int cols, const std::vector< cv::Point2f >& p )
{
   cv::Mat img = cv::Mat::zeros( rows, cols, CV_8UC3 );
   for ( size_t i = 0; i < p.size(); i++ ) {
      img.at<cv::Vec3b>(p[i]) = cv::Vec3b(255,255,255);
   }
   cv::imwrite(name,img);
}

float oriented_hausdorff_distance( const std::vector< cv::Point2f >& A, const std::vector< cv::Point2f >& B )
{
   float result = 0;
   for ( size_t i = 0; i < A.size(); i++ ) {
      float min_dist = std::numeric_limits<float>::max();
      for ( size_t j = 0; j < B.size(); j++ ) {
         const float dx = A[i].x-B[j].x;
         const float dy = A[i].y-B[j].y;
         min_dist = std::min( min_dist, dx*dx+dy*dy );
      }
      result = std::max( result, min_dist );
   }
   return std::sqrt(result);
}

float hausdorff_distance( const std::vector< cv::Point2f >& A, const std::vector< cv::Point2f >& B )
{
   return std::max( oriented_hausdorff_distance( A, B ), oriented_hausdorff_distance( B, A ) );
}

// http://stackoverflow.com/a/21735828/15485
double getOrientation(const std::vector<cv::Point2f> &pts, std::vector<cv::Point2f> &normalized)
{
   if (pts.size() == 0) return false;

   //Construct a buffer used by the pca analysis
   cv::Mat data_pts = cv::Mat(pts.size(), 2, CV_32FC1);
   for (int i = 0; i < data_pts.rows; ++i)
   {
      data_pts.at<float>(i, 0) = pts[i].x;
      data_pts.at<float>(i, 1) = pts[i].y;
   }

   //Perform PCA analysis
   cv::PCA pca_analysis(data_pts, cv::Mat(), CV_PCA_DATA_AS_ROW);

   //Store the position of the object
   cv::Point2f pos = cv::Point2f(pca_analysis.mean.at<float>(0, 0),
      pca_analysis.mean.at<float>(0, 1));

   //Store the eigenvalues and eigenvectors
   std::vector<cv::Point2f> eigen_vecs(2);
   std::vector<float> eigen_val(2);
   for (int i = 0; i < 2; ++i)
   {
      eigen_vecs[i] = cv::Point2f(pca_analysis.eigenvectors.at<float>(i, 0),
         pca_analysis.eigenvectors.at<float>(i, 1));

      eigen_val[i] = pca_analysis.eigenvalues.at<float>(i);
   }

   cv::Point2f O = pos;
   const float lambda = 1;
   cv::Point2f X = pos + lambda * cv::Point2f(eigen_vecs[0].x * eigen_val[0], eigen_vecs[0].y * eigen_val[0]);
   cv::Point2f Y = pos + lambda * cv::Point2f(eigen_vecs[1].x * eigen_val[1], eigen_vecs[1].y * eigen_val[1]);

   normalized.resize(pts.size());
   for ( size_t i = 0; i < pts.size(); i++ ) {
      // sottrarre O dovrebbe garantire l'invarianza alla traslazione
      // proiettare sugli assi principali (OX ed OY) dovrebbe garantire l'invarianza alla rotazione
      normalized[i]=cv::Point2f((pts[i]-O).dot(X-O)/cv::norm(X-O),(pts[i]-O).dot(Y-O)/cv::norm(Y-O));
   }
   // todo: manca l'invarianza alla scala

   // todo: il sort qui sotto non funzionera` benissimo perche' c'e` l'operatore == che sui float non lavorera` in maniera significativa
   std::sort( normalized.begin(), normalized.end(), [](const cv::Point2f& a, const cv::Point2f& b){return (a.x < b.x) ? true : (a.x == b.x && a.y < b.y);} );

   // todo il metodo "Geometric Hashing" descritto nel libro
   // Shapiro, Stockman - "Computer Vision" - anno 2000 capitolo 11
   // dice di usasre le triple di punti non collineari ma i caratteri hanno molti punti collineari (per esempio le barre verticali della I e della T)

   return atan2(eigen_vecs[0].y, eigen_vecs[0].x);
}

template < class NOISE >
std::vector< cv::Point2f > generate_sample( char c, const std::map< char, std::vector< cv::Point2f > >& m, NOISE& noise )
{
   std::vector< cv::Point2f > s(m.at(c));
   for ( size_t i = 0; i < m.at(c).size(); i++ ) {
      s[i].x += noise();
      s[i].y += noise();
   }
   return s;
}

int main(int argc,char** argv)
{
   const std::string characters("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789");

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
   for ( size_t c = 0; c < vertical_sampling.cols; c+=2 ) {
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

   std::vector< cv::Point2f > points;

   for ( size_t c = 0; c < img_points.cols; c++ ) {
      for ( size_t r = 0; r < img_points.rows; r++ ) {
         if (img_points.at<unsigned char>(r,c)) {
            points.push_back(cv::Point2f(c,r));
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
   cv::kmeans(samples,characters.length(),labels,cv::TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001),attempts, cv::KMEANS_PP_CENTERS, centers);

   cv::RNG rng(12345);
   std::vector< cv::Vec3b > colors;
   std::map< char, cv::Vec3b > colors2;
   for ( size_t i = 0; i < characters.length(); i++ ) {
      colors.push_back(cv::Vec3b(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255)));
      colors2[characters[i]]=colors.back();
   }

   cv::Mat clusters = cv::Mat::zeros( ground_truth.rows, ground_truth.cols, CV_8UC3 );
   for ( size_t i = 0; i < labels.size(); i++ ) {
      clusters.at<cv::Vec3b>(points[i]) = colors[labels[i]];
   }
   cv::namedWindow("clusters");
   cv::imshow("clusters",clusters);

   std::vector< std::vector< cv::Point2f > > temp(characters.length());
   for ( size_t i = 0; i < labels.size(); i++ ) {
      temp[labels[i]].push_back( points[i] );
   }

   // todo qui sotto si assume il constant pitch (distanza orizzontale tra i caratteri) che se pur vero nei dati di training non lo sara` nei dati reali...
   std::sort( temp.begin(), temp.end(), []( const std::vector< cv::Point2f >& lhs, const std::vector< cv::Point2f >& rhs ) {
      return centroid(lhs).x < centroid(rhs).x;
   });

   std::map< char, std::vector< cv::Point2f > > models;
   for ( size_t i = 0; i < temp.size(); i++ ) {
      models[characters[i]]=temp[i];
   }

   for ( auto it = models.begin(); it != models.end(); ++it ) {
      cv::Mat cluster = cv::Mat::zeros( ground_truth.rows, ground_truth.cols, CV_8UC3 );
      for ( size_t i = 0; i < it->second.size(); i++ ) {
         cluster.at<cv::Vec3b>(it->second[i]) = colors2[it->first];
      }

      std::vector< cv::Point2f > normalized;
      cv::Point2f O,X,Y;
      getOrientation( it->second, normalized );

      cv::line(cluster,O,X,CV_RGB(255,0,0));
      cv::line(cluster,O,Y,CV_RGB(0,255,0));

      std::ostringstream oss;
      oss << "cluster_" << it->first;
      cv::namedWindow(oss.str());
      cv::imshow(oss.str(),cluster);	
   }

   std::default_random_engine dre;
   std::normal_distribution<float> nd(0,0.3);
   auto gauss = std::bind(nd, dre);



   std::map< size_t, size_t > sizes;
   std::map< size_t, std::string > sizes2;
   for ( auto it = models.begin(); it != models.end(); ++it ) {
      if ( sizes.count(it->second.size()) ) {
         sizes[it->second.size()]++;
      } else {
         sizes.insert(std::make_pair(it->second.size(),1));
      }

      if ( sizes2.count(it->second.size()) ) {
         sizes2[it->second.size()] += it->first;
      } else {
         std::ostringstream oss;
         oss << it->first;
         sizes2.insert(std::make_pair(it->second.size(),oss.str()));
      }
   }

   for ( auto it = sizes2.begin(); it != sizes2.end(); ++it ) {
      std::cout << it->first << "\t" << it->second << "\n";
   }

   const size_t N = 10;
   // todo l'approccio di usare tante SVM e` fragile perche' il numero di punti di cui 
   // e` costituito il carattere potrebbe cambiare
   std::map< size_t, cv::Mat > training_data;
   std::map< size_t, cv::Mat > training_label;
   for ( auto it = sizes.begin(); it != sizes.end(); ++it ) {
      training_data[it->first] = cv::Mat::zeros(0,it->first*2,CV_32FC1);
      training_label[it->first] = cv::Mat::zeros(0,1,CV_32SC1);
   }

   for ( auto it = models.begin(); it != models.end(); ++it ) {
      for ( size_t j = 0; j < N; j++ ) {
         std::vector< cv::Point2f > v = generate_sample(it->first,models,gauss);
         std::vector< cv::Point2f > nv;
         getOrientation( v, nv );
         cv::Mat img = cv::Mat::zeros( ground_truth.rows, ground_truth.cols, CV_8UC3 );
         for ( size_t i = 0; i < v.size(); i++ ) {
            img.at<cv::Vec3b>(v[i]) = cv::Vec3b(255,255,255);
         }
         std::ostringstream oss;
         oss << "train_" << it->first << "_" << j << ".png";
         cv::imwrite(oss.str(),img);

         training_data[it->second.size()].push_back(cv::Mat(1,nv.size()*2, CV_32FC1, &nv[0]));
         size_t lab[1] = { characters.find(it->first) };
         training_label[it->second.size()].push_back(cv::Mat(1,1,CV_32SC1,lab));
      }
   }

   for ( auto it = training_data.begin(); it != training_data.end(); ++it ) {
      std::ostringstream oss;
      oss << "matrix_" << it->first << ".png";
      cv::imwrite(oss.str(),it->second);
   }

   // Set up SVM's parameters
   CvSVMParams params;
   params.svm_type    = CvSVM::C_SVC;
   params.kernel_type = CvSVM::LINEAR;
   params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);

   // Train the SVM
   CvSVM SVM;
   SVM.train(training_data.begin()->second, training_label.begin()->second, cv::Mat(), cv::Mat(), params);


   std::default_random_engine dre2;
   std::normal_distribution<float> nd2(0,0.6);
   auto gauss2 = std::bind(nd2, dre2);

   std::cout << "\nPredictions:\n";
   {
   char c = 'F';
   std::vector< cv::Point2f > v = generate_sample(c,models,gauss2);
   std::vector< cv::Point2f > nv;
   getOrientation( v, nv );
   float result = SVM.predict( cv::Mat(1,nv.size()*2, CV_32FC1, &nv[0]), false );
   std::cout << c << "\t" << characters[result] << "\n";
   }

   {
   char c = 'F';
   std::vector< cv::Point2f > v = generate_sample(c,models,gauss2);
   std::vector< cv::Point2f > nv;
   getOrientation( v, nv );
   float result = SVM.predict( cv::Mat(1,nv.size()*2, CV_32FC1, &nv[0]), false );
   std::cout << c << "\t" << characters[result] << "\n";
   }

   {
   char c = 'F';
   std::vector< cv::Point2f > v = generate_sample(c,models,gauss2);
   std::vector< cv::Point2f > nv;
   getOrientation( v, nv );
   float result = SVM.predict( cv::Mat(1,nv.size()*2, CV_32FC1, &nv[0]), false );
   std::cout << c << "\t" << characters[result] << "\n";
   }

   {
   char c = 'J';
   std::vector< cv::Point2f > v = generate_sample(c,models,gauss2);
   std::vector< cv::Point2f > nv;
   getOrientation( v, nv );
   float result = SVM.predict( cv::Mat(1,nv.size()*2, CV_32FC1, &nv[0]), false );
   std::cout << c << "\t" << characters[result] << "\n";
   }

   {
   char c = 'J';
   std::vector< cv::Point2f > v = generate_sample(c,models,gauss2);
   std::vector< cv::Point2f > nv;
   getOrientation( v, nv );
   float result = SVM.predict( cv::Mat(1,nv.size()*2, CV_32FC1, &nv[0]), false );
   std::cout << c << "\t" << characters[result] << "\n";
   }

   {
   char c = 'J';
   std::vector< cv::Point2f > v = generate_sample(c,models,gauss2);
   std::vector< cv::Point2f > nv;
   getOrientation( v, nv );
   float result = SVM.predict( cv::Mat(1,nv.size()*2, CV_32FC1, &nv[0]), false );
   std::cout << c << "\t" << characters[result] << "\n";
   }

   std::default_random_engine dre3;
   std::normal_distribution<float> nd3(0,.15);
   auto gauss3 = std::bind(nd3, dre3);

   std::map< char, std::vector< cv::Point2f > > normalized_models( models );
   for ( auto it = normalized_models.begin(); it != normalized_models.end(); ++it ) {
      getOrientation( models[it->first], it->second );
   }

   std::cout << "Wrong Hausdorff Predictions:\n";
   std::cout << "query\tresult\n";
   for ( size_t n = 0; n < 100; n++ ) {
      for ( size_t i = 0; i < characters.length(); i++ ) {
         const char q = characters[i];
         std::vector< cv::Point2f > query = generate_sample(q,models,gauss3);
         float theta = (1*M_PI)/180;
         query = affine_transformation( query, cos(theta), sin(theta), 0, -sin(theta), cos(theta), 0 );
         std::vector< cv::Point2f > normalized_query;
         getOrientation( query, normalized_query );
         float distance = std::numeric_limits<float>::max();
         char found = 0;
         for ( auto it = normalized_models.begin(); it != normalized_models.end(); ++it ) {
            const float d = hausdorff_distance( normalized_query, it->second );
            if ( d < distance ) {
               distance = d;
               found = it->first;
            }
         }
         if ( q != found ) {
            std::cout << n << "\t" << q << "\t" << found << "\n";
            std::ostringstream oss1;
            oss1 << n << "_" << q << "_wrong_query.png";
            save_point_set( oss1.str().c_str(), ground_truth.rows, ground_truth.cols, query );

            std::ostringstream oss2;
            oss2 << n << "_" << q << "_wrong_result.png";
            save_point_set( oss2.str().c_str(), ground_truth.rows, ground_truth.cols, models[found] );
         }
      }
   }

   cv::waitKey(0);

   return 0;
}


