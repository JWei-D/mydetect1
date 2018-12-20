#include "highgui/highgui.hpp"    
#include "opencv2/nonfree/nonfree.hpp"    
#include "opencv2/legacy/legacy.hpp"   
#include <iostream>  

using namespace cv;
using namespace std;

void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst, Point2f& a, Point2f& b);



int main(int argc, char *argv[])
{
		VideoCapture  cap2;
		cap2.open(2);
		cap2.set(CV_CAP_PROP_FRAME_WIDTH, 1280); //画面宽
		cap2.set(CV_CAP_PROP_FRAME_HEIGHT, 720);//画面高

		


		VideoCapture cap1;
		//cap1.set(CV_CAP_PROP_FPS, 120);//帧率
		//cap1.set(CV_CAP_PROP_EXPOSURE, -12.0);//曝光时间 10的-12次方秒
		cap1.open(1);
		//cap1.set(CV_CAP_PROP_FOURCC, CV_FOURCC('M', 'J', 'P', 'G'));//压缩格式 mjpg
		cap1.set(CV_CAP_PROP_FRAME_WIDTH, 1280); //画面宽
		cap1.set(CV_CAP_PROP_FRAME_HEIGHT, 720);//画面高


		//cout << '01' << endl;
		if (!cap1.isOpened())
		{
			cerr << "can not open a camera 1 or file." << endl;
			return -1;
		}
		if (!cap2.isOpened())
		{
			cerr << "can not open a camera 2 or file." << endl;
			return -1;
		}
		

		Mat orig1;
		Mat orig2;

		//cap1 >> orig1;//cap::read()
		//cap2 >> orig2;


		//orig1 = imread("..\\..\\data\\you1.jpg", -1);
		//orig2 = imread("..\\..\\data\\you1.jpg", -1);

		//cout << orig1.size()<< endl;

		

		while (orig1.empty())
		{
			//cout <<'3'<< endl;
			cap1 >> orig1;
		}
		while (orig2.empty())
		{
			//cout <<'3'<< endl;
			cap2 >> orig2;
		}


		Mat map1, map2, map3, map4;
		//Mat orig1 = imread("..\\..\\data\\zuo1.jpg", 1);
		//Mat orig2 = imread("..\\..\\data\\you1.jpg", 1);

		//if (!orig1.data) {
		//	return -1;
		//}
		//if (!orig2.data) {
		//	return -1;
		//}

		//cout << orig1.size() << endl;

		Mat image1 = Mat(orig1.size(), orig1.type());
		Mat image2 = Mat(orig2.size(), orig2.type());

		//Mat cameramatrix1 = (Mat_<double>(3, 3) << 839.123217919623, 0, 0, 2.89934947606147, 848.415989974001, 0, 591.692291960781, 328.139423356269, 1);
		Mat cameramatrix1 = (Mat_<double>(3, 3) << 839.117168065698, 0, 0,
			4.69323500542568, 847.074745464064, 0,
			594.221472972743, 333.392295488636, 1
			);
		cameramatrix1 = cameramatrix1.t();

		Mat cameramatrix2 = (Mat_<double>(3, 3) << 839.134307239760, 0, 0,
			3.91628695549750, 851.308496526333, 0,
			653.568783645606, 318.445627752491, 1
			);
		cameramatrix2 = cameramatrix2.t();

		Mat distcoeffs1 = (Mat_<double>(4, 1) << -0.461743276094410, 0.195973401440154, 0.00161856065236508, -0.000241795103464820);

		Mat distcoeffs2 = (Mat_<double>(4, 1) << -0.456408479090023, 0.191923480695735, -0.00277267665878999, -6.12052415459897e-05);


		Size orig1size = orig1.size();
		Size orig2size = orig2.size();

		initUndistortRectifyMap(cameramatrix1, distcoeffs1, Mat(), getOptimalNewCameraMatrix(cameramatrix1, distcoeffs1, orig1size, 0, orig1size, 0), orig1size, CV_16SC2, map1, map2);
		initUndistortRectifyMap(cameramatrix2, distcoeffs2, Mat(), getOptimalNewCameraMatrix(cameramatrix2, distcoeffs2, orig2size, 0, orig1size, 0), orig2size, CV_16SC2, map3, map4);

		remap(orig1, image1, map1, map2, INTER_LINEAR);
		remap(orig2, image2, map3, map4, INTER_LINEAR);

		//cout << image1.size() << endl;

		//imshow("orig", orig2);
		//imshow("image", image2);

		//Ptr<ORB>orb = ORB::create("ORB");

		//提取特征点    
		OrbFeatureDetector Detector(500);
		vector<KeyPoint> keyPoint1, keyPoint2;
		Detector.detect(image1, keyPoint1);
		Detector.detect(image2, keyPoint2);

		//特征点描述，为下边的特征点匹配做准备    
		OrbDescriptorExtractor Descriptor;
		Mat imageDesc1, imageDesc2;
		Descriptor.compute(image1, keyPoint1, imageDesc1);
		Descriptor.compute(image2, keyPoint2, imageDesc2); 



		//-- Step 3: Matching descriptor vectors using FLANN matcher
		//FlannBasedMatcher matcher;
		BFMatcher matcher(NORM_HAMMING);
		std::vector< DMatch > matches;
		matcher.match(imageDesc2, imageDesc1, matches);

		double max_dist = 0; double min_dist = 1000;

		//-- Quick calculation of max and min distances between keypoints
		for (int i = 0; i < imageDesc2.rows; i++)
		{
			double dist = matches[i].distance;
			if (dist < min_dist) min_dist = dist;
			if (dist > max_dist) max_dist = dist;
		}

		//printf("-- Max dist : %f \n", max_dist);
		//printf("-- Min dist : %f \n", min_dist);

		//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
		std::vector< DMatch > good_matches;

		for (int i = 0; i < imageDesc2.rows; i++)
		{
			if (matches[i].distance < 3* min_dist)
			{
				good_matches.push_back(matches[i]);
			}
		}
		//cout << good_matches.size() << endl;



		//BruteForceMatcher<L2<float>> matcher;
		//vector<vector<DMatch>>matches1;
		//matcher.knnMatch(imageDesc1, imageDesc2, matches1, 2);

		////vector < vector<DMatch>> matches2;
		////matcher.knnMatch(imageDesc2, imageDesc1, matches2, 2);

		//vector<DMatch> good_matches;
		//for (vector<vector<DMatch>>::iterator matchiterator = matches1.begin(); matchiterator != matches1.end(); ++matchiterator) {

		//	if (matchiterator->size() > 1) {

		//		if ((*matchiterator)[0].distance / (*matchiterator)[1].distance <= 0.6) {

		//			good_matches.push_back((*matchiterator)[0]);
		//		}
		//	
		//	
		//	}
		//	
		//}

		//cout << good_matches.size() << endl;




		//Mat img_matches;
		//drawMatches(imageDesc1, keyPoint1, imageDesc2, keyPoint2,
		//	good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		//	vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		//imshow("match", img_matches);

		//-- Localize the object
		std::vector<Point2f> imagePoints1;
		std::vector<Point2f> imagePoints2;

		for (int i = 0; i < good_matches.size(); i++)
		{
			//-- Get the keypoints from the good matches
			imagePoints2.push_back(keyPoint2[good_matches[i].queryIdx].pt);
			imagePoints1.push_back(keyPoint1[good_matches[i].trainIdx].pt);
		}

		////特征点描述，为下边的特征点匹配做准备    
		//OrbDescriptorExtractor OrbDescriptor;
		//Mat imageDesc1, imageDesc2;
		//OrbDescriptor.compute(image1, keyPoint1, imageDesc1);
		//OrbDescriptor.compute(image2, keyPoint2, imageDesc2);

		//flann::Index flannIndex(imageDesc1, flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);

		//vector<DMatch> GoodMatchePoints;

		//Mat macthIndex(imageDesc2.rows, 2, CV_32SC1), matchDistance(imageDesc2.rows, 2, CV_32FC1);
		//flannIndex.knnSearch(imageDesc2, macthIndex, matchDistance, 2, flann::SearchParams());

		//// Lowe's algorithm,获取优秀匹配点
		//for (int i = 0; i < matchDistance.rows; i++)
		//{
		//	if (matchDistance.at<float>(i, 0) < 0.8* matchDistance.at<float>(i, 1))
		//	{
		//		DMatch dmatches(i, macthIndex.at<int>(i, 0), matchDistance.at<float>(i, 0));
		//		GoodMatchePoints.push_back(dmatches);
		//	}
		//}

		//vector<Point2f> imagePoints1, imagePoints2;

		//cout << GoodMatchePoints.size() << endl;

		//for (int i = 0; i<GoodMatchePoints.size(); i++)
		//{
		//	imagePoints2.push_back(keyPoint2[GoodMatchePoints[i].queryIdx].pt);
		//	imagePoints1.push_back(keyPoint1[GoodMatchePoints[i].trainIdx].pt);
		//}

		//获取图像1到图像2的投影映射矩阵 尺寸为3*3  
		Mat homo = findHomography(imagePoints2, imagePoints1, CV_RANSAC);
		
		for (;;)
		{
			
			cap1 >> orig1;//cap::read()
			cap2 >> orig2;
			//cout << '2' << endl;
	
			//while (orig1.empty())
			//{
			//	//cout <<'3'<< endl;
			//	cap1 >> orig1;
			//}
			//while (orig2.empty())
			//{
			//	//cout <<'3'<< endl;
			//	cap2 >> orig2;
			//}
	
	
			remap(orig1, image1, map1, map2, INTER_LINEAR);
			remap(orig2, image2, map3, map4, INTER_LINEAR);
	
			imshow("cam1", image1);
	

			imshow("cam2", image2);

			//Mat image01 = imread("..\\..\\data\\lena.jpg", 1);    //右图
			//Mat image02 = imread("..\\..\\data\\lenafind.jpg", 1);    //左图

			//灰度图转换  

			//cvtColor(jiaozheng1, image1, CV_RGB2GRAY);
			//cvtColor(jiaozheng2, image2, CV_RGB2GRAY);

			if (waitKey(1) == 13) {

				//提取特征点    
				SurfFeatureDetector Detector(1000);
				vector<KeyPoint> keyPoint1, keyPoint2;
				Detector.detect(image1, keyPoint1);
				Detector.detect(image2, keyPoint2);

				//特征点描述，为下边的特征点匹配做准备    
				SurfDescriptorExtractor Descriptor;
				Mat imageDesc1, imageDesc2;
				Descriptor.compute(image1, keyPoint1, imageDesc1);
				Descriptor.compute(image2, keyPoint2, imageDesc2);



				//-- Step 3: Matching descriptor vectors using FLANN matcher
				FlannBasedMatcher matcher;
				//BFMatcher matcher(NORM_HAMMING);
				std::vector< DMatch > matches;
				matcher.match(imageDesc2, imageDesc1, matches);

				double max_dist = 0; double min_dist = 1000;

				//-- Quick calculation of max and min distances between keypoints
				for (int i = 0; i < imageDesc2.rows; i++)
				{
					double dist = matches[i].distance;
					if (dist < min_dist) min_dist = dist;
					if (dist > max_dist) max_dist = dist;
				}

				//printf("-- Max dist : %f \n", max_dist);
				//printf("-- Min dist : %f \n", min_dist);

				//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
				std::vector< DMatch > good_matches;

				for (int i = 0; i < imageDesc2.rows; i++)
				{
					if (matches[i].distance < 5 * min_dist)
					{
						good_matches.push_back(matches[i]);
					}
				}

				//获取图像1到图像2的投影映射矩阵 尺寸为3*3  
				Mat homo = findHomography(imagePoints2, imagePoints1, CV_RANSAC);
				
			}
			



			//-- Get the corners from the image_1 ( the object to be "detected" )
			std::vector<Point2f> obj_corners(4);
			obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(image2.cols, 0);
			obj_corners[2] = cvPoint(image2.cols, image1.rows); obj_corners[3] = cvPoint(0, image2.rows);
			std::vector<Point2f> scene_corners(4);

			//perspectiveTransform(obj_corners, scene_corners, homo);

			//图像配准  
			Mat imageTransform1;
			warpPerspective(image2, imageTransform1, homo, Size(MAX(scene_corners[1].x, scene_corners[2].x), image1.rows));
			//warpPerspective(image2, imageTransform1, homo, Size(2*image1.cols, image1.rows));
			//warpPerspective(image01, imageTransform2, adjustMat*homo, Size(image02.cols*1.3, image02.rows*1.8));
			imshow("直接经过透视矩阵变换", imageTransform1);
			//imwrite("trans1.jpg", imageTransform1);

			//Mat half(imageTransform1, Rect(0, 0, image1.cols, image1.rows));
			//image1.copyTo(half);
			//imshow("dst", half);

			//创建拼接后的图,需提前计算图的大小
			int dst_width = MAX(imageTransform1.cols,image1.cols);  //取最右点的长度为拼接图的长度
			int dst_height = image1.rows;// +imageTransform1.rows;

			Mat dst(dst_height, dst_width, CV_8UC3);
			dst.setTo(0);

			imageTransform1.copyTo(dst(Rect(0, 0, imageTransform1.cols, imageTransform1.rows)));
			image1.copyTo(dst(Rect(0, 0, image1.cols, image1.rows)));
			  
			imshow("b_dst", dst);


			//OptimizeSeam(jiaozheng2, imageTransform1, dst,scene_corners[0],scene_corners[3]);

			  
			//imshow("dst", dst);
			//imwrite("dst.jpg", dst);

	
	

			if (waitKey(1 ) ==32)
				break;
		}
	
	//Mat image01 = imread("..\\..\\data\\lena.jpg", 1);    //右图
	//Mat image02 = imread("..\\..\\data\\lenafind.jpg", 1);    //左图
	//imshow("p2", image01);
	//imshow("p1", image02);

	
	return 0;
}


//优化两图的连接处，使得拼接自然
void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst, Point2f& a, Point2f& b)
{
	int start = MIN(a.x, b.x);//开始位置，即重叠区域的左边界  

	double processWidth = abs(img1.cols - start);//重叠区域的宽度  
	int rows = dst.rows;
	int cols = img1.cols; //注意，是列数*通道数
	double alpha = 1;//img1中像素的权重  

	if ((img1.cols - trans.cols) >= 0) {

	}

	for (int i = 0; i < rows; i++)
	{
		uchar* p = img1.ptr<uchar>(i);  //获取第i行的首地址
		uchar* t = trans.ptr<uchar>(i);
		uchar* d = dst.ptr<uchar>(i);
		for (int j = start; j < cols; j++)
		{
			//如果遇到图像trans中无像素的黑点，则完全拷贝img1中的数据
			if (t[j * 3] == 0 && t[j * 3 + 1] == 0 && t[j * 3 + 2] == 0)
			{
				alpha = 1;
			}
			else
			{
				//img1中像素的权重，与当前处理点距重叠区域左边界的距离成正比，实验证明，这种方法确实好  
				alpha = (processWidth - (j - start)) / processWidth;
			}

			d[j * 3] = p[j * 3] * alpha + t[j * 3] * (1 - alpha);
			d[j * 3 + 1] = p[j * 3 + 1] * alpha + t[j * 3 + 1] * (1 - alpha);
			d[j * 3 + 2] = p[j * 3 + 2] * alpha + t[j * 3 + 2] * (1 - alpha);

		}
	}

}
