//#include "highgui/highgui.hpp"    
//#include "opencv2/nonfree/nonfree.hpp"    
//#include "opencv2/legacy/legacy.hpp"   
#include <iostream>

#include <opencv2/opencv.hpp>
//#include <iostream>
//#include <fstream>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/objdetect/objdetect.hpp>
//#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;


//继承自CvSVM的类，因为生成setSVMDetector()中用到的检测子参数时，需要用到训练好的SVM的decision_func参数，
//但通过查看CvSVM源码可知decision_func参数是protected类型变量，无法直接访问到，只能继承之后通过函数访问
class MySVM : public CvSVM
{
public:
	//获得SVM的决策函数中的alpha数组
	double * get_alpha_vector()
	{
		return this->decision_func->alpha;
	}

	//获得SVM的决策函数中的rho参数,即偏移量
	float get_rho()
	{
		return this->decision_func->rho;
	}
};


void main() {

	VideoCapture cap1, cap2;

	cap1.open("..\\..\\data\\new20.avi");
	cap2.open("..\\..\\data\\new30.avi");

	if (!cap1.isOpened()) {

		cerr << "can not open file1!" << endl;
	
	}

	if (!cap2.isOpened()) {

		cerr << "can not open file2!" << endl;
	
	}
	
	Mat frame1;
	Mat frame2;

	for (int i = 1; i < 41; i++) 
	{
		cap1 >> frame1;//cap::read()
		cap2 >> frame2;
	}


	while (frame1.empty())
	{
		cout <<"frame1 is empty"<< endl;
		//cap1 >> frame1;
	
	}
	while (frame2.empty())
	{
		cout << "frame2 is empty" << endl;
		//cap2 >> frame2;
	
	}



	////提取特征点    
	//SurfFeatureDetector Detector(5000);
	//vector<KeyPoint> keyPoint1, keyPoint2;
	//Detector.detect(frame1, keyPoint1);
	//Detector.detect(frame2, keyPoint2);
	//
	////特征点描述，为下边的特征点匹配做准备    
	//SurfDescriptorExtractor Descriptor;
	//Mat imageDesc1, imageDesc2;
	//Descriptor.compute(frame1, keyPoint1, imageDesc1);   
	//Descriptor.compute(frame2, keyPoint2, imageDesc2); 
	//
	//
	//
	//////-- Step 3: Matching descriptor vectors using FLANN matcher
	//////FlannBasedMatcher matcher;
	////FlannBasedMatcher matcher;
	////std::vector< DMatch > matches;
	////matcher.match(imageDesc1, imageDesc2, matches);
	////
	////double max_dist = 0; double min_dist = 1000;
	////
	//////-- Quick calculation of max and min distances between keypoints
	////for (int i = 0; i < imageDesc2.rows; i++)
	////{
	////	double dist = matches[i].distance;
	////	if (dist < min_dist) min_dist = dist;
	////	if (dist > max_dist) max_dist = dist;
	////}
	////
	//////printf("-- Max dist : %f \n", max_dist);
	//////printf("-- Min dist : %f \n", min_dist);
	////
	//////-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
	////std::vector< DMatch > good_matches;
	////
	////for (int i = 0; i < imageDesc2.rows; i++)
	////{
	////	if (matches[i].distance < 8* min_dist)
	////	{
	////		good_matches.push_back(matches[i]);
	////	}
	////}
	////cout << good_matches.size() << endl;

	//FlannBasedMatcher matcher;
	//vector<vector<DMatch> > matchePoints;
	////std::vector< DMatch > matchePoints;
	//vector<DMatch> good_matches;

	//vector<Mat> train_desc(1, imageDesc2);
	//matcher.add(train_desc);
	//matcher.train();

	//matcher.knnMatch(imageDesc1, matchePoints,5);

	////matcher.match(imageDesc2, imageDesc1, matchePoints);

	//cout << "total match points: " << matchePoints.size() << endl;

	//// Lowe's algorithm,获取优秀匹配点
	//for (int i = 0; i < matchePoints.size(); i++)
	//{
	//	if (matchePoints[i][0].distance < 0.5* matchePoints[i][1].distance)
	//	{
	//		good_matches.push_back(matchePoints[i][0]);
	//	}
	//}

	//cout << "total good match points: " << good_matches.size() << endl;


	////-- Localize the object
	//std::vector<Point2f> imagePoints1;
	//std::vector<Point2f> imagePoints2;
	//
	//for (int i = 0; i < good_matches.size(); i++)
	//{
	//	//-- Get the keypoints from the good matches
	//	imagePoints1.push_back(keyPoint1[good_matches[i].queryIdx].pt);
	//	imagePoints2.push_back(keyPoint2[good_matches[i].trainIdx].pt);
	//}

	std::vector<Point2f> imagePoints1, imagePoints2;

	
	imagePoints1.push_back(Point2f(785, 401));
	imagePoints1.push_back(Point2f(800, 527));
	imagePoints1.push_back(Point2f(825, 536));
	imagePoints1.push_back(Point2f(1135, 408));
	imagePoints1.push_back(Point2f(1231, 540));


	imagePoints2.push_back(Point2f(191, 381));
	imagePoints2.push_back(Point2f(200, 528));
	imagePoints2.push_back(Point2f(239, 529));
	imagePoints2.push_back(Point2f(549, 371));
	imagePoints2.push_back(Point2f(589, 463));


	//获取图像1到图像2的投影映射矩阵 尺寸为3*3  
	Mat homo = findHomography(imagePoints2, imagePoints1, CV_RANSAC);

	//FileStorage fs("..\\..\\data\\homo.xml", FileStorage::WRITE);
	//fs << "homo" << homo;
	//fs.release();

	//FileStorage fss("..\\..\\data\\homo.xml", FileStorage::READ);
	//Mat homo;
	//fss["homo"] >> homo;

	//-- Get the corners from the image_1 ( the object to be "detected" )
	//std::vector<Point2f> obj_corners(4);
	//obj_corners[0] = cvPoint(0, 0); 
	//obj_corners[1] = cvPoint(frame2.cols, 0);
	//obj_corners[2] = cvPoint(frame2.cols, frame2.rows);
	//obj_corners[3] = cvPoint(0, frame2.rows);
	//std::vector<Point2f> scene_corners(4);
	//
	//perspectiveTransform(obj_corners, scene_corners, homo);
	//
	////图像配准  
	//Mat imageTransform1;
	////warpPerspective(frame2, imageTransform1, homo, Size(MAX(scene_corners[1].x, scene_corners[2].x), frame1.rows));
	//warpPerspective(frame2, imageTransform1, homo, Size(2*frame2.cols, frame2.rows));
	////warpPerspective(image01, imageTransform2, adjustMat*homo, Size(image02.cols*1.3, image02.rows*1.8));
	////imshow("直接经过透视矩阵变换", imageTransform1);
	////imwrite("trans1.jpg", imageTransform1);
	//
	////Mat half(imageTransform1, Rect(0, 0, image1.cols, image1.rows));
	////image1.copyTo(half);
	////imshow("dst", half);
	//
	////创建拼接后的图,需提前计算图的大小
	//int dst_width = MAX(imageTransform1.cols,frame1.cols);  //取最右点的长度为拼接图的长度
	//int dst_height = frame1.rows;// +imageTransform1.rows;
	//
	//Mat dst(dst_height, dst_width, CV_8UC3);
	//dst.setTo(0);
	//
	//imageTransform1.copyTo(dst(Rect(0, 0, imageTransform1.cols, imageTransform1.rows)));
	//frame1.copyTo(dst(Rect(0, 0, frame1.cols, frame1.rows)));

	////HOG+SVM行人检测
	////【1】定义hog描述符
	//HOGDescriptor hog;
	////【2】设置SVM分类器
	//hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
	//////【3】在测试图像上检测行人区域
	////std::vector<cv::Rect> regions;
	////hog.detectMultiScale(dst, regions, 0, cv::Size(4, 4), cv::Size(32, 32), 1.05, 1);

	////for (size_t i = 0; i < regions.size(); i++)
	////{
	////	cv::rectangle(dst, regions[i], cv::Scalar(0, 0, 255), 2);
	////}
	//			  
	////imshow("cam1", frame1);
	////imshow("cam2", frame2);
	//imshow("b_dst", dst);

	//检测窗口(64,128),块尺寸(16,16),块步长(8,8),cell尺寸(8,8),直方图bin个数9
	//HOGDescriptor hog(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);//HOG检测器，用来计算HOG描述子的
	int DescriptorDim;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定
	MySVM svm;//SVM分类器

	svm.load("SVM_HOG_2400PosINRIA_12000Neg_HardExample(误报少了漏检多了1).xml");//从XML文件读取训练好的SVM模型

	DescriptorDim = svm.get_var_count();//特征向量的维数，即HOG描述子的维数
	int supportVectorNum = svm.get_support_vector_count();//支持向量的个数
	cout << "支持向量个数：" << supportVectorNum << endl;

	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha向量，长度等于支持向量个数
	Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//支持向量矩阵
	Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha向量乘以支持向量矩阵的结果

														   //将支持向量的数据复制到supportVectorMat矩阵中
	for (int i = 0; i<supportVectorNum; i++)
	{
		const float * pSVData = svm.get_support_vector(i);//返回第i个支持向量的数据指针
		for (int j = 0; j<DescriptorDim; j++)
		{
			//cout<<pData[j]<<" ";
			supportVectorMat.at<float>(i, j) = pSVData[j];
		}
	}

	//将alpha向量的数据复制到alphaMat中
	double * pAlphaData = svm.get_alpha_vector();//返回SVM的决策函数中的alpha向量
	for (int i = 0; i<supportVectorNum; i++)
	{
		alphaMat.at<float>(0, i) = pAlphaData[i];
	}

	//计算-(alphaMat * supportVectorMat),结果放到resultMat中
	//gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);//不知道为什么加负号？
	resultMat = -1 * alphaMat * supportVectorMat;

	//得到最终的setSVMDetector(const vector<float>& detector)参数中可用的检测子
	vector<float> myDetector;
	//将resultMat中的数据复制到数组myDetector中
	for (int i = 0; i<DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0, i));
	}
	//最后添加偏移量rho，得到检测子
	myDetector.push_back(svm.get_rho());
	cout << "检测子维数：" << myDetector.size() << endl;
	//设置HOGDescriptor的检测子
	HOGDescriptor myHOG;
	myHOG.setSVMDetector(myDetector);

	//waitKey(0);
	

	for (;;)
	{
;
		cap1 >> frame1;//cap::read()
		cap2 >> frame2;


		while (frame1.empty())
		{
			cout << "frame1 is empty" << endl;
			//cap1 >> frame1;
	
		}
		while (frame2.empty())
		{
			cout << "frame2 is empty" << endl;
			//cap2 >> frame2;

		}

		//图像配准  
		Mat imageTransform1;
		//warpPerspective(frame2, imageTransform1, homo, Size(MAX(scene_corners[1].x, scene_corners[2].x), frame1.rows));
		warpPerspective(frame2, imageTransform1, homo, Size(2* frame2.cols, frame2.rows));
		//warpPerspective(image01, imageTransform2, adjustMat*homo, Size(image02.cols*1.3, image02.rows*1.8));
		//imshow("直接经过透视矩阵变换", imageTransform1);
		//imwrite("trans1.jpg", imageTransform1);

		//Mat half(imageTransform1, Rect(0, 0, image1.cols, image1.rows));
		//image1.copyTo(half);
		//imshow("dst", half);

		//创建拼接后的图,需提前计算图的大小
		int dst_width = imageTransform1.cols;  //取最右点的长度为拼接图的长度
		int dst_height = frame1.rows;// +imageTransform1.rows;

		Mat dst(dst_height, dst_width, CV_8UC3);
		dst.setTo(0);

		imageTransform1.copyTo(dst(Rect(0, 0, imageTransform1.cols, imageTransform1.rows)));
		frame1.copyTo(dst(Rect(0, 0, frame1.cols, frame1.rows)));

		int d =50;

		for (int i = 0; i < d; i++)
		{
			dst.col(frame1.cols - d/2 + i) = (d - i) / (float)d*dst.col(frame1.cols - d/2 + i) + (i) / (float)d*imageTransform1.col(frame1.cols - d/2 + i);
		}

		//【3】在测试图像上检测行人区域
		std::vector<cv::Rect> regions;
		//hog.detectMultiScale(dst, regions, 0, cv::Size(4, 5), cv::Size(0, 0), 1.02, 1);//20,30
		myHOG.detectMultiScale(dst, regions, 0, cv::Size(8, 12), cv::Size(0, 0), 1.02, 1);//20,30


		for (size_t i = 0; i < regions.size(); i++)
		{
			cv::rectangle(dst, regions[i], cv::Scalar(0, 0, 255), 2);
		}

		//imshow("cam1", frame1);
		//imshow("cam2", frame2);
		imshow("b_dst", dst);

		if (waitKey(10) >= 0)
			break;


	}

	waitKey(0);
}


