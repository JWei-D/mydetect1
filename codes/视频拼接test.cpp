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


//�̳���CvSVM���࣬��Ϊ����setSVMDetector()���õ��ļ���Ӳ���ʱ����Ҫ�õ�ѵ���õ�SVM��decision_func������
//��ͨ���鿴CvSVMԴ���֪decision_func������protected���ͱ������޷�ֱ�ӷ��ʵ���ֻ�ܼ̳�֮��ͨ����������
class MySVM : public CvSVM
{
public:
	//���SVM�ľ��ߺ����е�alpha����
	double * get_alpha_vector()
	{
		return this->decision_func->alpha;
	}

	//���SVM�ľ��ߺ����е�rho����,��ƫ����
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



	////��ȡ������    
	//SurfFeatureDetector Detector(5000);
	//vector<KeyPoint> keyPoint1, keyPoint2;
	//Detector.detect(frame1, keyPoint1);
	//Detector.detect(frame2, keyPoint2);
	//
	////������������Ϊ�±ߵ�������ƥ����׼��    
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

	//// Lowe's algorithm,��ȡ����ƥ���
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


	//��ȡͼ��1��ͼ��2��ͶӰӳ����� �ߴ�Ϊ3*3  
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
	////ͼ����׼  
	//Mat imageTransform1;
	////warpPerspective(frame2, imageTransform1, homo, Size(MAX(scene_corners[1].x, scene_corners[2].x), frame1.rows));
	//warpPerspective(frame2, imageTransform1, homo, Size(2*frame2.cols, frame2.rows));
	////warpPerspective(image01, imageTransform2, adjustMat*homo, Size(image02.cols*1.3, image02.rows*1.8));
	////imshow("ֱ�Ӿ���͸�Ӿ���任", imageTransform1);
	////imwrite("trans1.jpg", imageTransform1);
	//
	////Mat half(imageTransform1, Rect(0, 0, image1.cols, image1.rows));
	////image1.copyTo(half);
	////imshow("dst", half);
	//
	////����ƴ�Ӻ��ͼ,����ǰ����ͼ�Ĵ�С
	//int dst_width = MAX(imageTransform1.cols,frame1.cols);  //ȡ���ҵ�ĳ���Ϊƴ��ͼ�ĳ���
	//int dst_height = frame1.rows;// +imageTransform1.rows;
	//
	//Mat dst(dst_height, dst_width, CV_8UC3);
	//dst.setTo(0);
	//
	//imageTransform1.copyTo(dst(Rect(0, 0, imageTransform1.cols, imageTransform1.rows)));
	//frame1.copyTo(dst(Rect(0, 0, frame1.cols, frame1.rows)));

	////HOG+SVM���˼��
	////��1������hog������
	//HOGDescriptor hog;
	////��2������SVM������
	//hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
	//////��3���ڲ���ͼ���ϼ����������
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

	//��ⴰ��(64,128),��ߴ�(16,16),�鲽��(8,8),cell�ߴ�(8,8),ֱ��ͼbin����9
	//HOGDescriptor hog(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);//HOG���������������HOG�����ӵ�
	int DescriptorDim;//HOG�����ӵ�ά������ͼƬ��С����ⴰ�ڴ�С�����С��ϸ����Ԫ��ֱ��ͼbin��������
	MySVM svm;//SVM������

	svm.load("SVM_HOG_2400PosINRIA_12000Neg_HardExample(������©�����1).xml");//��XML�ļ���ȡѵ���õ�SVMģ��

	DescriptorDim = svm.get_var_count();//����������ά������HOG�����ӵ�ά��
	int supportVectorNum = svm.get_support_vector_count();//֧�������ĸ���
	cout << "֧������������" << supportVectorNum << endl;

	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha���������ȵ���֧����������
	Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//֧����������
	Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha��������֧����������Ľ��

														   //��֧�����������ݸ��Ƶ�supportVectorMat������
	for (int i = 0; i<supportVectorNum; i++)
	{
		const float * pSVData = svm.get_support_vector(i);//���ص�i��֧������������ָ��
		for (int j = 0; j<DescriptorDim; j++)
		{
			//cout<<pData[j]<<" ";
			supportVectorMat.at<float>(i, j) = pSVData[j];
		}
	}

	//��alpha���������ݸ��Ƶ�alphaMat��
	double * pAlphaData = svm.get_alpha_vector();//����SVM�ľ��ߺ����е�alpha����
	for (int i = 0; i<supportVectorNum; i++)
	{
		alphaMat.at<float>(0, i) = pAlphaData[i];
	}

	//����-(alphaMat * supportVectorMat),����ŵ�resultMat��
	//gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);//��֪��Ϊʲô�Ӹ��ţ�
	resultMat = -1 * alphaMat * supportVectorMat;

	//�õ����յ�setSVMDetector(const vector<float>& detector)�����п��õļ����
	vector<float> myDetector;
	//��resultMat�е����ݸ��Ƶ�����myDetector��
	for (int i = 0; i<DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0, i));
	}
	//������ƫ����rho���õ������
	myDetector.push_back(svm.get_rho());
	cout << "�����ά����" << myDetector.size() << endl;
	//����HOGDescriptor�ļ����
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

		//ͼ����׼  
		Mat imageTransform1;
		//warpPerspective(frame2, imageTransform1, homo, Size(MAX(scene_corners[1].x, scene_corners[2].x), frame1.rows));
		warpPerspective(frame2, imageTransform1, homo, Size(2* frame2.cols, frame2.rows));
		//warpPerspective(image01, imageTransform2, adjustMat*homo, Size(image02.cols*1.3, image02.rows*1.8));
		//imshow("ֱ�Ӿ���͸�Ӿ���任", imageTransform1);
		//imwrite("trans1.jpg", imageTransform1);

		//Mat half(imageTransform1, Rect(0, 0, image1.cols, image1.rows));
		//image1.copyTo(half);
		//imshow("dst", half);

		//����ƴ�Ӻ��ͼ,����ǰ����ͼ�Ĵ�С
		int dst_width = imageTransform1.cols;  //ȡ���ҵ�ĳ���Ϊƴ��ͼ�ĳ���
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

		//��3���ڲ���ͼ���ϼ����������
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


