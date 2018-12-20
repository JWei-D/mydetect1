#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
# include "opencv2/features2d/features2d.hpp"
#include"opencv2/nonfree/nonfree.hpp"
#include"opencv2/calib3d/calib3d.hpp"
#include<iostream>


using namespace cv;
using namespace std;

int main()
{
	int frame_width = 640;   //ÿ֡�Ŀ��
	int frame_height = 480; //ÿ֡�ĸ߶�
	int cut_up = 15;    //����ͼ��У�������ͼ���α䣬�����м��ü��Ŀ��
	double rate = 60;   //֡��
	int delay = 1000 / rate;
	int d = 20;			//���뽥���ںϿ��
	bool stop(false);   //ֹͣλ
	int k = 0;          //��־λ


	Mat frame;        //������ͷ�����ÿһ֡
	Mat result;       //���ƴ�ӵĽ��
	Mat homography;   //�任����	
	Mat frameCalibration_left;  //��ĿУ��ͼ��
	Mat frameCalibration_right; //��ĿУ��ͼ��
	Mat left_image;   //��ĿУ���ü�ͼ��
	Mat right_image;  //��ĿУ���ü�ͼ��
	Mat  map1_left, map2_left;		//��ĿУ������
	Mat  map1_right, map2_right;	//��ĿУ������

	Rect rect_right(0, 0, frame_width, frame_height); //��Ŀ����
	Rect rect_left(frame_width, 0, frame_width, frame_height);//��Ŀ����
	Rect rect_right_final(cut_up, 0, frame_width - cut_up, frame_height);//��Ŀ�ü��������
	Rect rect_left_final(0, 0, frame_width - cut_up, frame_height);		 //��Ŀ�ü��������

	Mat cameraMatrix_left = Mat::eye(3, 3, CV_64F);				//��Ŀ�����������
	cameraMatrix_left.at<double>(0, 0) = 370.284329754200;
	cameraMatrix_left.at<double>(0, 1) = -0.807497420701725;
	cameraMatrix_left.at<double>(0, 2) = 311.484654416286;
	cameraMatrix_left.at<double>(1, 1) = 370.270241979614;
	cameraMatrix_left.at<double>(1, 2) = 243.750994967277;

	Mat distCoeffs_left = Mat::zeros(5, 1, CV_64F);
	distCoeffs_left.at<double>(0, 0) = -0.287717664165832;
	distCoeffs_left.at<double>(1, 0) = 0.0722479182618950;
	distCoeffs_left.at<double>(2, 0) = 0.00125236769806718;
	distCoeffs_left.at<double>(3, 0) = -0.00545614535429647;
	distCoeffs_left.at<double>(4, 0) = 0;


	Mat cameraMatrix_right = Mat::eye(3, 3, CV_64F);			//��Ŀ�������
	cameraMatrix_right.at<double>(0, 0) = 354.739591127572;
	cameraMatrix_right.at<double>(0, 1) = 1.07547546311553;
	cameraMatrix_right.at<double>(0, 2) = 319.141333252838;
	cameraMatrix_right.at<double>(1, 1) = 352.289914855028;
	cameraMatrix_right.at<double>(1, 2) = 222.904810013926;

	Mat distCoeffs_right = Mat::zeros(5, 1, CV_64F);
	distCoeffs_right.at<double>(0, 0) = -0.298767846489808;
	distCoeffs_right.at<double>(1, 0) = 0.0861247546453717;
	distCoeffs_right.at<double>(2, 0) = 0.000160002868328113;
	distCoeffs_right.at<double>(3, 0) = 0.000485586301726655;
	distCoeffs_right.at<double>(4, 0) = 0;


	namedWindow("right", CV_WINDOW_AUTOSIZE);
	namedWindow("left", CV_WINDOW_AUTOSIZE);
	namedWindow("stitch", CV_WINDOW_AUTOSIZE);   //������������

	
	const string videoStreamAddress1 = "http://192.168.0.129:8080/?action=stream.mjpg";
	VideoCapture cap1;
	cap1.set(CV_CAP_PROP_FRAME_WIDTH, frame_width*2+1);
	cap1.set(CV_CAP_PROP_FRAME_HEIGHT, frame_height);
	cap1.open(videoStreamAddress1);
	cap1 >> frame;
	
	//VideoCapture capture(1);
	//capture.set(CV_CAP_PROP_FRAME_WIDTH, frame_width*2+1);
	//capture.set(CV_CAP_PROP_FRAME_HEIGHT, frame_height);
	//capture >> frame;
	
	Size imageSize_left = frame(rect_left).size();
	Size imageSize_right = frame(rect_left).size();

	initUndistortRectifyMap(cameraMatrix_right, distCoeffs_right, Mat(),
		getOptimalNewCameraMatrix(cameraMatrix_right, distCoeffs_right, imageSize_left, 1, imageSize_left, 0),
		imageSize_left, CV_16SC2, map1_right, map2_right);
	initUndistortRectifyMap(cameraMatrix_left, distCoeffs_left, Mat(),
		getOptimalNewCameraMatrix(cameraMatrix_left, distCoeffs_left, imageSize_right, 1, imageSize_right, 0),
		imageSize_right, CV_16SC2, map1_left, map2_left);


	if (cap1.isOpened())
	{
		cout << "*** ***" << endl;
		cout << "����ͷ��������" << endl;
	}
	else
	{
		cout << "*** ***" << endl;
		cout << "���棺��������ͷ�Ƿ�װ��!" << endl;
		cout << "���������" << endl << "*** ***" << endl;
		getchar();
		return -1;
	}

	
	while (!stop)
	{
		if (cap1.read(frame))
		{

			remap(frame(rect_left), frameCalibration_left, map1_left, map2_left, INTER_LINEAR);
			remap(frame(rect_right), frameCalibration_right, map1_right, map2_right, INTER_LINEAR);
			right_image = frameCalibration_right(rect_right_final);
			left_image = frameCalibration_left(rect_left_final);
			//right_image = frame(rect_right);
			//left_image = frame(rect_left);

			imshow("right", right_image);  
			imshow("left", left_image);

			//���㵥Ӧ����
			if (k < 1 || waitKey(delay) == 13)
			{
				cout << "����ƥ��..." << endl;
				////////////////////////////////
				vector<KeyPoint> keypoints1, keypoints2;
				//��������
				Ptr<FeatureDetector> detector = new ORB(120);
				//Ptr<FeatureDetector> detector = new SIFT(80);
				detector->detect(right_image, keypoints1);
				detector->detect(left_image, keypoints2);
				//������������ȡ��
				Ptr<DescriptorExtractor> descriptor = detector; 
				//��ȡ������
				Mat descriptors1, descriptors2;

				descriptor->compute(right_image, keypoints1, descriptors1);
				descriptor->compute(left_image, keypoints2, descriptors2);
				//����ƥ����
				BFMatcher matcher(NORM_L2, true);
				//ƥ��������
				vector<DMatch> matches;
				matcher.match(descriptors1, descriptors2, matches);

				vector<Point2f> selPoints1, selPoints2;
				vector<int> pointIndexes1, pointIndexes2;
				for (vector<DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it)
				{
					selPoints1.push_back(keypoints1.at(it->queryIdx).pt);
					selPoints2.push_back(keypoints2.at(it->trainIdx).pt);
				}

				vector<uchar> inliers(selPoints1.size(), 0);
				homography = findHomography(selPoints1, selPoints2, inliers, CV_FM_RANSAC, 1.0);
				k++;


			}

			warpPerspective(right_image, result, homography, Size(2.5 * right_image.cols - d, right_image.rows ));//Size���ý��ͼ���ȣ���Ȳ�ȥһ���֣�d�ɵ����Ҳ�ͼ���Ѿ�����͸�ӱ任
			

			Mat half(result, Rect(0, 0, left_image.cols - d, left_image.rows));  //��result���ͼ�񲿷ֿ�����half
			left_image(Range::all(), Range(0, left_image.cols - d)).copyTo(half);//�����ͼ�񿽱���half,�൱�ڿ�������result
			
			for (int i = 0; i < d; i++)
			{
				result.col(left_image.cols - d + i) = (d - i) / (float)d*left_image.col(left_image.cols - d + i) + i / (float)d*result.col(left_image.cols - d + i);
			}
			imshow("stitch", result);
		}
		else
		{
			cout << "----------------------" << endl;
			cout << "waitting..." << endl;
		}

		if (waitKey(1) == 27)
		{
			stop = true;
			cout << "���������" << endl;
			cout << "*** ***" << endl;
		}
	}
	return 0;
}
