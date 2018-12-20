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
	int frame_width = 640;   //每帧的宽度
	int frame_height = 480; //每帧的高度
	int cut_up = 15;    //由于图像校正引起的图像形变，靠近中间侧裁剪的宽度
	double rate = 60;   //帧率
	int delay = 1000 / rate;
	int d = 20;			//渐入渐出融合宽度
	bool stop(false);   //停止位
	int k = 0;          //标志位


	Mat frame;        //从摄像头读入的每一帧
	Mat result;       //最后拼接的结果
	Mat homography;   //变换矩阵	
	Mat frameCalibration_left;  //左目校正图像
	Mat frameCalibration_right; //右目校正图像
	Mat left_image;   //左目校正裁剪图像
	Mat right_image;  //右目校正裁剪图像
	Mat  map1_left, map2_left;		//左目校正参数
	Mat  map1_right, map2_right;	//右目校正参数

	Rect rect_right(0, 0, frame_width, frame_height); //左目区域
	Rect rect_left(frame_width, 0, frame_width, frame_height);//右目区域
	Rect rect_right_final(cut_up, 0, frame_width - cut_up, frame_height);//左目裁剪后的区域
	Rect rect_left_final(0, 0, frame_width - cut_up, frame_height);		 //右目裁剪后的区域

	Mat cameraMatrix_left = Mat::eye(3, 3, CV_64F);				//左目畸变参数矩阵
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


	Mat cameraMatrix_right = Mat::eye(3, 3, CV_64F);			//右目畸变参数
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
	namedWindow("stitch", CV_WINDOW_AUTOSIZE);   //开启三个窗口

	
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
		cout << "摄像头已启动！" << endl;
	}
	else
	{
		cout << "*** ***" << endl;
		cout << "警告：请检查摄像头是否安装好!" << endl;
		cout << "程序结束！" << endl << "*** ***" << endl;
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

			//计算单应矩阵
			if (k < 1 || waitKey(delay) == 13)
			{
				cout << "正在匹配..." << endl;
				////////////////////////////////
				vector<KeyPoint> keypoints1, keypoints2;
				//构造检测器
				Ptr<FeatureDetector> detector = new ORB(120);
				//Ptr<FeatureDetector> detector = new SIFT(80);
				detector->detect(right_image, keypoints1);
				detector->detect(left_image, keypoints2);
				//构造描述子提取器
				Ptr<DescriptorExtractor> descriptor = detector; 
				//提取描述子
				Mat descriptors1, descriptors2;

				descriptor->compute(right_image, keypoints1, descriptors1);
				descriptor->compute(left_image, keypoints2, descriptors2);
				//构造匹配器
				BFMatcher matcher(NORM_L2, true);
				//匹配描述子
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

			warpPerspective(right_image, result, homography, Size(2.5 * right_image.cols - d, right_image.rows ));//Size设置结果图像宽度，宽度裁去一部分，d可调，右侧图像已经经过透视变换
			

			Mat half(result, Rect(0, 0, left_image.cols - d, left_image.rows));  //把result左侧图像部分拷贝给half
			left_image(Range::all(), Range(0, left_image.cols - d)).copyTo(half);//把左侧图像拷贝给half,相当于拷贝给了result
			
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
			cout << "程序结束！" << endl;
			cout << "*** ***" << endl;
		}
	}
	return 0;
}
