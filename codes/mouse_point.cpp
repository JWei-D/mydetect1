#include "highgui/highgui.hpp"    
#include "opencv2/nonfree/nonfree.hpp"    
#include "opencv2/legacy/legacy.hpp"   
#include <iostream>

using namespace std;
using namespace cv;

struct userdata {

	Mat im;

	vector<Point2f> points;

	string name;

};

void mouseDealer(int event, int x, int y, int flags, void* data_ptr)

{

	if (event == EVENT_LBUTTONDOWN)

	{

		userdata *data = ((userdata *)data_ptr);

		circle(data->im, Point(x, y), 3, Scalar(0, 0, 255), 5, CV_AA);

		imshow(data->name, data->im);

		if (data->points.size() < 5)

		{
			
			data->points.push_back(Point2f(x, y));
			cout << "point-"<<data->points.size()<<" of "<<data->name<<":  "<<x << ", " << y << endl;

		}

	}



}

void main(char argc, char ** argv)

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


	//namedWindow("right", CV_WINDOW_AUTOSIZE);
	//namedWindow("left", CV_WINDOW_AUTOSIZE);
	//namedWindow("stitch", CV_WINDOW_AUTOSIZE);   //开启三个窗口


	const string videoStreamAddress1 = "http://192.168.0.129:8080/?action=stream.mjpg";
	VideoCapture cap1;
	cap1.set(CV_CAP_PROP_FRAME_WIDTH, frame_width * 2 + 1);
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
	}


	for (int i = 0; i < 10; i++) {

		cap1 >> frame;

	}

	remap(frame(rect_left), frameCalibration_left, map1_left, map2_left, INTER_LINEAR);
	remap(frame(rect_right), frameCalibration_right, map1_right, map2_right, INTER_LINEAR);
	right_image = frameCalibration_right(rect_right_final);
	left_image = frameCalibration_left(rect_left_final);
	//right_image = frame(rect_right);
	//left_image = frame(rect_left);

	//imshow("right", right_image);
	//imshow("left", left_image);////————————————————

	cout << "Click on the four corners of the book -- 顺时针方向" << endl;

	userdata data1, data2;

	data1.im = left_image.clone();
	data1.name = "left";
	data2.im = right_image.clone();
	data2.name = "right";

	imshow(data1.name, data1.im);
	imshow(data2.name, data2.im);

	//cout << "4 points of frame1&2:" << endl;
	setMouseCallback(data1.name, mouseDealer, &data1);
	setMouseCallback(data2.name, mouseDealer, &data2);

	waitKey(0);

	Mat homo = findHomography(data2.points, data1.points);

	//Size dstsize = (frame2.rows, 2*frame2.cols);
	int dst_width = 2*right_image.cols;
	int dsts_height = right_image.rows;
	Mat dst = Mat::zeros(Size(dst_width,dsts_height), CV_8UC3);
	warpPerspective(right_image, dst, homo, Size(dst_width,dsts_height));

	imshow("frame2dst", dst);

	waitKey(0);



	//while (!stop)
	//{
	//	if (cap1.read(frame))
	//	{

	//		remap(frame(rect_left), frameCalibration_left, map1_left, map2_left, INTER_LINEAR);
	//		remap(frame(rect_right), frameCalibration_right, map1_right, map2_right, INTER_LINEAR);
	//		right_image = frameCalibration_right(rect_right_final);
	//		left_image = frameCalibration_left(rect_left_final);
	//		//right_image = frame(rect_right);
	//		//left_image = frame(rect_left);

	//		imshow("right", right_image);
	//		imshow("left", left_image);////————————————————




	//		//计算单应矩阵
	//		if (k < 1 || waitKey(delay) == 13)
	//		{
	//			cout << "正在匹配..." << endl;
	//			////////////////////////////////
	//			vector<KeyPoint> keypoints1, keypoints2;
	//			//构造检测器
	//			Ptr<FeatureDetector> detector = new ORB(120);
	//			//Ptr<FeatureDetector> detector = new SIFT(80);
	//			detector->detect(right_image, keypoints1);
	//			detector->detect(left_image, keypoints2);
	//			//构造描述子提取器
	//			Ptr<DescriptorExtractor> descriptor = detector;
	//			//提取描述子
	//			Mat descriptors1, descriptors2;

	//			descriptor->compute(right_image, keypoints1, descriptors1);
	//			descriptor->compute(left_image, keypoints2, descriptors2);
	//			//构造匹配器
	//			BFMatcher matcher(NORM_L2, true);
	//			//匹配描述子
	//			vector<DMatch> matches;
	//			matcher.match(descriptors1, descriptors2, matches);

	//			vector<Point2f> selPoints1, selPoints2;
	//			vector<int> pointIndexes1, pointIndexes2;
	//			for (vector<DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it)
	//			{
	//				selPoints1.push_back(keypoints1.at(it->queryIdx).pt);
	//				selPoints2.push_back(keypoints2.at(it->trainIdx).pt);
	//			}

	//			vector<uchar> inliers(selPoints1.size(), 0);
	//			homography = findHomography(selPoints1, selPoints2, inliers, CV_FM_RANSAC, 1.0);
	//			k++;


	//		}

	//		warpPerspective(right_image, result, homography, Size(2.5 * right_image.cols - d, right_image.rows));//Size设置结果图像宽度，宽度裁去一部分，d可调，右侧图像已经经过透视变换


	//		Mat half(result, Rect(0, 0, left_image.cols - d, left_image.rows));  //把result左侧图像部分拷贝给half
	//		left_image(Range::all(), Range(0, left_image.cols - d)).copyTo(half);//把左侧图像拷贝给half,相当于拷贝给了result

	//		for (int i = 0; i < d; i++)
	//		{
	//			result.col(left_image.cols - d + i) = (d - i) / (float)d*left_image.col(left_image.cols - d + i) + i / (float)d*result.col(left_image.cols - d + i);
	//		}
	//		imshow("stitch", result);
	//	}
	//	else
	//	{
	//		cout << "----------------------" << endl;
	//		cout << "waitting..." << endl;
	//	}

	//	if (waitKey(1) == 27)
	//	{
	//		stop = true;
	//		cout << "程序结束！" << endl;
	//		cout << "*** ***" << endl;
	//	}
	//}


	//VideoCapture cap1, cap2;

	//cap1.open("..\\..\\data\\new20.avi");
	//cap2.open("..\\..\\data\\new30.avi");

	//if (!cap1.isOpened()) {

	//	cerr << "can not open file1!" << endl;
	//
	//}

	//if (!cap2.isOpened()) {

	//	cerr << "can not open file2!" << endl;
	//
	//}

	//Mat frame1;
	//Mat frame2;

	//for (int i = 1; i < 30; i++)
	//{
	//	cap1 >> frame1;//cap::read()
	//	cap2 >> frame2;
	//}


	//while (frame1.empty())
	//{
	//	cout <<"frame1_empty"<< endl;
	//	//cap1 >> frame1;
	//
	//}
	//while (frame2.empty())
	//{
	//	cout <<"frame2.empty"<< endl;
	//	//cap2 >> frame2;
	//
	//}

	////Mat frame1 = imread("..\\..\\data\\zuo2.jpg");
	////Mat frame2 = imread("..\\..\\data\\you2.jpg");

	//cout << "Click on the four corners of the book -- 顺时针方向" << endl;

	//userdata data1, data2;

	//data1.im = frame1.clone();
	//data1.name = "frame1";
	//data2.im = frame2.clone();
	//data2.name = "frame2";

	//imshow(data1.name, data1.im);
	//imshow(data2.name, data2.im);

	////cout << "4 points of frame1&2:" << endl;
	//setMouseCallback(data1.name, mouseDealer, &data1);
	//setMouseCallback(data2.name, mouseDealer, &data2);

	//waitKey(0);

	//Mat homo = findHomography(data2.points, data1.points);

	////Size dstsize = (frame2.rows, 2*frame2.cols);
	//int dst_width = 2*frame2.cols;
	//int dsts_height = frame2.rows;
	//Mat dst = Mat::zeros(Size(dst_width,dsts_height), CV_8UC3);
	//warpPerspective(frame2, dst, homo, Size(dst_width,dsts_height));

	//imshow("frame2dst", dst);

	//waitKey(0);


	//// Read source image.

	//Mat im_src = imread("..\\..\\data\\lenafind.jpg");
	//



	//// Destination image. The aspect ratio of the book is 3/4

	//Size size(300, 400);

	//Mat im_dst = Mat::zeros(size, CV_8UC3);

	//// Create a vector of destination points.

	//vector<Point2f> pts_dst;

	//pts_dst.push_back(Point2f(0, 0));

	//pts_dst.push_back(Point2f(size.width - 1, 0));

	//pts_dst.push_back(Point2f(size.width - 1, size.height - 1));

	//pts_dst.push_back(Point2f(0, size.height - 1));



	//// Set data for mouse event

	//Mat im_temp = im_src.clone();

	//userdata data;

	//data.im = im_temp;



	//cout << "Click on the four corners of the book -- 顺时针方向" << endl;

	//// Show image and wait for 4 clicks. 

	//imshow("Image", im_temp);

	//// Set the callback function for any mouse event

	//setMouseCallback("Image", mouseDealer, &data);

	//waitKey(0);



	//// Calculate the homography

	//Mat h = findHomography(data.points, pts_dst);



	//// Warp source image to destination

	//warpPerspective(im_src, im_dst, h, size);



	//// Show image

	//imshow("Image", im_dst);

	//waitKey(0);

}
