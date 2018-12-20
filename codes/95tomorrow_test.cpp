#include <opencv2/opencv.hpp>

#include<Windows.h> 

using namespace cv;

using namespace std;



struct userdata {

	Mat im;

	vector<Point2f> points;

};





void mouseHandler_left(int event, int x, int y, int flags, void* data_ptr)

{

	if (event == EVENT_LBUTTONDOWN)

	{

		userdata *data_left = ((userdata *)data_ptr);

		circle(data_left->im, Point(x, y), 3, Scalar(0, 0, 255), 5, CV_AA);

		imshow("left", data_left->im);

		if (data_left->points.size() < 4)

		{

			data_left->points.push_back(Point2f(x, y));

			cout << "x:" << x << "  y:" << y << endl;

		}

	}



}





void mouseHandler_right(int event, int x, int y, int flags, void* data_ptr)

{

	if (event == EVENT_LBUTTONDOWN)

	{

		userdata *data_right = ((userdata *)data_ptr);

		circle(data_right->im, Point(x, y), 3, Scalar(0, 0, 255), 5, CV_AA);

		imshow("right", data_right->im);

		if (data_right->points.size() < 4)

		{

			data_right->points.push_back(Point2f(x, y));

			cout << "x:" << x << "  y:" << y << endl;

		}

	}



}





void mouseHandler_middle(int event, int x, int y, int flags, void* data_ptr)

{

	if (event == EVENT_LBUTTONDOWN)

	{

		userdata *data_left = ((userdata *)data_ptr);

		circle(data_left->im, Point(x, y), 3, Scalar(0, 0, 255), 5, CV_AA);

		imshow("left2middle", data_left->im);

		if (data_left->points.size() < 4)

		{

			data_left->points.push_back(Point2f(x, y));

			cout << "x:" << x << "  y:" << y << endl;

		}

	}



}





void main()

{

	//VideoCapture capture_left(1);

	//VideoCapture capture_right(0);

	//capture_left.set(CV_CAP_PROP_FRAME_WIDTH, 640);

	//capture_left.set(CV_CAP_PROP_FRAME_HEIGHT, 360);

	//capture_right.set(CV_CAP_PROP_FRAME_WIDTH, 640);

	//capture_right.set(CV_CAP_PROP_FRAME_HEIGHT, 360);

	DWORD start_time;

	DWORD end_time;



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





	//namedWindow("right", CV_WINDOW_AUTOSIZE);

	//namedWindow("left", CV_WINDOW_AUTOSIZE);

	//namedWindow("stitch", CV_WINDOW_AUTOSIZE);   //������������





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

	

	

	//if (capture_left.isOpened() && capture_right.isOpened())

	//{

	//	cout << "*** ***" << endl;

	//	cout << "���??ͷ��������" << endl;

	//}

	//else

	//{

	//	cout << "*** ***" << endl;

	//	cout << "���棺��������ͷ�Ƿ�װ��!" << endl;

	//	cout << "���������" << endl << "*** ***" << endl;

	//	return;

	//}



	if(cap1.isOpened())

	{

		cout << "*** ***" << endl;

		cout << "����ͷ��������" << endl;

	}

	else

	{

		cout << "*** ***" << endl;

		cout << "���棺��������ͷ���??��װ��!" << endl;

		cout << "���������" << endl << "*** ***" << endl;

		getchar();



	}





	//Mat frame_left, frame_right;

	//int frame_start = 0;

	//while (frame_start < 15)        //�����ֱ������ͷ��15�ĳ�1

	//{

	//	capture_left >> frame_left;

	//	capture_right >> frame_right;

	//	frame_start++;

	//}

	



//	int d = 90;



	remap(frame(rect_left), frameCalibration_left, map1_left, map2_left, INTER_LINEAR);

	remap(frame(rect_right), frameCalibration_right, map1_right, map2_right, INTER_LINEAR);

	right_image = frameCalibration_right(rect_right_final);

	left_image = frameCalibration_left(rect_left_final);



	Mat im_src_left = left_image;

	Mat im_src_right = right_image;

	Mat im_dst_left, im_dst_right;



	Size size_double(left_image.cols * 2.5, left_image.rows);



	Mat im_temp_left = im_src_left.clone();

	userdata data_left;

	data_left.im = im_temp_left;









	Mat im_temp_right = im_src_right.clone();

	userdata data_right;

	data_right.im = im_temp_right;



	imshow("left", im_temp_left);

	setMouseCallback("left", mouseHandler_left, &data_left);

	waitKey(0);





	// Create a vector of destination points.

	vector<Point2f> data_middle;

	Point2f original_point = Point2f(440, 180);

	int original_width=121;

	int original_height = 53;

	int move_right = 230;

	data_middle.push_back(Point2f(original_point.x , original_point.y));

	data_middle.push_back(Point2f(original_point.x+original_width , original_point.y));

	data_middle.push_back(Point2f(original_point.x , original_point.y + original_height));

	data_middle.push_back(Point2f(original_point.x + original_width , original_point.y+ original_height));                                    //���??�ĸ���任���м��Ŀ���





	Mat h_left2middle = findHomography(data_left.points, data_middle);           //��ͼ�任�ľ��??


	Size size_original(left_image.cols, left_image.rows);

	// Warp source image to destination

	warpPerspective(im_src_left, im_dst_left, h_left2middle, size_original);

	

	userdata data_left_transform;

	data_left_transform.im = im_dst_left;



	



	// Show image and wait for 4 clicks.

	imshow("left2middle", im_dst_left);

	//setMouseCallback("left2middle", mouseHandler_middle, &data_left_transform);

	//waitKey(0);



	imshow("right", im_temp_right);

	setMouseCallback("right", mouseHandler_right, &data_right);

	waitKey(0);



	Mat h_right2middle = findHomography(data_right.points, data_middle);      //��ͼ�任�????��





	// Warp source image to destination

	warpPerspective(im_src_right, im_dst_right, h_right2middle, size_double);



	// Show image

	imshow("right", im_dst_right);

	waitKey(0);



	Mat half(im_dst_right, Rect(0, 0, im_temp_left.cols - d, im_temp_left.rows));  //��result���ͼ�񲿷ֿ�����half

	im_dst_left(Range::all(), Range(0, im_temp_left.cols - d)).copyTo(half);       //�����ͼ�񿽱���half,�൱�ڿ�������result





	for (int i = 0; i < d; i++)

	{

		im_dst_right.col(im_temp_left.cols - d + i) = (d - i) / (float)d* im_dst_left.col(im_temp_left.cols - d + i) + i / (float)d*im_dst_right.col(im_dst_left.cols - d + i);

	}

	//ͼ��������ں�



	imshow("stitch", im_dst_right);

	waitKey(0);

	

	int flag = 0;

	int width = im_dst_right.cols;

	int hight = im_dst_right.rows;



	while (cap1.read(frame))

	{

		start_time = GetTickCount();



		remap(frame(rect_left), frameCalibration_left, map1_left, map2_left, INTER_LINEAR);

		remap(frame(rect_right), frameCalibration_right, map1_right, map2_right, INTER_LINEAR);

		right_image = frameCalibration_right(rect_right_final);

		left_image = frameCalibration_left(rect_left_final);



		im_src_left = left_image;

		im_src_right = right_image;

		

		imshow("left", left_image);

		imshow("right", right_image);

		warpPerspective(im_src_left, im_dst_left, h_left2middle, size_original);

		im_temp_left = im_dst_left.clone();

		warpPerspective(im_src_right, im_dst_right, h_right2middle, size_double);

		Mat half(im_dst_right, Rect(0, 0, im_src_left.cols - d, im_src_left.rows));  //��result����??�񲿷ֿ�����half

		im_temp_left(Range::all(), Range(0, im_temp_left.cols - d)).copyTo(half);    //�����ͼ�񿽱���half,���????��������result



		for (int i = 0; i < d; i++)

		{

			im_dst_right.col(im_temp_left.cols - d + i) = (d - i) / (float)d*im_temp_left.col(im_temp_left.cols - d + i) + (i) / (float)d*im_dst_right.col(im_temp_left.cols - d + i);

		}



		imshow("stitch", im_dst_right(Rect(0, 0, width, hight)));

		if (flag == 0) {

			cvWaitKey(0);

			flag++;

		}

		int key = waitKey(1);



		end_time = GetTickCount();

		cout << "The run time is:" << (end_time - start_time) << "ms!" << endl;

	}

	return;

}





/*          Ѱ�Ҿ���

//���Ƶľ���

double bbb[3][3] = { 1,0,125 ,0,1,0,0,0,1 };

Mat h_move_right(3, 3, CV_64FC1, bbb);



//��ת�ľ���

double angle = 30;

double angle_unit = angle * 3.14159 / 180;

float aaa[3][3] = { cos(angle_unit),-sin(angle_unit),0, sin(angle_unit),cos(angle_unit),0 ,0,0,1 };

vector<Point2f> pts_dst22, pts_dst33;

pts_dst33.push_back(Point2f(0, 0));

pts_dst33.push_back(Point2f(0, 1));

pts_dst33.push_back(Point2f(1, 0));

pts_dst33.push_back(Point2f(1, 1));  //left



pts_dst22.push_back(Point2f(0, 0));

pts_dst22.push_back(Point2f(-sin(angle_unit), cos(angle_unit)));

pts_dst22.push_back(Point2f(cos(angle_unit), sin(angle_unit)));

pts_dst22.push_back(Point2f(cos(angle_unit) - sin(angle_unit), cos(angle_unit) + sin(angle_unit)));

Mat hhh = findHomography(pts_dst33, pts_dst22);

cout << hhh << endl;

*/
