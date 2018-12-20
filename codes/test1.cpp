
#include <stdio.h>
#include <opencv2/opencv.hpp>



using namespace std;
using namespace cv;

int main()
{
	//Mat z = Mat::zeros(3, 2, CV_8UC1);
	//cout << "Mat z = " << z << endl;
	//system("pause");
	//return 0;
	//cout << '00' << endl;
	VideoCapture  cap0;
	cap0.open(1);
	VideoCapture cap1;
	cap1.open(0);
	//cout << '01' << endl;
	if (!cap0.isOpened())
	{
		cerr << "can not open a camera 2 or file." << endl;
		system("pause");
		return -1;
	}
	if (!cap1.isOpened())
	{
		cerr << "can not open a camera 1 or file." << endl;
		system("pause");
		return -1;
	}

	Mat cam0;
	namedWindow("cam0", 1);
	Mat cam1;
	namedWindow("cam1", 1);

	for (;;)
	{
		Mat frame0;
		Mat frame1;
		//cout << '1' << endl;
		cap0 >> frame0;//cap::read()
		cap1 >> frame1;
		//cout << '2' << endl;

		while (frame0.empty())
		{
			//cout <<'3'<< endl;
			cap0 >> frame0;
		}
		while (frame1.empty())
		{
			//cout <<'3'<< endl;
			cap1 >> frame1;
		}

		//cvtColor(frame, edges, CV_BGR2GRAY);//转成灰度图
		//Canny(edges, edges, 0, 30, 3);//canny 边缘提取
		imshow("cam0", frame0);
		imshow("cam1", frame1);
		if (waitKey(30) >= 0)
			break;
	}
	system("pause");
	return 0;



}
