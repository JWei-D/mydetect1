#include "opencv2/core/core.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stdio.h>
#include <Windows.h>
#include <iostream>

using namespace std;
using namespace cv;
//string face_cascade_name = "E:\\jwei\\OpenCV2413\\opencv\\sources\\data\\haarcascadeshaarcascade_frontalface_alt.xml";
string face_cascade_name = "E:\\jwei\\OpenCV2413\\opencv\\sources\\data\\hogcascades\\hogcascade_pedestrians.xml";
//string face_cascade_name = "SVM_HOG_2400PosINRIA_12000Neg_HardExample(误报少了漏检多了1).xml";
//该文件存在于OpenCV安装目录下的\sources\data\haarcascades内，需要将该xml文件复制到当前工程目录下
CascadeClassifier face_cascade;
void detectAndDisplay(Mat frame);
int main(int argc, char** argv) {

	Mat frame1;
	VideoCapture cap1;

	cap1.open("..\\..\\data\\new30.avi");
	//cap1.set(CV_CAP_PROP_FRAME_WIDTH, 60); //画面宽
	//cap1.set(CV_CAP_PROP_FRAME_HEIGHT, 40);//画面高

	if (!cap1.isOpened()) {

		cout << "cap1 can not open" << endl;
		return -1;

	}
	//namedWindow("test");
	HWND winhand = GetActiveWindow();

	for (;;) {


		double t_frame = getTickCount();
		cap1 >> frame1;

		double t_resize = getTickCount();
		resize(frame1, frame1, Size(640, 480), 0, 0, INTER_LINEAR);
		t_resize = (getTickCount() - t_resize) / getTickFrequency();
		cout << "resize 时间： " << t_resize << endl;

		while(frame1.empty()) {

			cout << "frame1 is empty" << endl;
			//system("pause");
			return -1;
		}

		//Mat image;
		//image = imread("..\\..\\data\\lena.jpg", 1);  //当前工程的image目录下的mm.jpg文件，注意目录符号

		if (!face_cascade.load(face_cascade_name)) {
			printf("级联分类器错误，可能未找到文件，拷贝该文件到工程目录下！\n");
			system("pause");
			return -1;
		}

		double t_detec = getTickCount();
		detectAndDisplay(frame1); //调用人脸检测函数
		t_detec = (getTickCount() - t_detec) / getTickFrequency();
		cout << "detec 时间： " << t_detec << endl;

		//imwrite("frame1.jpg", frame1);


		////if(int i = system("demo.exe image frame1.jpg 100")==0);

		//SHELLEXECUTEINFO demo;
		//ZeroMemory(&demo, sizeof(SHELLEXECUTEINFO));
		//demo.cbSize = sizeof(SHELLEXECUTEINFO);
		//demo.hwnd = winhand;
		//demo.lpFile = "demo.exe";
		//demo.lpParameters = "image frame1.jpg 10";
		//demo.nShow = SW_SHOWNOACTIVATE;
		//demo.fMask = SEE_MASK_NO_CONSOLE;//SEE_MASK_NO_CONSOLE;//使用 SEE_MASK_NOCLOSEPROCESS 参数
		//demo.lpVerb = "open";
		//if (ShellExecuteEx(&demo))//执行成功
		//{
		//	//if (demo.hProcess)//指定 SEE_MASK_NOCLOSEPROCESS 并其成功执行，则 hProcess 将会返回执行成功的进程句柄
		//	//{//WaitForSingleObject(demo.hProcess, INFINITE);//等待执行完毕
		//		//waitKey(1000000000000000);
		//		//TerminateProcess(demo.hProcess, 0);
		//		//demo.hProcess = NULL;
		//		//continue;
		//	//}	
		//	double t_delay = getTickCount();
		//	double t_delay2;
		//	do {
		//		t_delay2 = getTickCount();
		//	} while ((t_delay2 - t_delay) < 1000000);

		//	//TerminateProcess(demo.hProcess, 0);
		//	//demo.hProcess = NULL;
		//	continue;

		//}
		//else
		//{
		//	String s;
		//	s.Format(("ShellExecuteEx error,error code:%d"), GetLastError());
		//	MessageBox(s);
		//}


		//cout << i << endl;

		if (cv::waitKey(1) >= 0) {

			break;
		}

		t_frame = (getTickCount() - t_frame) / getTickFrequency();
		cout << "per frame 时间： " << t_frame << "\n" << endl;

	}
	
	
	cv::waitKey(0);
	//暂停显示一下。
}

void detectAndDisplay(Mat face) {
	std::vector<Rect> faces;
	Mat face_gray;

	//cvtColor(face, face_gray, CV_BGR2GRAY);  //rgb类型转换为灰度类型
	//equalizeHist(face_gray, face_gray);   //直方图均衡化
	GaussianBlur(face, face_gray, Size(5, 5), 5, 5);

	//face_cascade.detectMultiScale(face, faces, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(1, 1));
	face_cascade.detectMultiScale(face_gray, faces, 1.4, 3, 0 | CV_HAAR_SCALE_IMAGE,Size(55, 55), Size(200, 200));

	for (int i = 0; i < faces.size(); i++) {
		//Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
		//ellipse(face, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 0), 2, 7, 0);

		cv::rectangle(face, faces[i], cv::Scalar(0, 0, 255), 2);

	}

	imshow("人脸识别", face);
}
