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
//string face_cascade_name = "SVM_HOG_2400PosINRIA_12000Neg_HardExample(������©�����1).xml";
//���ļ�������OpenCV��װĿ¼�µ�\sources\data\haarcascades�ڣ���Ҫ����xml�ļ����Ƶ���ǰ����Ŀ¼��
CascadeClassifier face_cascade;
void detectAndDisplay(Mat frame);
int main(int argc, char** argv) {

	Mat frame1;
	VideoCapture cap1;

	cap1.open("..\\..\\data\\new30.avi");
	//cap1.set(CV_CAP_PROP_FRAME_WIDTH, 60); //�����
	//cap1.set(CV_CAP_PROP_FRAME_HEIGHT, 40);//�����

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
		cout << "resize ʱ�䣺 " << t_resize << endl;

		while(frame1.empty()) {

			cout << "frame1 is empty" << endl;
			//system("pause");
			return -1;
		}

		//Mat image;
		//image = imread("..\\..\\data\\lena.jpg", 1);  //��ǰ���̵�imageĿ¼�µ�mm.jpg�ļ���ע��Ŀ¼����

		if (!face_cascade.load(face_cascade_name)) {
			printf("�������������󣬿���δ�ҵ��ļ����������ļ�������Ŀ¼�£�\n");
			system("pause");
			return -1;
		}

		double t_detec = getTickCount();
		detectAndDisplay(frame1); //����������⺯��
		t_detec = (getTickCount() - t_detec) / getTickFrequency();
		cout << "detec ʱ�䣺 " << t_detec << endl;

		//imwrite("frame1.jpg", frame1);


		////if(int i = system("demo.exe image frame1.jpg 100")==0);

		//SHELLEXECUTEINFO demo;
		//ZeroMemory(&demo, sizeof(SHELLEXECUTEINFO));
		//demo.cbSize = sizeof(SHELLEXECUTEINFO);
		//demo.hwnd = winhand;
		//demo.lpFile = "demo.exe";
		//demo.lpParameters = "image frame1.jpg 10";
		//demo.nShow = SW_SHOWNOACTIVATE;
		//demo.fMask = SEE_MASK_NO_CONSOLE;//SEE_MASK_NO_CONSOLE;//ʹ�� SEE_MASK_NOCLOSEPROCESS ����
		//demo.lpVerb = "open";
		//if (ShellExecuteEx(&demo))//ִ�гɹ�
		//{
		//	//if (demo.hProcess)//ָ�� SEE_MASK_NOCLOSEPROCESS ����ɹ�ִ�У��� hProcess ���᷵��ִ�гɹ��Ľ��̾��
		//	//{//WaitForSingleObject(demo.hProcess, INFINITE);//�ȴ�ִ�����
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
		cout << "per frame ʱ�䣺 " << t_frame << "\n" << endl;

	}
	
	
	cv::waitKey(0);
	//��ͣ��ʾһ�¡�
}

void detectAndDisplay(Mat face) {
	std::vector<Rect> faces;
	Mat face_gray;

	//cvtColor(face, face_gray, CV_BGR2GRAY);  //rgb����ת��Ϊ�Ҷ�����
	//equalizeHist(face_gray, face_gray);   //ֱ��ͼ���⻯
	GaussianBlur(face, face_gray, Size(5, 5), 5, 5);

	//face_cascade.detectMultiScale(face, faces, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(1, 1));
	face_cascade.detectMultiScale(face_gray, faces, 1.4, 3, 0 | CV_HAAR_SCALE_IMAGE,Size(55, 55), Size(200, 200));

	for (int i = 0; i < faces.size(); i++) {
		//Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
		//ellipse(face, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 0), 2, 7, 0);

		cv::rectangle(face, faces[i], cv::Scalar(0, 0, 255), 2);

	}

	imshow("����ʶ��", face);
}
