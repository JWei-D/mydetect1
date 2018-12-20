#include "network.h"
#include "mtcnn.h"
#include <time.h>
#include <opencv2/contrib/contrib.hpp>

#include "opencv2/core/core.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
string face_cascade_name = "E:\\jwei\\OpenCV2413\\opencv\\sources\\data\\hogcascades\\hogcascade_pedestrians.xml";
//string face_cascade_name = "E:\\jwei\\OpenCV2413\\opencv\\sources\\data\\lbpcascades\\lbpcascade_frontalcatface.xml";
CascadeClassifier face_cascade;
void detectAndDisplay(Mat frame);


int main()
{
	//initial models without image's width or height
	mtcnn find;

	//test_type choice
	string test_type = "img_dir";
	if (test_type == "img_dir")
	{
		Directory dir;
        string imgpath = "/home/jwei/mydetec1/data/";
        vector<string> filenames = dir.GetListFiles(imgpath, ".jpg", false);
        cout<<filenames.size()<<endl;
		for (int i = 0; i < filenames.size(); i++)
		{
			string imgname = imgpath + filenames[i];
			Mat image = imread(imgname, 1);

			clock_t start_t = clock();
			//detect face by min_size(30)
			find.findFace(image, 30);
			cout << "Cost time: " << clock() - start_t << endl;

			imshow("test", image);
			waitKey(0);
		}
	}
	else if(test_type == "video")
	{
		VideoCapture cap("E:\\jwei\\code\\data\\huanghh\\qian.avi");
		if (!cap.isOpened())
			cout << "fail to open!" << endl;
		Mat image;

		while (true) {
			
			cap >> image;
			if (!image.data) {
				cout << "fail to read image!" << endl;
				return -1;
			}



			clock_t start_t = clock();

			if (!face_cascade.load(face_cascade_name)) {
				printf("�������������󣬿���δ�ҵ��ļ����������ļ�������Ŀ¼�£�\n");
				system("pause");
			}

			double t_detec = getTickCount();
			detectAndDisplay(image); //����������⺯��
			t_detec = (getTickCount() - t_detec) / getTickFrequency();
			cout << "���� ʱ�䣺 " << t_detec << endl;


			//detect face by min_size(60)
			find.findFace(image, 50);
			cout << "Cost time: " << clock() - start_t <<"\n"<< endl;

			imshow("result", image);
			if (waitKey(1) >= 0) break;
		}
	}
	else if (test_type == "camera") {


	}
	
	else
	{
		cout << "Unknow test type!" << endl;
	}

    //system("pause");
	return 0;
}

void detectAndDisplay(Mat face) {
	std::vector<Rect> faces;
	Mat face_gray;

	cvtColor(face, face_gray, CV_BGR2GRAY);  //rgb����ת��Ϊ�Ҷ�����
	//equalizeHist(face_gray, face_gray);   //ֱ��ͼ���⻯
	//GaussianBlur(face_gray, face_gray, Size(5, 5), 5, 5);

	//face_cascade.detectMultiScale(face, faces, 1.1, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(1, 1));
	face_cascade.detectMultiScale(face_gray, faces, 1.9, 8, 0 | CV_HAAR_SCALE_IMAGE, Size(10, 10), Size(120, 120));//80,200
	//face_cascade.detectMultiScale(face, faces, 1.9, 3, 0 | CV_HAAR_SCALE_IMAGE, Size(1, 1), Size(120, 120));//80,200

	for (int i = 0; i < faces.size(); i++) {
		//Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
		//ellipse(face, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 0), 2, 7, 0);

		cv::rectangle(face, faces[i], cv::Scalar(0, 0, 255), 2,8,0);

	}

	//imshow("����ʶ��", face);
}
