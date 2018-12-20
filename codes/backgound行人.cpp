#include<stdio.h>

#include<opencv2/opencv.hpp>

using namespace std;

using namespace cv;

int main() {

	VideoCapture cap("..\\..\\data\\anew3000000.avi");

	if (!cap.isOpened()) {

		cout << "video not exist!" << endl;

		return -1;

	}

	long FRAMECNT = cap.get(CV_CAP_PROP_FRAME_COUNT);

	//cout << FRAMECNT << endl;

	Mat frame, mask, maskCp;

	vector<vector<Point>> cnts;

	Rect maxRect;

	const double RECT_HW_RATIO = 2.5;	// ���峤�����ֵ

	const double RECT_AREA_RATIO = 0.003;	// ����ռ����ͼ����С������ֵ

	const double RECT_AREA_RATIO2 = 0.5;	// ����ռ����ͼ����������ֵ

	BackgroundSubtractorMOG2 bgsubtractor(1,16,true);

	//bgsubtractor->setHistory(20);

	//bgsubtractor->setVarThreshold(100);

	//bgsubtractor->setDetectShadows(true);

	bool hasPeople = false;		// �Ƿ�����

	int count = 0;	// ֡��

	int hasPeopleFrameCnt = 0; // ÿK֡ͳ�Ƶ�������֡��

	int spaceFrames = 0;		// ÿ��5֡ͳ��һ��

	const int SPACE_FRAME = 1;



	while (++count < FRAMECNT - 10) {



		double t_frame = getTickCount();

		cap >> frame;

		//resize(frame, frame, Size(frame.cols / 4, frame.rows / 4));

		double t_resize = getTickCount();

		resize(frame, frame, Size(640, 480), 0, 0, INTER_LINEAR);

		t_resize = (getTickCount() - t_resize) / getTickFrequency();

		cout << "resize ʱ�䣺 " << t_resize << endl;



		double t_detec = getTickCount();

		// ��������

		bgsubtractor(frame, mask, 0.002);

		// ��ֵ�˲�

		medianBlur(mask, mask, 3);

		// ��ֵ�ָȥ��Ӱ

		threshold(mask, mask, 200, 255, CV_THRESH_BINARY);

		// ������

		maskCp = mask.clone();

		findContours(maskCp, cnts, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		t_detec = (getTickCount() - t_detec) / getTickFrequency();

		cout << "detec ʱ�䣺 " << t_detec << endl;



		vector<Point> maxCnt;

		for (int i = 0; i < cnts.size(); ++i) {

			maxCnt = maxCnt.size() > cnts[i].size() ? maxCnt : cnts[i];

		}

		// �������Ӿ���

		if (maxCnt.size() > 0) {

			maxRect = boundingRect(maxCnt);

			double rectAreaRatio = (double)maxRect.area() / (frame.cols * frame.rows);

			if ((double)maxRect.height / maxRect.width > RECT_HW_RATIO && rectAreaRatio > RECT_AREA_RATIO &&

				rectAreaRatio < RECT_AREA_RATIO2) {

				rectangle(frame, maxRect.tl(), maxRect.br(), Scalar(0, 0, 255), 2);

				++hasPeopleFrameCnt;

			}

		}

		//++spaceFrames;

		//if (spaceFrames >= SPACE_FRAME) {

		//	if (hasPeopleFrameCnt > SPACE_FRAME / 8) {

		//		hasPeople = true;

		//		cout << count << ":����" << endl;

		//	}

		//	else {

		//		hasPeople = false;

		//		cout << count << ":����" << endl;

		//	}

		//	hasPeopleFrameCnt = 0;

		//	spaceFrames = 0;

		//}



		imshow("frame", frame);

		imshow("mask", mask);

		if (waitKey(10) == 27) {

			break;

		}

		t_frame = (getTickCount() - t_frame) / getTickFrequency();

		cout << "per frame ʱ�䣺 " << t_frame << "\n" << endl;

	}

	return 0;

};
