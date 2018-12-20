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

	const double RECT_HW_RATIO = 2.5;	// 人体长宽比阈值

	const double RECT_AREA_RATIO = 0.003;	// 人体占整个图像最小比例阈值

	const double RECT_AREA_RATIO2 = 0.5;	// 人体占整体图像最大比例阈值

	BackgroundSubtractorMOG2 bgsubtractor(1,16,true);

	//bgsubtractor->setHistory(20);

	//bgsubtractor->setVarThreshold(100);

	//bgsubtractor->setDetectShadows(true);

	bool hasPeople = false;		// 是否有人

	int count = 0;	// 帧数

	int hasPeopleFrameCnt = 0; // 每K帧统计到的有人帧数

	int spaceFrames = 0;		// 每隔5帧统计一次

	const int SPACE_FRAME = 1;



	while (++count < FRAMECNT - 10) {



		double t_frame = getTickCount();

		cap >> frame;

		//resize(frame, frame, Size(frame.cols / 4, frame.rows / 4));

		double t_resize = getTickCount();

		resize(frame, frame, Size(640, 480), 0, 0, INTER_LINEAR);

		t_resize = (getTickCount() - t_resize) / getTickFrequency();

		cout << "resize 时间： " << t_resize << endl;



		double t_detec = getTickCount();

		// 背景更新

		bgsubtractor(frame, mask, 0.002);

		// 中值滤波

		medianBlur(mask, mask, 3);

		// 阈值分割，去阴影

		threshold(mask, mask, 200, 255, CV_THRESH_BINARY);

		// 找轮廓

		maskCp = mask.clone();

		findContours(maskCp, cnts, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		t_detec = (getTickCount() - t_detec) / getTickFrequency();

		cout << "detec 时间： " << t_detec << endl;



		vector<Point> maxCnt;

		for (int i = 0; i < cnts.size(); ++i) {

			maxCnt = maxCnt.size() > cnts[i].size() ? maxCnt : cnts[i];

		}

		// 画最大外接矩形

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

		//		cout << count << ":有人" << endl;

		//	}

		//	else {

		//		hasPeople = false;

		//		cout << count << ":无人" << endl;

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

		cout << "per frame 时间： " << t_frame << "\n" << endl;

	}

	return 0;

};
