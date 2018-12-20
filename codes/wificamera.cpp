#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>


using namespace std;
using namespace cv;

int main() {

	//string videoStreamAddress1 = "rtsp://admin:admin@192.168.1.105:554/cam/realmonitor?channel=1&subtype=0";

	const string videoStreamAddress1 = "http://192.168.0.129:8080/?action=stream.mjpg";
	const string videoStreamAddress2 = "http://192.168.137.22:8080/?action=stream.mjpg";

	VideoCapture cap1,cap2;
	cap1.open(videoStreamAddress1);
	//cap2.open(videoStreamAddress2);

	if (!cap1.isOpened()) {

		cout << "cap1!open" << endl;
		system("pause");
		return -1;
	}
	
	for (;;)
	{
		Mat frame1,frame2;

		cap1 >> frame1;
		//cap2 >> frame2;

		while (frame1.empty()/*||frame2.empty()*/) {

			cout << "frame1 empty" << endl;
			cap1 >> frame1;
			//cap2 >> frame2;

		}

		imshow("wificam1", frame1);
		//imshow("wificam2", frame2);

		if (waitKey(10) >= 0)
			break;

	}
	return 0;
}
