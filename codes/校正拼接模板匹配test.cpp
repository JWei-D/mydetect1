#include <stdio.h>
#include <opencv2/opencv.hpp>


using namespace cv;
using namespace std;

Mat cylinder(Mat imgIn, int f);
Mat linearStitch(Mat img, Mat img1, Point2i a);
Point2i getOffset(Mat img, Mat img1);

int main() {
	
	VideoCapture  cap2;
	cap2.open(2);
	VideoCapture cap1;
	cap1.open(1);
	//cout << '01' << endl;
	if (!cap1.isOpened())
	{
		cerr << "can not open a camera 1 or file." << endl;
		return -1;
	}
	if (!cap2.isOpened())
	{
		cerr << "can not open a camera 2 or file." << endl;
		return -1;
	}
	
	Mat cam1;
	//namedWindow("cam1", 1);
	Mat cam2;
	//namedWindow("cam2", 1);
	
	for (;;)
	{
		Mat orig1;
		Mat orig2;
		//cout << '1' << endl;
		cap1 >> orig1;//cap::read()
		cap2 >> orig2;
		//cout << '2' << endl;

		while (orig1.empty())
		{
			//cout <<'3'<< endl;
			cap1 >> orig1;
		}
		while (orig2.empty())
		{
			//cout <<'3'<< endl;
			cap2 >> orig2;
		}

	VideoCapture cap1, cap2;
	cap1.open("..\\..\\data\\new10.avi");
	cap2.open("..\\..\\data\\new20.avi");

	if (!cap1.isOpened()) {

		cerr << "can not open file1!" << endl;
		return -1;
	}

	if (!cap2.isOpened()) {

		cerr << "can not open file2!" << endl;
		return -1;
	}

	Mat jiaozheng1;
	Mat jiaozheng2;

	for (int i = 1; i < 15; i++)
	{
		cap1 >> jiaozheng1;//cap::read()
		cap2 >> jiaozheng2;
	}


	while (jiaozheng1.empty())
	{
		cout <<'3'<< endl;
		cap1 >> frame1;
		return -1;
	}
	while (jiaozheng2.empty())
	{
		cout <<'3'<< endl;
		cap2 >> frame2;
		return -1;
	}

		Mat map1, map2, map3, map4;
		//Mat orig1 = imread("..\\..\\data\\zuo1.jpg", 1);
		//Mat orig2 = imread("..\\..\\data\\you1.jpg", 1);

		//if (!orig1.data) {
		//	return -1;
		//}
		//if (!orig2.data) {
		//	return -1;
		//}

		Mat jiaozheng1 = Mat(orig1.size(), orig1.type());
		Mat jiaozheng2 = Mat(orig2.size(), orig2.type());

		//Mat cameramatrix1 = (Mat_<double>(3, 3) << 839.123217919623, 0, 0, 2.89934947606147, 848.415989974001, 0, 591.692291960781, 328.139423356269, 1);
		Mat cameramatrix1 = (Mat_<double>(3, 3) << 839.117168065698, 0, 0,
			4.69323500542568, 847.074745464064, 0,
			594.221472972743, 333.392295488636, 1
			);
		cameramatrix1 = cameramatrix1.t();

		Mat cameramatrix2 = (Mat_<double>(3, 3) << 839.134307239760, 0, 0,
			3.91628695549750, 851.308496526333, 0,
			653.568783645606, 318.445627752491, 1
			);
		cameramatrix2 = cameramatrix2.t();

		Mat distcoeffs1 = (Mat_<double>(4, 1) << -0.461743276094410, 0.195973401440154, 0.00161856065236508, -0.000241795103464820);

		Mat distcoeffs2 = (Mat_<double>(4, 1) << -0.456408479090023, 0.191923480695735, -0.00277267665878999, -6.12052415459897e-05);


		Size orig1size = orig1.size();
		Size orig2size = orig2.size();

		initUndistortRectifyMap(cameramatrix1, distcoeffs1, Mat(), getOptimalNewCameraMatrix(cameramatrix1, distcoeffs1, orig1size, 0, orig1size, 0), orig1size, CV_16SC2, map1, map2);
		initUndistortRectifyMap(cameramatrix2, distcoeffs2, Mat(), getOptimalNewCameraMatrix(cameramatrix2, distcoeffs2, orig2size, 0, orig1size, 0), orig2size, CV_16SC2, map3, map4);

		remap(orig1, jiaozheng1, map1, map2, INTER_LINEAR);
		remap(orig2, jiaozheng2, map3, map4, INTER_LINEAR);


		//undistort(orig1, jiaozheng1, cameramatrix1, distcoeffs);

		//namedWindow("orig1", CV_WINDOW_AUTOSIZE);
		//namedWindow("jiaozheng1", CV_WINDOW_AUTOSIZE);

		//namedWindow("orig2", CV_WINDOW_AUTOSIZE);
		//namedWindow("jiaozheng2", CV_WINDOW_AUTOSIZE);

		//imshow("orig1", orig1);
		imshow("cam1", jiaozheng1);

		//imshow("orig2", orig2);
		imshow("cam2", jiaozheng2);


		double t = (double)getTickCount();
		柱形投影
		double t3 = (double)getTickCount();
		Mat img = cylinder(jiaozheng1, 10000);
		Mat img1 = cylinder(jiaozheng2, 10000);
		t3 = ((double)getTickCount() - t3) / getTickFrequency();
		匹配
		double t1 = (double)getTickCount();
		Point2i a = getOffset(jiaozheng1, jiaozheng2);
		t1 = ((double)getTickCount() - t1) / getTickFrequency();
		拼接
		double t2 = (double)getTickCount();
		Mat stitch = linearStitch(jiaozheng1, jiaozheng2, a);
		t2 = ((double)getTickCount() - t2) / getTickFrequency();
		t = ((double)getTickCount() - t) / getTickFrequency();

		cout << "各阶段耗时：" << endl;
		cout << "模板匹配：" << t1 << '\n' << "渐入渐出拼接：" << t2 << endl;
		cout << "总时间：" << t << endl;

		imshow("柱面校正-左图像", img);
		imshow("柱面校正-右图像", img1);
		imshow("拼接结果", stitch);
		imwrite("rectify.jpg", img);
		imwrite("rectify1.jpg", img1);
		imwrite("stitch.jpg", stitch);



		bool try_use_gpu = true;
		vector<Mat> imgs;
		string result_name = "result"; // 默认输出文件名及格式
		
		imgs.push_back(jiaozheng1);
		imgs.push_back(jiaozheng2);

		Mat pano;

		Stitcher stitcher = Stitcher::createDefault(try_use_gpu);
		Stitcher::Status status = stitcher.stitch(imgs, pano);

		if (status != Stitcher::OK)
		{
			cout << "Can't stitch images, error code = " << status << endl;
			system("pause");
			return -1;
		}

		imshow(result_name, pano);
				if (waitKey(10) >= 0)
					break;

		waitKey(0);
	
	
	return 0;

	

}

/**柱面投影函数
*参数列表中imgIn为输入图像，f为焦距
*返回值为柱面投影后的图像
*/
Mat cylinder(Mat imgIn, int f)
{
	int colNum, rowNum;
	colNum = 2 * f*atan(0.5*imgIn.cols / f);//柱面图像宽
	rowNum = 0.5*imgIn.rows*f / sqrt(pow(f, 2)) + 0.5*imgIn.rows;//柱面图像高

	Mat imgOut = Mat::zeros(rowNum, colNum, CV_8UC3);
	Mat_<uchar> im1(imgIn);
	Mat_<uchar> im2(imgOut);

	//正向插值
	int x1(0), y1(0);
	for (int i = 0; i < imgIn.rows; i++)
		for (int j = 0; j < imgIn.cols; j++)
		{
			x1 = f * atan((j - 0.5*imgIn.cols) / f) + f * atan(0.5*imgIn.cols / f);
			y1 = f * (i - 0.5*imgIn.rows) / sqrt(pow(j - 0.5*imgIn.cols, 2) + pow(f, 2)) + 0.5*imgIn.rows;
			if (x1 >= 0 && x1 < colNum&&y1 >= 0 && y1<rowNum)
			{
				im2(y1, x1) = im1(i, j);
			}
		}
	return imgOut;

}

/**求平移量
*参数表为输入两幅图像（有一定重叠区域）
*返回值为点类型，存储x,y方向的偏移量
*/
Point2i getOffset(Mat img, Mat img1)
{
	vector<Mat> channels1,channels2;
	split(img, channels1);
	split(img1, channels2);
	img = channels1[0];
	img1 = channels2[0];
	Mat templ(img1, Rect(0, 0.4*img1.rows, 0.2*img1.cols, 0.2*img1.rows));
	Mat result(img.cols - templ.cols + 1, img.rows - templ.rows + 1, img.type());//result存放匹配位置信息
	matchTemplate(img, templ, result, CV_TM_CCORR_NORMED);
	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
	double minVal; double maxVal; Point minLoc; Point maxLoc; Point matchLoc;
	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
	matchLoc = maxLoc;//获得最佳匹配位置
	int dx = matchLoc.x;
	int dy = matchLoc.y - 0.4*img1.rows;//右图像相对左图像的位移
	Point2i a(dx, dy);
	return a;
}

/*渐入渐出拼接
*参数列表中，img1,img2为待拼接的两幅图像，a为偏移量
*返回值为拼接后的图像
*/
Mat linearStitch(Mat img, Mat img1, Point2i a)
{
	int d = img.cols - a.x;//过渡区宽度
	int ms = img.rows - abs(a.y);//拼接图行数
	int ns = img.cols + a.x;//拼接图列数
	Mat stitch = Mat::zeros(ms, ns, img.type());
	拼接
	Mat_<uchar> ims(stitch);
	Mat_<uchar> im(img);
	Mat_<uchar> im1(img1);

	if (a.y >= 0)
	{

		Mat roi1(stitch, Rect(0, 0, a.x, ms));
		img(Range(a.y, img.rows), Range(0, a.x)).copyTo(roi1);
		Mat roi2(stitch, Rect(img.cols, 0, a.x, ms));
		img1(Range(0, ms), Range(d, img1.cols)).copyTo(roi2);
		for (int i = 0; i < ms; i++) {
			uchar* p1 = im.ptr<uchar>(i + a.y);
			uchar* p2 = im1.ptr<uchar>(i);
			uchar* p3 = ims.ptr<uchar>(i); 
			for (int j = a.x; j < img.cols; j++) {
				ims(i, j) = uchar((img.cols - j) / float(d)*im(i + a.y, j) + (j - a.x) / float(d)*im1(i, j - a.x));
				p3[j * 3] = uchar((img.cols - j) / float(d)*p1[j * 3] + (j - a.x) / float(d)*p2[(j - a.x) * 3]);
				p3[j * 3+1] = uchar((img.cols - j) / float(d)*p1[j * 3+1] + (j - a.x) / float(d)*p2[(j - a.x) * 3+1]);
				p3[j * 3+2] = uchar((img.cols - j) / float(d)*p1[j * 3+2] + (j - a.x) / float(d)*p2[(j - a.x) * 3+2]);
			}
		}

	}
	else
	{
		Mat roi1(stitch, Rect(0, 0, a.x, ms));
		img(Range(0, ms), Range(0, a.x)).copyTo(roi1);
		Mat roi2(stitch, Rect(img.cols, 0, a.x, ms));
		img1(Range(-a.y, img.rows), Range(d, img1.cols)).copyTo(roi2);
		for (int i = 0; i < ms; i++) {
			uchar* p1 = im.ptr<uchar>(i);
			uchar* p2 = im1.ptr<uchar>(i+abs(a.y));
			uchar* p3 = ims.ptr<uchar>(i);
			for (int j = a.x; j < img.cols; j++) {

				ims(i, j) = uchar((img.cols - j) / float(d)*im(i, j) + (j - a.x) / float(d)*im1(i + abs(a.y), j - a.x));
				p3[j * 3] = uchar((img.cols - j) / float(d)*p1[j * 3] + (j - a.x) / float(d)*p2[(j - a.x) * 3]);
				p3[j * 3+1] = uchar((img.cols - j) / float(d)*p1[j * 3+1] + (j - a.x) / float(d)*p2[(j - a.x) * 3+1]);
				p3[j * 3+2] = uchar((img.cols - j) / float(d)*p1[j * 3+2] + (j - a.x) / float(d)*p2[(j - a.x) * 3+2]);
			}
				
		}
	}


	return stitch;
}


