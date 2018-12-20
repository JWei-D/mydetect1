////eroding and dilating
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "highgui.h"
//#include <stdlib.h>
//#include <stdio.h>
//
//using namespace cv;
//
///// 全局变量
//Mat src, erosion_dst, dilation_dst;
//
//int erosion_elem = 0;
//int erosion_size = 0;
//int dilation_elem = 0;
//int dilation_size = 0;
//int const max_elem = 2;
//int const max_kernel_size = 21;
//
///** Function Headers */
//void Erosion(int, void*);
//void Dilation(int, void*);
//
///** @function main */
//int main(int argc, char** argv)
//{
//	/// Load 图像
//	src = imread("E:\\jwei\\code\\data\\lena.jpg");
//
//	if (!src.data)
//	{
//		return -1;
//	}
//
//	/// 创建显示窗口
//	namedWindow("Erosion Demo", CV_WINDOW_AUTOSIZE);
//	namedWindow("Dilation Demo", CV_WINDOW_AUTOSIZE);
//	cvMoveWindow("Dilation Demo", src.cols, 0);
//
//	/// 创建腐蚀 Trackbar
//	createTrackbar("Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Erosion Demo",
//		&erosion_elem, max_elem,
//		Erosion);
//
//	createTrackbar("Kernel size:\n 2n +1", "Erosion Demo",
//		&erosion_size, max_kernel_size,
//		Erosion);
//
//	/// 创建膨胀 Trackbar
//	createTrackbar("Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Dilation Demo",
//		&dilation_elem, max_elem,
//		Dilation);
//
//	createTrackbar("Kernel size:\n 2n +1", "Dilation Demo",
//		&dilation_size, max_kernel_size,
//		Dilation);
//
//	/// Default start
//	Erosion(0, 0);
//	Dilation(0, 0);
//
//	waitKey(0);
//	return 0;
//}
//
///**  @function Erosion  */
//void Erosion(int, void*)
//{
//	int erosion_type;
//	if (erosion_elem == 0) { erosion_type = MORPH_RECT; }
//	else if (erosion_elem == 1) { erosion_type = MORPH_CROSS; }
//	else if (erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }
//
//	Mat element = getStructuringElement(erosion_type,
//		Size(2 * erosion_size + 1, 2 * erosion_size + 1),
//		Point(erosion_size, erosion_size));
//
//	/// 腐蚀操作
//	erode(src, erosion_dst, element);
//	imshow("Erosion Demo", erosion_dst);
//}
//
///** @function Dilation */
//void Dilation(int, void*)
//{
//	int dilation_type;
//	if (dilation_elem == 0) { dilation_type = MORPH_RECT; }
//	else if (dilation_elem == 1) { dilation_type = MORPH_CROSS; }
//	else if (dilation_elem == 2) { dilation_type = MORPH_ELLIPSE; }
//
//	Mat element = getStructuringElement(dilation_type,
//		Size(2 * dilation_size + 1, 2 * dilation_size + 1),
//		Point(dilation_size, dilation_size));
//	///膨胀操作
//	dilate(src, dilation_dst, element);
//	imshow("Dilation Demo", dilation_dst);
//}



////fuliyebianhuan
//#include "opencv2/core/core.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include <iostream>
//
//using namespace cv;
//
//int main(int argc, char ** argv)
//{
//	const char* filename = argc >= 2 ? argv[1] : "lena.jpg";
//
//	Mat I = imread("E:\\jwei\\code\\data\\lena.jpg", CV_LOAD_IMAGE_GRAYSCALE);
//	if (I.empty())
//		return -1;
//
//	Mat padded;                            //expand input image to optimal size
//	int m = getOptimalDFTSize(I.rows);
//	int n = getOptimalDFTSize(I.cols); // on the border add zero values
//	copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));
//
//	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
//	Mat complexI;
//	merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
//
//	dft(complexI, complexI);            // this way the result may fit in the source matrix
//
//										// compute the magnitude and switch to logarithmic scale
//										// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
//	split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
//	magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude  
//	Mat magI = planes[0];
//
//	magI += Scalar::all(1);                    // switch to logarithmic scale
//	log(magI, magI);
//
//	// crop the spectrum, if it has an odd number of rows or columns
//	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
//
//	// rearrange the quadrants of Fourier image  so that the origin is at the image center        
//	int cx = magI.cols / 2;
//	int cy = magI.rows / 2;
//
//	Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant 
//	Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
//	Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
//	Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right
//
//	Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
//	q0.copyTo(tmp);
//	q3.copyTo(q0);
//	tmp.copyTo(q3);
//
//	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
//	q2.copyTo(q1);
//	tmp.copyTo(q2);
//
//	normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a 
//											// viewable image form (float between values 0 and 1).
//
//	imshow("Input Image", I);    // Show the result
//	imshow("spectrum magnitude", magI);
//	waitKey();
//
//	return 0;
//}

//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include <stdlib.h>
//#include <stdio.h>
//
//using namespace cv;
//
///// 全局变量
//Mat src, dst;
//
//int morph_elem = 0;
//int morph_size = 0;
//int morph_operator = 0;
//int const max_operator = 4;
//int const max_elem = 2;
//int const max_kernel_size = 21;
//
//const char* window_name = "Morphology Transformations Demo";
//
///** 回调函数申明 */
//void Morphology_Operations(int, void*);
//
///** @函数 main */
//int main(int argc, char** argv)
//{
//	/// 装载图像
//	src = imread("E:\\jwei\\code\\data\\lena.jpg");
//
//	if (!src.data)
//	{
//		return -1;
//	}
//
//	/// 创建显示窗口
//	namedWindow(window_name, CV_WINDOW_AUTOSIZE);
//
//	/// 创建选择具体操作的 trackbar
//	createTrackbar("Operator:\n 0: Opening - 1: Closing \n 2: Gradient - 3: Top Hat \n 4: Black Hat", window_name, &morph_operator, max_operator, Morphology_Operations);
//
//	/// 创建选择内核形状的 trackbar
//	createTrackbar("Element:\n 0: Rect - 1: Cross - 2: Ellipse", window_name,
//		&morph_elem, max_elem,
//		Morphology_Operations);
//
//	/// 创建选择内核大小的 trackbar
//	createTrackbar("Kernel size:\n 2n +1", window_name,
//		&morph_size, max_kernel_size,
//		Morphology_Operations);
//
//	/// 启动使用默认值
//	Morphology_Operations(0, 0);
//
//	waitKey(0);
//	return 0;
//}
//
///**
//* @函数 Morphology_Operations
//*/
//void Morphology_Operations(int, void*)
//{
//	// 由于 MORPH_X的取值范围是: 2,3,4,5 和 6
//	int operation = morph_operator + 2;
//
//	Mat element = getStructuringElement(morph_elem, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
//
//	/// 运行指定形态学操作
//	morphologyEx(src, dst, operation, element);
//	imshow(window_name, dst);
//}


////sobel and scharr
//
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include <stdlib.h>
//#include <stdio.h>
//
//using namespace cv;
//
///** @function main */
//int main(int argc, char** argv)
//{
//
//	Mat src, src_gray;
//	Mat grad;
//	const char* window_name = "Sobel Demo - Simple Edge Detector";
//	int scale = 1;
//	int delta = 0;
//	int ddepth = CV_16S;
//
//	int c;
//
//	/// 装载图像
//	src = imread("E:\\jwei\\code\\data\\lena.jpg");
//
//	if (!src.data)
//	{
//		return -1;
//	}
//
//	GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);
//
//	/// 转换为灰度图
//	cvtColor(src, src_gray, CV_RGB2GRAY);
//
//	/// 创建显示窗口
//	namedWindow(window_name, CV_WINDOW_AUTOSIZE);
//
//	/// 创建 grad_x 和 grad_y 矩阵
//	Mat grad_x, grad_y;
//	Mat abs_grad_x, abs_grad_y;
//
//	/// 求 X方向梯度
//	Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
//	//Sobel(src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
//	convertScaleAbs(grad_x, abs_grad_x);
//
//	/// 求Y方向梯度
//	Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
//	//Sobel(src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
//	convertScaleAbs(grad_y, abs_grad_y);
//
//	/// 合并梯度(近似)
//	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
//
//	imshow(window_name, grad);
//
//	waitKey(0);
//
//	return 0;
//}

////laplace
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include <stdlib.h>
//#include <stdio.h>
//
//using namespace cv;
//
///** @函数 main */
//int main(int argc, char** argv)
//{
//	Mat src, src_gray, dst;
//	int kernel_size = 3;
//	int scale = 1;
//	int delta = 0;
//	int ddepth = CV_16S;
//	const char* window_name = "Laplace Demo";
//
//	int c;
//
//	/// 装载图像
//	src = imread("E:\\jwei\\code\\data\\lena.jpg");
//
//	if (!src.data)
//	{
//		return -1;
//	}
//
//	/// 使用高斯滤波消除噪声
//	GaussianBlur(src, src, Size(3, 3), 0, 0, BORDER_DEFAULT);
//
//	/// 转换为灰度图
//	cvtColor(src, src_gray, CV_RGB2GRAY);
//
//	/// 创建显示窗口
//	namedWindow(window_name, CV_WINDOW_AUTOSIZE);
//
//	/// 使用Laplace函数
//	Mat abs_dst;
//
//	Laplacian(src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT);
//	convertScaleAbs(dst, abs_dst);
//
//	/// 显示结果
//	imshow(window_name, abs_dst);
//
//	waitKey(0);
//
//	return 0;
//}


////canny
//
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include <stdlib.h>
//#include <stdio.h>
//
//using namespace cv;
//
///// 全局变量
//
//Mat src, src_gray;
//Mat dst, detected_edges;
//
//int edgeThresh = 1;
//int lowThreshold;
//int const max_lowThreshold = 100;
//int ratio = 3;
//int kernel_size = 3;
//const char* window_name = "Edge Map";
//
///**
//* @函数 CannyThreshold
//* @简介： trackbar 交互回调 - Canny阈值输入比例1:3
//*/
//void CannyThreshold(int, void*)
//{
//	/// 使用 3x3内核降噪
//	blur(src_gray, detected_edges, Size(3, 3));
//
//	/// 运行Canny算子
//	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);
//
//	/// 使用 Canny算子输出边缘作为掩码显示原图像
//	dst = Scalar::all(0);
//
//	src.copyTo(dst, detected_edges);
//	imshow(window_name, dst);
//}
//
//
///** @函数 main */
//int main(int argc, char** argv)
//{
//	/// 装载图像
//	src = imread("E:\\jwei\\code\\data\\lena.jpg");
//
//	if (!src.data)
//	{
//		return -1;
//	}
//
//	/// 创建与src同类型和大小的矩阵(dst)
//	dst.create(src.size(), src.type());
//
//	/// 原图像转换为灰度图像
//	cvtColor(src, src_gray, CV_BGR2GRAY);
//
//	/// 创建显示窗口
//	namedWindow(window_name, CV_WINDOW_AUTOSIZE);
//
//	/// 创建trackbar
//	createTrackbar("Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold);
//
//	/// 显示图像
//	CannyThreshold(0, 0);
//
//	/// 等待用户反应
//	waitKey(0);
//
//	return 0;
//}

////houghlines
//
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//
//#include <iostream>
//
//using namespace cv;
//using namespace std;
//
//void help()
//{
//	cout << "\nThis program demonstrates line finding with the Hough transform.\n"
//		"Usage:\n"
//		"./houghlines <image_name>, Default is pic1.jpg\n" << endl;
//}
//
//int main(int argc, char** argv)
//{
//	const char* filename = argc >= 2 ? argv[1] : "..\\..\\data\\lena.jpg";
//
//	Mat src = imread(filename, 0);
//	if (src.empty())
//	{
//		help();
//		cout << "can not open " << filename << endl;
//		return -1;
//	}
//
//	Mat dst, cdst;
//	Canny(src, dst, 50, 200, 3);
//	cvtColor(dst, cdst, CV_GRAY2BGR);
//
//#if 0
//	vector<Vec2f> lines;
//	HoughLines(dst, lines, 1, CV_PI / 180, 100, 0, 0);
//
//	for (size_t i = 0; i < lines.size(); i++)
//	{
//		float rho = lines[i][0], theta = lines[i][1];
//		Point pt1, pt2;
//		double a = cos(theta), b = sin(theta);
//		double x0 = a * rho, y0 = b * rho;
//		pt1.x = cvRound(x0 + 1000 * (-b));
//		pt1.y = cvRound(y0 + 1000 * (a));
//		pt2.x = cvRound(x0 - 1000 * (-b));
//		pt2.y = cvRound(y0 - 1000 * (a));
//		line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, CV_AA);
//	}
//#else
//	vector<Vec4i> lines;
//	HoughLinesP(dst, lines, 1, CV_PI / 180, 50, 50, 10);
//	for (size_t i = 0; i < lines.size(); i++)
//	{
//		Vec4i l = lines[i];
//		line(cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, CV_AA);
//	}
//#endif
//	imshow("source", src);
//	imshow("detected lines", cdst);
//
//	waitKey();
//
//	return 0;
//}


////houghcircle
//
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include <iostream>
//#include <stdio.h>
//
//using namespace cv;
//
///** @function main */
//int main(int argc, char** argv)
//{
//	Mat src, src_gray;
//
//	/// Read the image
//	src = imread("..\\..\\data\\lena.jpg", 1);
//
//	if (!src.data)
//	{
//		return -1;
//	}
//
//	/// Convert it to gray
//	cvtColor(src, src_gray, CV_BGR2GRAY);
//
//	/// Reduce the noise so we avoid false circle detection
//	GaussianBlur(src_gray, src_gray, Size(9, 9), 2, 2);
//
//	vector<Vec3f> circles;
//
//	/// Apply the Hough Transform to find the circles
//	HoughCircles(src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows / 8, 200, 100, 0, 0);
//
//	/// Draw the circles detected
//	for (size_t i = 0; i < circles.size(); i++)
//	{
//		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
//		int radius = cvRound(circles[i][2]);
//		// circle center
//		circle(src, center, 3, Scalar(0, 255, 0), -1, 8, 0);
//		// circle outline
//		circle(src, center, radius, Scalar(0, 0, 255), 3, 8, 0);
//	}
//
//	/// Show your results
//	namedWindow("Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE);
//	imshow("Hough Circle Transform Demo", src);
//
//	waitKey(0);
//	return 0;
//}


////remap
//
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include <iostream>
//#include <stdio.h>
//
//using namespace cv;
//
///// Global variables
//Mat src, dst;
//Mat map_x, map_y;
//const char* remap_window = "Remap demo";
//int ind = 0;
//
///// Function Headers
//void update_map(void);
//
///**
//* @function main
//*/
//int main(int argc, char** argv)
//{
//	/// Load the image
//	src = imread("..\\..\\data\\lena.jpg", 1);
//
//	/// Create dst, map_x and map_y with the same size as src:
//	dst.create(src.size(), src.type());
//	map_x.create(src.size(), CV_32FC1);
//	map_y.create(src.size(), CV_32FC1);
//
//	/// Create window
//	namedWindow(remap_window, CV_WINDOW_AUTOSIZE);
//
//	/// Loop
//	while (true)
//	{
//		/// Each 1 sec. Press ESC to exit the program
//		int c = waitKey(1000);
//
//		if ((char)c == 27)
//		{
//			break;
//		}
//
//		/// Update map_x & map_y. Then apply remap
//		update_map();
//		remap(src, dst, map_x, map_y, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
//
//		/// Display results
//		imshow(remap_window, dst);
//	}
//	return 0;
//}
//
///**
//* @function update_map
//* @brief Fill the map_x and map_y matrices with 4 types of mappings
//*/
//void update_map(void)
//{
//	ind = ind % 4;
//
//	for (int j = 0; j < src.rows; j++)
//	{
//		for (int i = 0; i < src.cols; i++)
//		{
//			switch (ind)
//			{
//			case 0:
//				if (i > src.cols*0.25 && i < src.cols*0.75 && j > src.rows*0.25 && j < src.rows*0.75)
//				{
//					map_x.at<float>(j, i) = 2 * (i - src.cols*0.25) + 0.5;
//					map_y.at<float>(j, i) = 2 * (j - src.rows*0.25) + 0.5;
//				}
//				else
//				{
//					map_x.at<float>(j, i) = 0;
//					map_y.at<float>(j, i) = 0;
//				}
//				break;
//			case 1:
//				map_x.at<float>(j, i) = i;
//				map_y.at<float>(j, i) = src.rows - j;
//				break;
//			case 2:
//				map_x.at<float>(j, i) = src.cols - i;
//				map_y.at<float>(j, i) = j;
//				break;
//			case 3:
//				map_x.at<float>(j, i) = src.cols - i;
//				map_y.at<float>(j, i) = src.rows - j;
//				break;
//			} // end of switch
//		}
//	}
//	ind++;
//}

////fangshebianhuan
//
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include <iostream>
//#include <stdio.h>
//
//using namespace cv;
//using namespace std;
//
///// 全局变量
//const char* source_window = "Source image";
//const char* warp_window = "Warp";
//const char* warp_rotate_window = "Warp + Rotate";
//
///** @function main */
//int main(int argc, char** argv)
//{
//	Point2f srcTri[3];
//	Point2f dstTri[3];
//
//	Mat rot_mat(2, 3, CV_32FC1);
//	Mat warp_mat(2, 3, CV_32FC1);
//	Mat src, warp_dst, warp_rotate_dst;
//
//	/// 加载源图像
//	src = imread("..\\..\\data\\lena.jpg", 1);
//
//	/// 设置目标图像的大小和类型与源图像一致
//	warp_dst = Mat::zeros(src.rows, src.cols, src.type());
//
//	/// 设置源图像和目标图像上的三组点以计算仿射变换
//	srcTri[0] = Point2f(0, 0);
//	srcTri[1] = Point2f(src.cols - 1, 0);
//	srcTri[2] = Point2f(0, src.rows - 1);
//
//	dstTri[0] = Point2f(src.cols*0.0, src.rows*0.33);
//	dstTri[1] = Point2f(src.cols*0.85, src.rows*0.25);
//	dstTri[2] = Point2f(src.cols*0.15, src.rows*0.7);
//
//	/// 求得仿射变换
//	warp_mat = getAffineTransform(srcTri, dstTri);
//
//	/// 对源图像应用上面求得的仿射变换
//	warpAffine(src, warp_dst, warp_mat, warp_dst.size());
//
//	/** 对图像扭曲后再旋转 */
//
//	/// 计算绕图像中点顺时针旋转50度缩放因子为0.6的旋转矩阵
//	Point center = Point(warp_dst.cols / 2, warp_dst.rows / 2);
//	double angle = -50.0;
//	double scale = 0.6;
//
//	/// 通过上面的旋转细节信息求得旋转矩阵
//	rot_mat = getRotationMatrix2D(center, angle, scale);
//
//	/// 旋转已扭曲图像
//	warpAffine(warp_dst, warp_rotate_dst, rot_mat, warp_dst.size());
//
//	/// 显示结果
//	namedWindow(source_window, CV_WINDOW_AUTOSIZE);
//	imshow(source_window, src);
//
//	namedWindow(warp_window, CV_WINDOW_AUTOSIZE);
//	imshow(warp_window, warp_dst);
//
//	namedWindow(warp_rotate_window, CV_WINDOW_AUTOSIZE);
//	imshow(warp_rotate_window, warp_rotate_dst);
//
//	/// 等待用户按任意按键退出程序
//	waitKey(0);
//
//	return 0;
//}

////zhifangtujunhenghua
//
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include <iostream>
//#include <stdio.h>
//
//using namespace cv;
//using namespace std;
//
///**  @function main */
//int main(int argc, char** argv)
//{
//	Mat src, dst;
//
//	const char* source_window = "Source image";
//	const char* equalized_window = "Equalized Image";
//
//	/// 加载源图像
//	src = imread("..\\..\\data\\lena.jpg", 1);
//
//	if (!src.data)
//	{
//		cout << "Usage: ./Histogram_Demo <path_to_image>" << endl;
//		return -1;
//	}
//
//	/// 转为灰度图
//	cvtColor(src, src, CV_BGR2GRAY);
//
//	/// 应用直方图均衡化
//	equalizeHist(src, dst);
//
//	/// 显示结果
//	namedWindow(source_window, CV_WINDOW_AUTOSIZE);
//	namedWindow(equalized_window, CV_WINDOW_AUTOSIZE);
//
//	imshow(source_window, src);
//	imshow(equalized_window, dst);
//
//	/// 等待用户按键退出程序
//	waitKey(0);
//
//	return 0;
//}

////zhifangtujisuan
//
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include <iostream>
//#include <stdio.h>
//
//using namespace std;
//using namespace cv;
//
///** @函数 main */
//int main(int argc, char** argv)
//{
//	Mat src, dst;
//
//	/// 装载图像
//	src = imread("..\\..\\data\\lena.jpg", 1);
//
//	if (!src.data)
//	{
//		return -1;
//	}
//
//	/// 分割成3个单通道图像 ( R, G 和 B )
//	vector<Mat> rgb_planes;
//	split(src, rgb_planes);
//
//	/// 设定bin数目
//	int histSize = 255;
//
//	/// 设定取值范围 ( R,G,B) )
//	float range[] = { 0, 255 };
//	const float* histRange = { range };
//
//	bool uniform = true; bool accumulate = false;
//
//	Mat r_hist, g_hist, b_hist;
//
//	/// 计算直方图:
//	calcHist(&rgb_planes[0], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);
//	calcHist(&rgb_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
//	calcHist(&rgb_planes[2], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
//
//	// 创建直方图画布
//	int hist_w = 400; int hist_h = 400;
//	int bin_w = cvRound((double)hist_w / histSize);
//
//	Mat histImage(hist_w, hist_h, CV_8UC3, Scalar(0, 0, 0));
//
//	/// 将直方图归一化到范围 [ 0, histImage.rows ]
//	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
//	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
//	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
//
//	/// 在直方图画布上画出直方图
//	for (int i = 1; i < histSize; i++)
//	{
//		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
//			Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
//			Scalar(0, 0, 255), 2, 8, 0);
//		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
//			Point(bin_w*(i), hist_h - cvRound(g_hist.at<float>(i))),
//			Scalar(0, 255, 0), 2, 8, 0);
//		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
//			Point(bin_w*(i), hist_h - cvRound(b_hist.at<float>(i))),
//			Scalar(255, 0, 0), 2, 8, 0);
//	}
//
//	/// 显示直方图
//	namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE);
//	imshow("calcHist Demo", histImage);
//
//	waitKey(0);
//
//	return 0;
//
//}

////fanxiangtouying
//
//#include "opencv2/imgproc/imgproc.hpp"
//#include "opencv2/highgui/highgui.hpp"
//
//#include <iostream>
//
//using namespace cv;
//using namespace std;
//
///// 全局变量
//Mat src; Mat hsv; Mat hue;
//int bins = 25;
//
///// 函数申明
//void Hist_and_Backproj(int, void*);
//
///** @函数 main */
//int main(int argc, char** argv)
//{
//	/// 读取图像
//	src = imread("..\\..\\data\\lena.jpg", 1);
//	/// 转换到 HSV 空间
//	cvtColor(src, hsv, CV_BGR2HSV);
//
//	/// 分离 Hue 通道
//	hue.create(hsv.size(), hsv.depth());
//	int ch[] = { 0, 0 };
//	mixChannels(&hsv, 1, &hue, 1, ch, 1);
//
//	/// 创建 Trackbar 来输入bin的数目
//	const char* window_image = "Source image";
//	namedWindow(window_image, CV_WINDOW_AUTOSIZE);
//	createTrackbar("* Hue  bins: ", window_image, &bins, 180, Hist_and_Backproj);
//	Hist_and_Backproj(0, 0);
//
//	/// 现实图像
//	imshow(window_image, src);
//
//	/// 等待用户反应
//	waitKey(0);
//	return 0;
//}
//
//
///**
//* @函数 Hist_and_Backproj
//* @简介：Trackbar事件的回调函数
//*/
//void Hist_and_Backproj(int, void*)
//{
//	MatND hist;
//	int histSize = MAX(bins, 2);
//	float hue_range[] = { 0, 180 };
//	const float* ranges = { hue_range };
//
//	/// 计算直方图并归一化
//	calcHist(&hue, 1, 0, Mat(), hist, 1, &histSize, &ranges, true, false);
//	normalize(hist, hist, 0, 255, NORM_MINMAX, -1, Mat());
//
//	/// 计算反向投影
//	MatND backproj;
//	calcBackProject(&hue, 1, 0, hist, backproj, &ranges, 1, true);
//
//	/// 显示反向投影
//	imshow("BackProj", backproj);
//
//	/// 显示直方图
//	int w = 400; int h = 400;
//	int bin_w = cvRound((double)w / histSize);
//	Mat histImg = Mat::zeros(w, h, CV_8UC3);
//
//	for (int i = 0; i < bins; i++)
//	{
//		rectangle(histImg, Point(i*bin_w, h), Point((i + 1)*bin_w, h - cvRound(hist.at<float>(i)*h / 255.0)), Scalar(0, 0, 255), -1);
//	}
//
//	imshow("Histogram", histImg);
//}

////match
//
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include <iostream>
//#include <stdio.h>
//
//using namespace std;
//using namespace cv;
//
///// 全局变量
//Mat img; Mat templ; Mat result;
//const char* image_window = "Source Image";
//const char* result_window = "Result window";
//
//int match_method;
//int max_Trackbar = 5;
//
///// 函数声明
//void MatchingMethod(int, void*);
//
///** @主函数 */
//int main(int argc, char** argv)
//{
//	/// 载入原图像和模板块
//	img = imread("..\\..\\data\\lena.jpg", 1);
//	templ = imread("..\\..\\data\\lenapart.jpg", 1);
//
//	/// 创建窗口
//	namedWindow(image_window, CV_WINDOW_AUTOSIZE);
//	namedWindow(result_window, CV_WINDOW_AUTOSIZE);
//
//	/// 创建滑动条
//	const char* trackbar_label = "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED";
//	createTrackbar(trackbar_label, image_window, &match_method, max_Trackbar, MatchingMethod);
//
//	MatchingMethod(0, 0);
//
//	waitKey(0);
//	return 0;
//}
//
///**
//* @函数 MatchingMethod
//* @简单的滑动条回调函数
//*/
//void MatchingMethod(int, void*)
//{
//	/// 将被显示的原图像
//	Mat img_display;
//	img.copyTo(img_display);
//
//	/// 创建输出结果的矩阵
//	int result_cols = img.cols - templ.cols + 1;
//	int result_rows = img.rows - templ.rows + 1;
//
//	result.create(result_cols, result_rows, CV_32FC1);
//
//	/// 进行匹配和标准化
//	matchTemplate(img, templ, result, match_method);
//	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
//
//	/// 通过函数 minMaxLoc 定位最匹配的位置
//	double minVal; double maxVal; Point minLoc; Point maxLoc;
//	Point matchLoc;
//
//	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
//
//	/// 对于方法 SQDIFF 和 SQDIFF_NORMED, 越小的数值代表更高的匹配结果. 而对于其他方法, 数值越大匹配越好
//	if (match_method == CV_TM_SQDIFF || match_method == CV_TM_SQDIFF_NORMED)
//	{
//		matchLoc = minLoc;
//	}
//	else
//	{
//		matchLoc = maxLoc;
//	}
//
//	/// 让我看看您的最终结果
//	rectangle(img_display, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);
//	rectangle(result, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar::all(0), 2, 8, 0);
//
//	imshow(image_window, img_display);
//	imshow(result_window, result);
//
//	return;
//}

////contours
//
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include <iostream>
//#include <stdio.h>
//#include <stdlib.h>
//
//using namespace cv;
//using namespace std;
//
//Mat src; Mat src_gray;
//int thresh = 100;
//int max_thresh = 255;
//RNG rng(12345);
//
///// Function header
//void thresh_callback(int, void*);
//
///** @function main */
//int main(int argc, char** argv)
//{
//	/// 加载源图像
//	src = imread("..\\..\\data\\lena.jpg", 1);
//
//	/// 转成灰度并模糊化降噪
//	cvtColor(src, src_gray, CV_BGR2GRAY);
//	blur(src_gray, src_gray, Size(3, 3));
//
//	/// 创建窗体
//	const char* source_window = "Source";
//	namedWindow(source_window, CV_WINDOW_AUTOSIZE);
//	imshow(source_window, src);
//
//	createTrackbar(" Canny thresh:", "Source", &thresh, max_thresh, thresh_callback);
//	thresh_callback(0, 0);
//
//	waitKey(0);
//	return(0);
//}
//
///** @function thresh_callback */
//void thresh_callback(int, void*)
//{
//	Mat canny_output;
//	vector<vector<Point> > contours;
//	vector<Vec4i> hierarchy;
//
//	/// 用Canny算子检测边缘
//	Canny(src_gray, canny_output, thresh, thresh * 2, 3);
//	/// 寻找轮廓
//	findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
//
//	/// 绘出轮廓
//	Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
//	for (int i = 0; i< contours.size(); i++)
//	{
//		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
//		drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
//	}
//
//	/// 在窗体中显示结果
//	namedWindow("Contours", CV_WINDOW_AUTOSIZE);
//	imshow("Contours", drawing);
//}

////tubao convexHull
//
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include <iostream>
//#include <stdio.h>
//#include <stdlib.h>
//
//using namespace cv;
//using namespace std;
//
//Mat src; Mat src_gray;
//int thresh = 100;
//int max_thresh = 255;
//RNG rng(12345);
//
///// Function header
//void thresh_callback(int, void*);
//
///** @function main */
//int main(int argc, char** argv)
//{
//	/// 加载源图像
//	src = imread("..\\..\\data\\lena.jpg", 1);
//
//	/// 转成灰度图并进行模糊降噪
//	cvtColor(src, src_gray, CV_BGR2GRAY);
//	blur(src_gray, src_gray, Size(3, 3));
//
//	/// 创建窗体
//	const char* source_window = "Source";
//	namedWindow(source_window, CV_WINDOW_AUTOSIZE);
//	imshow(source_window, src);
//
//	createTrackbar(" Threshold:", "Source", &thresh, max_thresh, thresh_callback);
//	thresh_callback(0, 0);
//
//	waitKey(0);
//	return(0);
//}
//
///** @function thresh_callback */
//void thresh_callback(int, void*)
//{
//	Mat src_copy = src.clone();
//	Mat threshold_output;
//	vector<vector<Point> > contours;
//	vector<Vec4i> hierarchy;
//
//	/// 对图像进行二值化
//	threshold(src_gray, threshold_output, thresh, 255, THRESH_BINARY);
//
//	/// 寻找轮廓
//	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
//
//	/// 对每个轮廓计算其凸包
//	vector<vector<Point> >hull(contours.size());
//	for (int i = 0; i < contours.size(); i++)
//	{
//		convexHull(Mat(contours[i]), hull[i], false);
//	}
//
//	/// 绘出轮廓及其凸包
//	Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
//	for (int i = 0; i< contours.size(); i++)
//	{
//		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
//		drawContours(drawing, contours, i, color, 1, 8, vector<Vec4i>(), 0, Point());
//		drawContours(drawing, hull, i, color, 1, 8, vector<Vec4i>(), 0, Point());
//	}
//
//	/// 把结果显示在窗体
//	namedWindow("Hull demo", CV_WINDOW_AUTOSIZE);
//	imshow("Hull demo", drawing);
//}

////boundingrect minennclosingcircle
//
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include <iostream>
//#include <stdio.h>
//#include <stdlib.h>
//
//using namespace cv;
//using namespace std;
//
//Mat src; Mat src_gray;
//int thresh = 100;
//int max_thresh = 255;
//RNG rng(12345);
//
///// 函数声明
//void thresh_callback(int, void*);
//
///** @主函数 */
//int main(int argc, char** argv)
//{
//	/// 载入原图像, 返回3通道图像
//	src = imread("..\\..\\data\\lena.jpg", 1);
//
//	/// 转化成灰度图像并进行平滑
//	cvtColor(src, src_gray, CV_BGR2GRAY);
//	blur(src_gray, src_gray, Size(3, 3));
//
//	/// 创建窗口
//	const char* source_window = "Source";
//	namedWindow(source_window, CV_WINDOW_AUTOSIZE);
//	imshow(source_window, src);
//
//	createTrackbar(" Threshold:", "Source", &thresh, max_thresh, thresh_callback);
//	thresh_callback(0, 0);
//
//	waitKey(0);
//	return(0);
//}
//
///** @thresh_callback 函数 */
//void thresh_callback(int, void*)
//{
//	Mat threshold_output;
//	vector<vector<Point> > contours;
//	vector<Vec4i> hierarchy;
//
//	/// 使用Threshold检测边缘
//	threshold(src_gray, threshold_output, thresh, 255, THRESH_BINARY);
//	/// 找到轮廓
//	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
//
//	/// 多边形逼近轮廓 + 获取矩形和圆形边界框
//	vector<vector<Point> > contours_poly(contours.size());
//	vector<Rect> boundRect(contours.size());
//	vector<Point2f>center(contours.size());
//	vector<float>radius(contours.size());
//
//	for (int i = 0; i < contours.size(); i++)
//	{
//		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
//		boundRect[i] = boundingRect(Mat(contours_poly[i]));
//		minEnclosingCircle(contours_poly[i], center[i], radius[i]);
//	}
//
//
//	/// 画多边形轮廓 + 包围的矩形框 + 圆形框
//	Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
//	for (int i = 0; i< contours.size(); i++)
//	{
//		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
//		drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
//		rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
//		circle(drawing, center[i], (int)radius[i], color, 2, 8, 0);
//	}
//
//	/// 显示在一个窗口
//	namedWindow("Contours", CV_WINDOW_AUTOSIZE);
//	imshow("Contours", drawing);
//}

////Video PSNR SSIM
//
//#include <iostream> // for standard I/O
//#include <string>   // for strings
//#include <iomanip>  // for controlling float print precision
//#include <sstream>  // string to number conversion
//
//#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
//#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
//#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O
//
//using namespace std;
//using namespace cv;
//
//double getPSNR(const Mat& I1, const Mat& I2);
//Scalar getMSSIM(const Mat& I1, const Mat& I2);
//int main(int argc, char *argv[], char *window_name)
//{
//	/*if (argc != 5)
//	{
//		cout << "Not enough parameters" << endl;
//		return -1;
//	}*/
//	stringstream conv;
//
//	const string sourceReference = "..\\..\\data\\cankao.avi", sourceCompareWith = "..\\..\\data\\yasuo.avi";
//	int psnrTriggerValue, delay;
//	//conv << argv[3] << endl << argv[4];       // put in the strings
//	conv >> psnrTriggerValue >> delay;// take out the numbers
//
//	char c;
//	int frameNum = -1;          // Frame counter
//
//	VideoCapture captRefrnc(sourceReference),
//		captUndTst(sourceCompareWith);
//
//	if (!captRefrnc.isOpened())
//	{
//		cout << "Could not open reference " << sourceReference << endl;
//		return -1;
//	}
//
//	if (!captUndTst.isOpened())
//	{
//		cout << "Could not open case test " << sourceCompareWith << endl;
//		return -1;
//	}
//
//	Size refS = Size((int)captRefrnc.get(CV_CAP_PROP_FRAME_WIDTH),
//		(int)captRefrnc.get(CV_CAP_PROP_FRAME_HEIGHT)),
//		uTSi = Size((int)captUndTst.get(CV_CAP_PROP_FRAME_WIDTH),
//		(int)captUndTst.get(CV_CAP_PROP_FRAME_HEIGHT));
//
//	if (refS != uTSi)
//	{
//		cout << "Inputs have different size!!! Closing." << endl;
//		return -1;
//	}
//
//	const char* WIN_UT = "Under Test";
//	const char* WIN_RF = "Reference";
//
//	// Windows
//	namedWindow(WIN_RF, CV_WINDOW_AUTOSIZE);
//	namedWindow(WIN_UT, CV_WINDOW_AUTOSIZE);
//	cvMoveWindow(WIN_RF, 400, 0);      //750,  2 (bernat =0)
//	cvMoveWindow(WIN_UT, refS.width, 0);      //1500, 2
//
//	cout << "Reference frame resolution: Width=" << refS.width << "  Height=" << refS.height
//		<< " of nr#: " << captRefrnc.get(CV_CAP_PROP_FRAME_COUNT) << endl;
//
//	cout << "PSNR trigger value " <<
//		setiosflags(ios::fixed) << setprecision(3) << psnrTriggerValue << endl;
//
//	Mat frameReference, frameUnderTest;
//	double psnrV;
//	Scalar mssimV;
//
//	while (true) //Show the image captured in the window and repeat
//	{
//		captRefrnc >> frameReference;
//		captUndTst >> frameUnderTest;
//
//		if (frameReference.empty() || frameUnderTest.empty())
//		{
//			cout << " < < <  Game over!  > > > ";
//			break;
//		}
//
//		++frameNum;
//		cout << "Frame:" << frameNum << "# ";
//
//		///////////////////////////////// PSNR ////////////////////////////////////////////////////
//		psnrV = getPSNR(frameReference, frameUnderTest);                 //get PSNR
//		cout << setiosflags(ios::fixed) << setprecision(3) << psnrV << "dB";
//
//		//////////////////////////////////// MSSIM /////////////////////////////////////////////////
//		if (psnrV < psnrTriggerValue && psnrV)
//		{
//			mssimV = getMSSIM(frameReference, frameUnderTest);
//
//			cout << " MSSIM: "
//				<< " R " << setiosflags(ios::fixed) << setprecision(2) << mssimV.val[2] * 100 << "%"
//				<< " G " << setiosflags(ios::fixed) << setprecision(2) << mssimV.val[1] * 100 << "%"
//				<< " B " << setiosflags(ios::fixed) << setprecision(2) << mssimV.val[0] * 100 << "%";
//		}
//
//		cout << endl;
//
//		////////////////////////////////// Show Image /////////////////////////////////////////////
//		imshow(WIN_RF, frameReference);
//		imshow(WIN_UT, frameUnderTest);
//
//		c = cvWaitKey(delay);
//		if (c == 27) break;
//	}
//
//	return 0;
//}
//
//double getPSNR(const Mat& I1, const Mat& I2)
//{
//	Mat s1;
//	absdiff(I1, I2, s1);       // |I1 - I2|
//	s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
//	s1 = s1.mul(s1);           // |I1 - I2|^2
//
//	Scalar s = sum(s1);         // sum elements per channel
//
//	double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels
//
//	if (sse <= 1e-10) // for small values return zero
//		return 0;
//	else
//	{
//		double  mse = sse / (double)(I1.channels() * I1.total());
//		double psnr = 10.0*log10((255 * 255) / mse);
//		return psnr;
//	}
//}
//
//Scalar getMSSIM(const Mat& i1, const Mat& i2)
//{
//	const double C1 = 6.5025, C2 = 58.5225;
//	/***************************** INITS **********************************/
//	int d = CV_32F;
//
//	Mat I1, I2;
//	i1.convertTo(I1, d);           // cannot calculate on one byte large values
//	i2.convertTo(I2, d);
//
//	Mat I2_2 = I2.mul(I2);        // I2^2
//	Mat I1_2 = I1.mul(I1);        // I1^2
//	Mat I1_I2 = I1.mul(I2);        // I1 * I2
//
//								   /*************************** END INITS **********************************/
//
//	Mat mu1, mu2;   // PRELIMINARY COMPUTING
//	GaussianBlur(I1, mu1, Size(11, 11), 1.5);
//	GaussianBlur(I2, mu2, Size(11, 11), 1.5);
//
//	Mat mu1_2 = mu1.mul(mu1);
//	Mat mu2_2 = mu2.mul(mu2);
//	Mat mu1_mu2 = mu1.mul(mu2);
//
//	Mat sigma1_2, sigma2_2, sigma12;
//
//	GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
//	sigma1_2 -= mu1_2;
//
//	GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
//	sigma2_2 -= mu2_2;
//
//	GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
//	sigma12 -= mu1_mu2;
//
//	///////////////////////////////// FORMULA ////////////////////////////////
//	Mat t1, t2, t3;
//
//	t1 = 2 * mu1_mu2 + C1;
//	t2 = 2 * sigma12 + C2;
//	t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
//
//	t1 = mu1_2 + mu2_2 + C1;
//	t2 = sigma1_2 + sigma2_2 + C2;
//	t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
//
//	Mat ssim_map;
//	divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;
//
//	Scalar mssim = mean(ssim_map); // mssim = average of ssim map
//	return mssim;
//}

////Harris jiaodian
//
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include <iostream>
//#include <stdio.h>
//#include <stdlib.h>
//
//using namespace cv;
//using namespace std;
//
///// Global variables
//Mat src, src_gray;
//int thresh = 200;
//int max_thresh = 255;
//
//const char* source_window = "Source image";
//const char* corners_window = "Corners detected";
//
///// Function header
//void cornerHarris_demo(int, void*);
//
///** @function main */
//int main(int argc, char** argv)
//{
//	/// Load source image and convert it to gray
//	src = imread("..\\..\\data\\lena.jpg", 1);
//	cvtColor(src, src_gray, CV_BGR2GRAY);
//
//	/// Create a window and a trackbar
//	namedWindow(source_window, CV_WINDOW_AUTOSIZE);
//	createTrackbar("Threshold: ", source_window, &thresh, max_thresh, cornerHarris_demo);
//	imshow(source_window, src);
//
//	cornerHarris_demo(0, 0);
//
//	waitKey(0);
//	return(0);
//}
//
///** @function cornerHarris_demo */
//void cornerHarris_demo(int, void*)
//{
//
//	Mat dst, dst_norm, dst_norm_scaled;
//	dst = Mat::zeros(src.size(), CV_32FC1);
//
//	/// Detector parameters
//	int blockSize = 2;
//	int apertureSize = 3;
//	double k = 0.04;
//
//	/// Detecting corners
//	cornerHarris(src_gray, dst, blockSize, apertureSize, k, BORDER_DEFAULT);
//
//	/// Normalizing
//	normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
//	convertScaleAbs(dst_norm, dst_norm_scaled);
//
//	/// Drawing a circle around corners
//	for (int j = 0; j < dst_norm.rows; j++)
//	{
//		for (int i = 0; i < dst_norm.cols; i++)
//		{
//			if ((int)dst_norm.at<float>(j, i) > thresh)
//			{
//				circle(dst_norm_scaled, Point(i, j), 5, Scalar(0), 2, 8, 0);
//			}
//		}
//	}
//	/// Showing the result
//	namedWindow(corners_window, CV_WINDOW_AUTOSIZE);
//	imshow(corners_window, dst_norm_scaled);
//}

////FLANN tezhendianpipei
//
//#include <stdio.h>
//#include <iostream>
//#include "opencv2/core/core.hpp"
//#include "opencv2/features2d/features2d.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include <opencv2/nonfree/features2d.hpp>
////#include <opencv2/nonfree/nonfree.hpp>
//
//using namespace cv;
//
//void readme();
//
///** @function main */
//int main(int argc, char** argv)
//{
//	//if (argc != 3)
//	//{
//	//	readme(); return -1;
//	//}
//
//	Mat img_1 = imread("..\\..\\data\\lena.jpg", CV_LOAD_IMAGE_GRAYSCALE);
//	Mat img_2 = imread("..\\..\\data\\lenafind.jpg", CV_LOAD_IMAGE_GRAYSCALE);
//
//	if (!img_1.data || !img_2.data)
//	{
//		std::cout << " --(!) Error reading images " << std::endl; return -1;
//	}
//
//	//-- Step 1: Detect the keypoints using SURF Detector
//	int minHessian = 400;
//
//	SurfFeatureDetector detector(minHessian);
//
//	std::vector<KeyPoint> keypoints_1, keypoints_2;
//
//	detector.detect(img_1, keypoints_1);
//	detector.detect(img_2, keypoints_2);
//
//	//-- Step 2: Calculate descriptors (feature vectors)
//	SurfDescriptorExtractor extractor;
//
//	Mat descriptors_1, descriptors_2;
//
//	extractor.compute(img_1, keypoints_1, descriptors_1);
//	extractor.compute(img_2, keypoints_2, descriptors_2);
//
//	//-- Step 3: Matching descriptor vectors using FLANN matcher
//	FlannBasedMatcher matcher;
//	std::vector< DMatch > matches;
//	matcher.match(descriptors_1, descriptors_2, matches);
//
//	double max_dist = 0; double min_dist = 100;
//
//	//-- Quick calculation of max and min distances between keypoints
//	for (int i = 0; i < descriptors_1.rows; i++)
//	{
//		double dist = matches[i].distance;
//		if (dist < min_dist) min_dist = dist;
//		if (dist > max_dist) max_dist = dist;
//	}
//
//	printf("-- Max dist : %f \n", max_dist);
//	printf("-- Min dist : %f \n", min_dist);
//
//	//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist )
//	//-- PS.- radiusMatch can also be used here.
//	std::vector< DMatch > good_matches;
//
//	for (int i = 0; i < descriptors_1.rows; i++)
//	{
//		if (matches[i].distance < 2 * min_dist)
//		{
//			good_matches.push_back(matches[i]);
//		}
//	}
//
//	//-- Draw only "good" matches
//	Mat img_matches;
//	drawMatches(img_1, keypoints_1, img_2, keypoints_2,
//		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
//		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
//
//	//-- Show detected matches
//	imshow("Good Matches", img_matches);
//
//	for (int i = 0; i < good_matches.size(); i++)
//	{
//		printf("-- Good Match [%d] Keypoint 1: %d  -- Keypoint 2: %d  \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx);
//	}
//
//	waitKey(0);
//
//	return 0;
//}
//
///** @function readme */
//void readme()
//{
//	std::cout << " Usage: ./SURF_FlannMatcher <img1> <img2>" << std::endl;
//}

////shi-tomasi jiaodian
//
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//#include <iostream>
//#include <stdio.h>
//#include <stdlib.h>
//
//using namespace cv;
//using namespace std;
//
///// Global variables
//Mat src, src_gray;
//
//int maxCorners = 23;
//int maxTrackbar = 100;
//
//RNG rng(12345);
//const char* source_window = "Image";
//
///// Function header
//void goodFeaturesToTrack_Demo(int, void*);
//
///**
//* @function main
//*/
//int main(int argc, char** argv)
//{
//	/// Load source image and convert it to gray
//	src = imread("..\\..\\data\\lena.jpg", 1);
//	cvtColor(src, src_gray, CV_BGR2GRAY);
//
//	/// Create Window
//	namedWindow(source_window, CV_WINDOW_AUTOSIZE);
//
//	/// Create Trackbar to set the number of corners
//	createTrackbar("Max  corners:", source_window, &maxCorners, maxTrackbar, goodFeaturesToTrack_Demo);
//
//	imshow(source_window, src);
//
//	goodFeaturesToTrack_Demo(0, 0);
//
//	waitKey(0);
//	return(0);
//}
//
///**
//* @function goodFeaturesToTrack_Demo.cpp
//* @brief Apply Shi-Tomasi corner detector
//*/
//void goodFeaturesToTrack_Demo(int, void*)
//{
//	if (maxCorners < 1) { maxCorners = 1; }
//
//	/// Parameters for Shi-Tomasi algorithm
//	vector<Point2f> corners;
//	double qualityLevel = 0.01;
//	double minDistance = 10;
//	int blockSize = 3;
//	bool useHarrisDetector = false;
//	double k = 0.04;
//
//	/// Copy the source image
//	Mat copy;
//	copy = src.clone();
//
//	/// Apply corner detection
//	goodFeaturesToTrack(src_gray,
//		corners,
//		maxCorners,
//		qualityLevel,
//		minDistance,
//		Mat(),
//		blockSize,
//		useHarrisDetector,
//		k);
//
//
//	/// Draw corners detected
//	cout << "** Number of corners detected: " << corners.size() << endl;
//	int r = 4;
//	for (int i = 0; i < corners.size(); i++)
//	{
//		circle(copy, corners[i], r, Scalar(rng.uniform(0, 255), rng.uniform(0, 255),
//			rng.uniform(0, 255)), -1, 8, 0);
//	}
//
//	/// Show what you got
//	namedWindow(source_window, CV_WINDOW_AUTOSIZE);
//	imshow(source_window, copy);
//}

////寻找已知物体
//
//#include <stdio.h>
//#include <iostream>
//#include "opencv2/core/core.hpp"
//#include "opencv2/features2d/features2d.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/calib3d/calib3d.hpp"
//#include <opencv2/nonfree/features2d.hpp>
//
//using namespace cv;
//
//void readme();
//
///** @function main */
//int main(int argc, char** argv)
//{
//	//if (argc != 3)
//	//{
//	//	readme(); return -1;
//	//}
//
//	Mat img_object = imread("..\\..\\data\\lena.jpg", CV_LOAD_IMAGE_GRAYSCALE);
//	Mat img_scene = imread("..\\..\\data\\lenafind.jpg", CV_LOAD_IMAGE_GRAYSCALE);
//
//	if (!img_object.data || !img_scene.data)
//	{
//		std::cout << " --(!) Error reading images " << std::endl; return -1;
//	}
//
//	//-- Step 1: Detect the keypoints using SURF Detector
//	int minHessian = 400;
//
//	SurfFeatureDetector detector(minHessian);
//
//	std::vector<KeyPoint> keypoints_object, keypoints_scene;
//
//	detector.detect(img_object, keypoints_object);
//	detector.detect(img_scene, keypoints_scene);
//
//	//-- Step 2: Calculate descriptors (feature vectors)
//	SurfDescriptorExtractor extractor;
//
//	Mat descriptors_object, descriptors_scene;
//
//	extractor.compute(img_object, keypoints_object, descriptors_object);
//	extractor.compute(img_scene, keypoints_scene, descriptors_scene);
//
//	//-- Step 3: Matching descriptor vectors using FLANN matcher
//	FlannBasedMatcher matcher;
//	std::vector< DMatch > matches;
//	matcher.match(descriptors_object, descriptors_scene, matches);
//
//	double max_dist = 0; double min_dist = 100;
//
//	//-- Quick calculation of max and min distances between keypoints
//	for (int i = 0; i < descriptors_object.rows; i++)
//	{
//		double dist = matches[i].distance;
//		if (dist < min_dist) min_dist = dist;
//		if (dist > max_dist) max_dist = dist;
//	}
//
//	printf("-- Max dist : %f \n", max_dist);
//	printf("-- Min dist : %f \n", min_dist);
//
//	//-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
//	std::vector< DMatch > good_matches;
//
//	for (int i = 0; i < descriptors_object.rows; i++)
//	{
//		if (matches[i].distance < 3 * min_dist)
//		{
//			good_matches.push_back(matches[i]);
//		}
//	}
//
//	Mat img_matches;
//	drawMatches(img_object, keypoints_object, img_scene, keypoints_scene,
//		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
//		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
//
//	//-- Localize the object
//	std::vector<Point2f> obj;
//	std::vector<Point2f> scene;
//
//	for (int i = 0; i < good_matches.size(); i++)
//	{
//		//-- Get the keypoints from the good matches
//		obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
//		scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
//	}
//
//	Mat H = findHomography(obj, scene, CV_RANSAC);
//
//	//-- Get the corners from the image_1 ( the object to be "detected" )
//	std::vector<Point2f> obj_corners(4);
//	obj_corners[0] = cvPoint(0, 0);
//	obj_corners[1] = cvPoint(img_object.cols, 0);
//	obj_corners[2] = cvPoint(img_object.cols, img_object.rows);
//	obj_corners[3] = cvPoint(0, img_object.rows);
//	std::vector<Point2f> scene_corners(4);
//
//	perspectiveTransform(obj_corners, scene_corners, H);
//
//	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
//	line(img_matches, scene_corners[0] + Point2f(img_object.cols, 0), scene_corners[1] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
//	line(img_matches, scene_corners[1] + Point2f(img_object.cols, 0), scene_corners[2] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
//	line(img_matches, scene_corners[2] + Point2f(img_object.cols, 0), scene_corners[3] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
//	line(img_matches, scene_corners[3] + Point2f(img_object.cols, 0), scene_corners[0] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
//
//	//-- Show detected matches
//	imshow("Good Matches & Object detection", img_matches);
//
//	waitKey(0);
//	return 0;
//}
//
///** @function readme */
//void readme()
//{
//	std::cout << " Usage: ./SURF_descriptor <img1> <img2>" << std::endl;
//}

////级联分类器
//
//#include "opencv2/objdetect/objdetect.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//
//#include <iostream>
//#include <stdio.h>
//
//using namespace std;
//using namespace cv;
//
///** 函数声明 */
//void detectAndDisplay(Mat frame);
//
///** 全局变量 */
//string face_cascade_name = "haarcascade_frontalface_alt.xml";
//string eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
//CascadeClassifier face_cascade;
//CascadeClassifier eyes_cascade;
//string window_name = "Capture - Face detection";
//RNG rng(12345);
//
///** @主函数 */
//int main(int argc, const char** argv)
//{
//	CvCapture* capture;
//	Mat frame;
//
//	//-- 1. 加载级联分类器文件
//	if (!face_cascade.load(face_cascade_name)) { printf("--(!)Error loading\n"); return -1; };
//	if (!eyes_cascade.load(eyes_cascade_name)) { printf("--(!)Error loading\n"); return -1; };
//
//	//-- 2. 打开内置摄像头视频流
//	capture = cvCaptureFromCAM(1);
//	if (capture)
//	{
//		while (true)
//		{
//			frame = cvQueryFrame(capture);
//
//			//-- 3. 对当前帧使用分类器进行检测
//			if (!frame.empty())
//			{
//				detectAndDisplay(frame);
//			}
//			else
//			{
//				//printf(" --(!) No captured frame -- Break!"); break;
//				frame = cvQueryFrame(capture);
//			}
//
//			int c = waitKey(10);
//			if ((char)c == 'c') { break; }
//		}
//	}
//	return 0;
//}
//
///** @函数 detectAndDisplay */
//void detectAndDisplay(Mat frame)
//{
//	std::vector<Rect> faces;
//	Mat frame_gray;
//
//	cvtColor(frame, frame_gray, CV_BGR2GRAY);
//	equalizeHist(frame_gray, frame_gray);
//
//	//-- 多尺寸检测人脸
//	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
//
//	for (int i = 0; i < faces.size(); i++)
//	{
//		Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
//		ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
//
//		Mat faceROI = frame_gray(faces[i]);
//		std::vector<Rect> eyes;
//
//		//-- 在每张人脸上检测双眼
//		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
//
//		for (int j = 0; j < eyes.size(); j++)
//		{
//			Point center(faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5);
//			int radius = cvRound((eyes[j].width + eyes[i].height)*0.25);
//			circle(frame, center, radius, Scalar(255, 0, 0), 4, 8, 0);
//		}
//	}
//	//-- 显示结果图像
//	imshow(window_name, frame);
//}

////SVM
//
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/ml/ml.hpp>
//
//using namespace cv;
//
//int main()
//{
//	// Data for visual representation
//	int width = 512, height = 512;
//	Mat image = Mat::zeros(height, width, CV_8UC3);
//
//	// Set up training data
//	float labels[4] = { 1.0, -1.0, -1.0, -1.0 };
//	Mat labelsMat(3, 1, CV_32FC1, labels);
//
//	float trainingData[4][2] = { { 501, 10 },{ 255, 10 },{ 501, 255 },{ 10, 501 } };
//	Mat trainingDataMat(3, 2, CV_32FC1, trainingData);
//
//	// Set up SVM's parameters
//	CvSVMParams params;
//	params.svm_type = CvSVM::C_SVC;
//	params.kernel_type = CvSVM::LINEAR;
//	params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
//
//	// Train the SVM
//	CvSVM SVM;
//	SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);
//
//	Vec3b green(0, 255, 0), blue(255, 0, 0);
//	// Show the decision regions given by the SVM
//	for (int i = 0; i < image.rows; ++i)
//		for (int j = 0; j < image.cols; ++j)
//		{
//			Mat sampleMat = (Mat_<float>(1, 2) << i, j);
//			float response = SVM.predict(sampleMat);
//
//			if (response == 1)
//				image.at<Vec3b>(j, i) = green;
//			else if (response == -1)
//				image.at<Vec3b>(j, i) = blue;
//		}
//
//	// Show the training data
//	int thickness = -1;
//	int lineType = 8;
//	circle(image, Point(501, 10), 5, Scalar(0, 0, 0), thickness, lineType);
//	circle(image, Point(255, 10), 5, Scalar(255, 255, 255), thickness, lineType);
//	circle(image, Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType);
//	circle(image, Point(10, 501), 5, Scalar(255, 255, 255), thickness, lineType);
//
//	// Show support vectors
//	thickness = 2;
//	lineType = 8;
//	int c = SVM.get_support_vector_count();
//
//	for (int i = 0; i < c; ++i)
//	{
//		const float* v = SVM.get_support_vector(i);
//		circle(image, Point((int)v[0], (int)v[1]), 6, Scalar(0, 0, 128), thickness, lineType);
//	}
//
//	imwrite("result.png", image);        // save the image 
//
//	imshow("SVM Simple Example", image); // show it to the user
//	waitKey(0);
//
//}

////SVM feixianxing
//
//#include <iostream>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/ml/ml.hpp>
//
//#define NTRAINING_SAMPLES   100         // Number of training samples per class
//#define FRAC_LINEAR_SEP     0.9f        // Fraction of samples which compose the linear separable part
//
//using namespace cv;
//using namespace std;
//
//int main()
//{
//	// Data for visual representation
//	const int WIDTH = 512, HEIGHT = 512;
//	Mat I = Mat::zeros(HEIGHT, WIDTH, CV_8UC3);
//
//	//--------------------- 1. Set up training data randomly ---------------------------------------
//	Mat trainData(2 * NTRAINING_SAMPLES, 2, CV_32FC1);
//	Mat labels(2 * NTRAINING_SAMPLES, 1, CV_32FC1);
//
//	RNG rng(100); // Random value generation class
//
//				  // Set up the linearly separable part of the training data
//	int nLinearSamples = (int)(FRAC_LINEAR_SEP * NTRAINING_SAMPLES);
//
//	// Generate random points for the class 1
//	Mat trainClass = trainData.rowRange(0, nLinearSamples);
//	// The x coordinate of the points is in [0, 0.4)
//	Mat c = trainClass.colRange(0, 1);
//	rng.fill(c, RNG::UNIFORM, Scalar(1), Scalar(0.4 * WIDTH));
//	// The y coordinate of the points is in [0, 1)
//	c = trainClass.colRange(1, 2);
//	rng.fill(c, RNG::UNIFORM, Scalar(1), Scalar(HEIGHT));
//
//	// Generate random points for the class 2
//	trainClass = trainData.rowRange(2 * NTRAINING_SAMPLES - nLinearSamples, 2 * NTRAINING_SAMPLES);
//	// The x coordinate of the points is in [0.6, 1]
//	c = trainClass.colRange(0, 1);
//	rng.fill(c, RNG::UNIFORM, Scalar(0.6*WIDTH), Scalar(WIDTH));
//	// The y coordinate of the points is in [0, 1)
//	c = trainClass.colRange(1, 2);
//	rng.fill(c, RNG::UNIFORM, Scalar(1), Scalar(HEIGHT));
//
//	//------------------ Set up the non-linearly separable part of the training data ---------------
//
//	// Generate random points for the classes 1 and 2
//	trainClass = trainData.rowRange(nLinearSamples, 2 * NTRAINING_SAMPLES - nLinearSamples);
//	// The x coordinate of the points is in [0.4, 0.6)
//	c = trainClass.colRange(0, 1);
//	rng.fill(c, RNG::UNIFORM, Scalar(0.4*WIDTH), Scalar(0.6*WIDTH));
//	// The y coordinate of the points is in [0, 1)
//	c = trainClass.colRange(1, 2);
//	rng.fill(c, RNG::UNIFORM, Scalar(1), Scalar(HEIGHT));
//
//	//------------------------- Set up the labels for the classes ---------------------------------
//	labels.rowRange(0, NTRAINING_SAMPLES).setTo(1);  // Class 1
//	labels.rowRange(NTRAINING_SAMPLES, 2 * NTRAINING_SAMPLES).setTo(2);  // Class 2
//
//																		 //------------------------ 2. Set up the support vector machines parameters --------------------
//	CvSVMParams params;
//	params.svm_type = SVM::C_SVC;
//	params.C = 0.1;/////
//	params.kernel_type = SVM::LINEAR;
//	params.term_crit = TermCriteria(CV_TERMCRIT_ITER, (int)1e7, 1e-6);
//
//	//------------------------ 3. Train the svm ----------------------------------------------------
//	cout << "Starting training process" << endl;
//	CvSVM svm;
//	svm.train(trainData, labels, Mat(), Mat(), params);
//	cout << "Finished training process" << endl;
//
//	//------------------------ 4. Show the decision regions ----------------------------------------
//	Vec3b green(0, 100, 0), blue(100, 0, 0);
//	for (int i = 0; i < I.rows; ++i)
//		for (int j = 0; j < I.cols; ++j)
//		{
//			Mat sampleMat = (Mat_<float>(1, 2) << i, j);
//			float response = svm.predict(sampleMat);
//
//			if (response == 1)    I.at<Vec3b>(j, i) = green;
//			else if (response == 2)    I.at<Vec3b>(j, i) = blue;
//		}
//
//	//----------------------- 5. Show the training data --------------------------------------------
//	int thick = -1;
//	int lineType = 8;
//	float px, py;
//	// Class 1
//	for (int i = 0; i < NTRAINING_SAMPLES; ++i)
//	{
//		px = trainData.at<float>(i, 0);
//		py = trainData.at<float>(i, 1);
//		circle(I, Point((int)px, (int)py), 3, Scalar(0, 255, 0), thick, lineType);
//	}
//	// Class 2
//	for (int i = NTRAINING_SAMPLES; i <2 * NTRAINING_SAMPLES; ++i)
//	{
//		px = trainData.at<float>(i, 0);
//		py = trainData.at<float>(i, 1);
//		circle(I, Point((int)px, (int)py), 3, Scalar(255, 0, 0), thick, lineType);
//	}
//
//	//------------------------- 6. Show support vectors --------------------------------------------
//	thick = 2;
//	lineType = 8;
//	int x = svm.get_support_vector_count();
//
//	for (int i = 0; i < x; ++i)
//	{
//		const float* v = svm.get_support_vector(i);
//		circle(I, Point((int)v[0], (int)v[1]), 6, Scalar(128, 128, 128), thick, lineType);
//	}
//
//	imwrite("result.png", I);                      // save the Image
//	imshow("SVM for Non-Linear Training Data", I); // show it to the user
//	waitKey(0);
//}
