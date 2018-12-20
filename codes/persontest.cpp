#include <stdio.h>
#include <opencv2/opencv.hpp>



using namespace std;
using namespace cv;

//继承自CvSVM的类，因为生成setSVMDetector()中用到的检测子参数时，需要用到训练好的SVM的decision_func参数，
//但通过查看CvSVM源码可知decision_func参数是protected类型变量，无法直接访问到，只能继承之后通过函数访问
class MySVM : public CvSVM
{
public:
	//获得SVM的决策函数中的alpha数组
	double * get_alpha_vector()
	{
		return this->decision_func->alpha;
	}

	//获得SVM的决策函数中的rho参数,即偏移量
	float get_rho()
	{
		return this->decision_func->rho;
	}
};

void main() {

	Mat frame1;
	VideoCapture cap1;
	
	cap1.open("..\\..\\data\\new00.avi");
	//cap1.set(CV_CAP_PROP_FRAME_WIDTH, 60); //画面宽
	//cap1.set(CV_CAP_PROP_FRAME_HEIGHT, 40);//画面高

	if (!cap1.isOpened()) {
		
		cout << "cap1 can not open" << endl;

	}

	////HOG+SVM行人检测
	////【1】定义hog描述符
	//HOGDescriptor hog;
	////【2】设置SVM分类器
	//hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

	

	//检测窗口(64,128),块尺寸(16,16),块步长(8,8),cell尺寸(8,8),直方图bin个数9
	//HOGDescriptor hog(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);//HOG检测器，用来计算HOG描述子的
	int DescriptorDim;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定
	MySVM svm;//SVM分类器

	svm.load("SVM_HOG_2400PosINRIA_12000Neg_HardExample(误报少了漏检多了1).xml");//从XML文件读取训练好的SVM模型

	DescriptorDim = svm.get_var_count();//特征向量的维数，即HOG描述子的维数
	int supportVectorNum = svm.get_support_vector_count();//支持向量的个数
	cout << "支持向量个数：" << supportVectorNum << endl;

	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha向量，长度等于支持向量个数
	Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//支持向量矩阵
	Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha向量乘以支持向量矩阵的结果

														   //将支持向量的数据复制到supportVectorMat矩阵中
	for (int i = 0; i<supportVectorNum; i++)
	{
		const float * pSVData = svm.get_support_vector(i);//返回第i个支持向量的数据指针
		for (int j = 0; j<DescriptorDim; j++)
		{
			//cout<<pData[j]<<" ";
			supportVectorMat.at<float>(i, j) = pSVData[j];
		}
	}

	//将alpha向量的数据复制到alphaMat中
	double * pAlphaData = svm.get_alpha_vector();//返回SVM的决策函数中的alpha向量
	for (int i = 0; i<supportVectorNum; i++)
	{
		alphaMat.at<float>(0, i) = pAlphaData[i];
	}

	//计算-(alphaMat * supportVectorMat),结果放到resultMat中
	//gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);//不知道为什么加负号？
	resultMat = -1 * alphaMat * supportVectorMat;

	//得到最终的setSVMDetector(const vector<float>& detector)参数中可用的检测子
	vector<float> myDetector;
	//将resultMat中的数据复制到数组myDetector中
	for (int i = 0; i<DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0, i));
	}
	//最后添加偏移量rho，得到检测子
	myDetector.push_back(svm.get_rho());
	cout << "检测子维数：" << myDetector.size() << endl;
	//设置HOGDescriptor的检测子
	HOGDescriptor myHOG;
	myHOG.setSVMDetector(myDetector);


	for (;;) {

		cap1 >> frame1;

		if (frame1.empty()) {

			cout << "frame1 is empty" << endl;
		}

		double t1 = getTickCount();
		resize(frame1, frame1, Size(720, 405), 0, 0, INTER_LINEAR);
		t1 = (getTickCount() - t1) / getTickFrequency();
		cout << "resize 时间： " << t1 << endl;
		
		double t2 = getTickCount();
		//【3】在测试图像上检测行人区域
		std::vector<cv::Rect> regions,foundRect;
		myHOG.detectMultiScale(frame1, regions, 0, cv::Size(4, 4), cv::Size(0, 0), 1.04, 1);
		t2 = (getTickCount() - t2) / getTickFrequency();
		cout << "行人检测时间： " << t2 << endl;// "\n" << endl;

		double t3 = getTickCount();
		//for (size_t i = 0; i < regions.size(); i++)
		//{
		//	cv::rectangle(frame1, regions[i], cv::Scalar(0, 0, 255), 2);
		//}

		for (int i = 0; i < regions.size(); i++) {
			Rect r = regions[i];

			int j = 0;
			for (; j < regions.size(); j++) {
				//如果时嵌套的就推出循环
				if (j != i && (r & regions[j]) == r)
					break;
			}
			if (j == regions.size()) {
				foundRect.push_back(r);
			}
		}

		//画长方形，圈出行人
		for (int i = 0; i < foundRect.size(); i++) {
			Rect r = foundRect[i];
			rectangle(frame1, r.tl(), r.br(), Scalar(0, 0, 255), 3);
		}

		t3 = (getTickCount() - t3) / getTickFrequency();
		cout << "画框框时间： " << t3 <<"\n"<< endl;

		imshow("fram1", frame1);
		
		if (waitKey(10) >= 0) {

			break;
		}


	}

	waitKey(0);



}
