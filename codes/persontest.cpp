#include <stdio.h>
#include <opencv2/opencv.hpp>



using namespace std;
using namespace cv;

//�̳���CvSVM���࣬��Ϊ����setSVMDetector()���õ��ļ���Ӳ���ʱ����Ҫ�õ�ѵ���õ�SVM��decision_func������
//��ͨ���鿴CvSVMԴ���֪decision_func������protected���ͱ������޷�ֱ�ӷ��ʵ���ֻ�ܼ̳�֮��ͨ����������
class MySVM : public CvSVM
{
public:
	//���SVM�ľ��ߺ����е�alpha����
	double * get_alpha_vector()
	{
		return this->decision_func->alpha;
	}

	//���SVM�ľ��ߺ����е�rho����,��ƫ����
	float get_rho()
	{
		return this->decision_func->rho;
	}
};

void main() {

	Mat frame1;
	VideoCapture cap1;
	
	cap1.open("..\\..\\data\\new00.avi");
	//cap1.set(CV_CAP_PROP_FRAME_WIDTH, 60); //�����
	//cap1.set(CV_CAP_PROP_FRAME_HEIGHT, 40);//�����

	if (!cap1.isOpened()) {
		
		cout << "cap1 can not open" << endl;

	}

	////HOG+SVM���˼��
	////��1������hog������
	//HOGDescriptor hog;
	////��2������SVM������
	//hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());

	

	//��ⴰ��(64,128),��ߴ�(16,16),�鲽��(8,8),cell�ߴ�(8,8),ֱ��ͼbin����9
	//HOGDescriptor hog(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);//HOG���������������HOG�����ӵ�
	int DescriptorDim;//HOG�����ӵ�ά������ͼƬ��С����ⴰ�ڴ�С�����С��ϸ����Ԫ��ֱ��ͼbin��������
	MySVM svm;//SVM������

	svm.load("SVM_HOG_2400PosINRIA_12000Neg_HardExample(������©�����1).xml");//��XML�ļ���ȡѵ���õ�SVMģ��

	DescriptorDim = svm.get_var_count();//����������ά������HOG�����ӵ�ά��
	int supportVectorNum = svm.get_support_vector_count();//֧�������ĸ���
	cout << "֧������������" << supportVectorNum << endl;

	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha���������ȵ���֧����������
	Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//֧����������
	Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha��������֧����������Ľ��

														   //��֧�����������ݸ��Ƶ�supportVectorMat������
	for (int i = 0; i<supportVectorNum; i++)
	{
		const float * pSVData = svm.get_support_vector(i);//���ص�i��֧������������ָ��
		for (int j = 0; j<DescriptorDim; j++)
		{
			//cout<<pData[j]<<" ";
			supportVectorMat.at<float>(i, j) = pSVData[j];
		}
	}

	//��alpha���������ݸ��Ƶ�alphaMat��
	double * pAlphaData = svm.get_alpha_vector();//����SVM�ľ��ߺ����е�alpha����
	for (int i = 0; i<supportVectorNum; i++)
	{
		alphaMat.at<float>(0, i) = pAlphaData[i];
	}

	//����-(alphaMat * supportVectorMat),����ŵ�resultMat��
	//gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);//��֪��Ϊʲô�Ӹ��ţ�
	resultMat = -1 * alphaMat * supportVectorMat;

	//�õ����յ�setSVMDetector(const vector<float>& detector)�����п��õļ����
	vector<float> myDetector;
	//��resultMat�е����ݸ��Ƶ�����myDetector��
	for (int i = 0; i<DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0, i));
	}
	//������ƫ����rho���õ������
	myDetector.push_back(svm.get_rho());
	cout << "�����ά����" << myDetector.size() << endl;
	//����HOGDescriptor�ļ����
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
		cout << "resize ʱ�䣺 " << t1 << endl;
		
		double t2 = getTickCount();
		//��3���ڲ���ͼ���ϼ����������
		std::vector<cv::Rect> regions,foundRect;
		myHOG.detectMultiScale(frame1, regions, 0, cv::Size(4, 4), cv::Size(0, 0), 1.04, 1);
		t2 = (getTickCount() - t2) / getTickFrequency();
		cout << "���˼��ʱ�䣺 " << t2 << endl;// "\n" << endl;

		double t3 = getTickCount();
		//for (size_t i = 0; i < regions.size(); i++)
		//{
		//	cv::rectangle(frame1, regions[i], cv::Scalar(0, 0, 255), 2);
		//}

		for (int i = 0; i < regions.size(); i++) {
			Rect r = regions[i];

			int j = 0;
			for (; j < regions.size(); j++) {
				//���ʱǶ�׵ľ��Ƴ�ѭ��
				if (j != i && (r & regions[j]) == r)
					break;
			}
			if (j == regions.size()) {
				foundRect.push_back(r);
			}
		}

		//�������Σ�Ȧ������
		for (int i = 0; i < foundRect.size(); i++) {
			Rect r = foundRect[i];
			rectangle(frame1, r.tl(), r.br(), Scalar(0, 0, 255), 3);
		}

		t3 = (getTickCount() - t3) / getTickFrequency();
		cout << "�����ʱ�䣺 " << t3 <<"\n"<< endl;

		imshow("fram1", frame1);
		
		if (waitKey(10) >= 0) {

			break;
		}


	}

	waitKey(0);



}
