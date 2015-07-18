#include <stdio.h>
#include "CaffeCls.hpp"
#include "imagehelper.hpp"


using namespace std;
using namespace sdktest;

int main() {

	/*
	   ImageHelper *helper = new ImageHelperBackend();
	   Image *img2 = helper->LoadBGRImage("1.jpg");
	   helper->DrawRect(img2, 10,10,60,60, 2);
	   helper->SaveImage("2.jpg", img2);
	   helper->FreeImage(img2);
	   delete helper;
	   */

	FastRCNNCls* fastRCNN = new FastRCNNCls;
	Mat img = imread("/home/user/3.jpg");
	printf("load image ok %d %d %d\n", img.cols, img.rows, img.channels());

	fastRCNN->init_box_model("../models/People/propose.model");
	fastRCNN->set_model("../models/CaffeNet/test.prototxt", 
			"../output/default/clothesDataset_500/caffenet_fast_rcnn_iter_40000.caffemodel");
	float beginTime =clock();

	fastRCNN->get_boxes(img);

	float endTime = clock();

	cout << "Time: " << (endTime - beginTime)/CLOCKS_PER_SEC << 's' << endl;

	vector<int> tgt_cls(26);
	for (int i=0; i<26; i++) {
		tgt_cls[i] = i+1;
	}

	vector<PropBox> randPboxes = fastRCNN->detect(img, tgt_cls);
	/* sort the results according to their confidence */
	vector<PropBox> pboxes(randPboxes.size());
	for (int iSort = 0; iSort < randPboxes.size(); iSort ++) {
		cout<<"Sorting the "<<iSort<<"ones"<<endl;
		float maxVal = 0;
		int maxIndex = 0;
		/* find the max value */
		for (int iFind = 0; iFind < randPboxes.size(); iFind ++) {
			if (randPboxes[iFind].confidence > maxVal) {
				cout<<"The "<<iFind<<" is bigger than "<<maxIndex<<endl;
				cout<<"With "<<randPboxes[iFind].confidence<<" Bigger than "<<maxVal<<endl;
				maxVal = randPboxes[iFind].confidence;
				maxIndex = iFind;
			}
		}
		pboxes[iSort] = randPboxes[maxIndex];
		cout<<pboxes[iSort].confidence;
		randPboxes[maxIndex].confidence = 0;
	}


	ImageHelper *helper = new ImageHelperBackend();
	Image *img_show = helper->CreateImage(img.cols, img.rows, Image::Image_BGR);
	memcpy(img_show->data, img.data, sizeof(unsigned char) * img.rows * img.cols * 3);
	helper->DrawRect(img_show, pboxes[0].y1, pboxes[0].x1, pboxes[0].x2, pboxes[0].y2, 3);
	helper->SaveImage("abcd.jpg", img_show);



	for (int i=0; i<pboxes.size(); i++) {
		printf("%f in the %d class\n", pboxes[i].confidence, pboxes[i].cls_id);
		fastRCNN->plotresult(img, pboxes[i]);
	}


	helper->FreeImage(img_show);
	delete helper;
	return 0;	
}
