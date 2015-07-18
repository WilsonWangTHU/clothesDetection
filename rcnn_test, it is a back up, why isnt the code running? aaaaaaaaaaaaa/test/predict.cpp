#include <stdio.h>
#include "CaffeCls.hpp"
#include "imagehelper.hpp"
#include "readMapping.hpp"

using namespace std;
using namespace sdktest;
using namespace cv;

const int numCategory = 26;
const int fontFace = FONT_HERSHEY_SIMPLEX;
const float fontScale = 0.5;
const int thickness = 1;

int main(int argc, char** argv) {

	/* set the number of test, default is 10 */
	int numTestPerCls, showResult;
	if (argc > 2) { 
		numTestPerCls = atoi(argv[1]);
		showResult = atoi(argv[2]);
	} else if (argc > 1) {
		numTestPerCls = atoi(argv[1]);
		showResult = 1;
	} else {
		numTestPerCls = 5;
		showResult = 1;
	}

	/* set the directories of the files */
	string outputDir = "/home/user/rccn_for_cloth/VisResults";
	string testDataDir = "/home/user/rccn_for_cloth/data/clothesDataset/test";

	/* setting the number of class, containing the background */
	vector<int> tgt_cls(numCategory);
	for (int i = 0; i < numCategory; i++) {
		tgt_cls[i] = i + 1;
	}

	/* initial the edge detection for getting proposals */
	FastRCNNCls* fastRCNN = new FastRCNNCls;
	fastRCNN->init_box_model("../models/People/propose.model");

	/* initial the caffe model */
	fastRCNN->set_model("/home/user/rccn_for_cloth/models/CaffeNet/test.prototxt", 
			"/home/user/rccn_for_cloth/output/default/clothesDataset/caffenet_fast_rcnn_iter_40000.caffemodel");

	///* initial image helper */
	//ImageHelper *helper = new ImageHelperBackend();

	for (int subCategory = 1; subCategory <= numCategory; subCategory ++) {

		/* read the mapping files */
		string subCategoryDir = testDataDir + "/" + to_string(subCategory);
		string imageSubDir = subCategoryDir + "/" + "images";
		string* imageName = NULL;
		int itemNumber;
		bool mappingflag = guidMapping::getImageName(
				subCategoryDir + "/" + "GUIDMapping.txt",
				&imageName, &itemNumber);
		if (mappingflag == false) {
			cout<<"Error reading the mapping files!"<<endl;
			return false;
		}


		/* processing each image */
		for (int nImage = 0; nImage< min(itemNumber, numTestPerCls); nImage ++) {

			float beginTime =clock();

			cout<<"Now processing the "<<nImage<<" in the "<<subCategory<<" th dir"<<endl;

			Mat img = imread(imageSubDir + "/" + imageName[nImage]);
			fastRCNN->get_boxes(img);

			/* forward into the caffe net */
			vector<PropBox> randPboxes = fastRCNN->detect(img, tgt_cls);

			/* sort the results according to their confidence */
			vector<PropBox> pboxes(randPboxes.size());
			for (int iSort = 0; iSort < pboxes.size(); iSort ++) {
				float maxVal = 0;
				int maxIndex = 0;
				/* find the max value */
				for (int iFind = 0; iFind < pboxes.size(); iFind ++) {
					if (randPboxes[iFind].confidence > maxVal) {
						maxVal = randPboxes[iFind].confidence;
						maxIndex = iFind;
					}
				}
				pboxes[iSort] = randPboxes[maxIndex];
				randPboxes[maxIndex].confidence = 0;
			}

			float endTime =clock();
			cout << "Time: " << (endTime - beginTime)/CLOCKS_PER_SEC << 's' << endl;

			/* plot the prediction infomation */
			for (int nResults = 0; nResults < showResult; nResults ++) {
				string preText = "P:" + to_string(pboxes[nResults].confidence) +
					"C:" + to_string(pboxes[nResults].cls_id);
				putText(img, preText,
						Point(int(pboxes[nResults].x1), int(pboxes[nResults].y1)),
						fontFace, fontScale, Scalar::all(0),
						thickness, 8);
				rectangle(img, 
						Point(int(pboxes[nResults].x1), int(pboxes[nResults].y1)),
						Point(int(pboxes[nResults].x2), int(pboxes[nResults].y2)),
						cvScalar(0, 0, 255), 2, 8, 0);
			}

			//Image *img_show = helper->CreateImage(img.cols, img.rows, Image::Image_BGR);
			//memcpy(img_show->data, img.data, sizeof(unsigned char) * img.rows * img.cols * 3);

			/* plot the results */
			string outputImageDir;
			if (pboxes[0].cls_id == subCategory) {
				outputImageDir = outputDir + "/" + 
					to_string(subCategory) + "/rightCls" 
					+ to_string(nImage) + ".jpg";
			} else outputImageDir = outputDir + "/" + 
				to_string(subCategory) + "/wrongCls" 
					+ to_string(nImage) + ".jpg";
			imwrite(outputImageDir, img);

			//helper->SaveImage(outputImageDir, img_show);
			//helper->FreeImage(img_show);

			img.release();

		}


		delete[] imageName;
	}


	//delete helper;
	delete fastRCNN;
	return 0;	
}
