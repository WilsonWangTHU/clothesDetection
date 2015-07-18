/*
 * processing into usefull results, there are few things to considered,
 * 1. top 1 accuracy, top 1 bounding box 
 * 2. top 5 accuracy, top 5 bounding box
 * */
#include <stdio.h>
#include "CaffeCls.hpp"
#include "imagehelper.hpp"
#include "clothxml_reader.hpp"
#include "dbFeature.hpp"
#include "readMapping.hpp"

using namespace std;
using namespace sdktest;
using namespace cv;
const int nPrediction = 5;
const float overlaps_threshold = 0.4;

const int numCategory = 26;
const int fontFace = FONT_HERSHEY_SIMPLEX;
const float fontScale = 0.8;
const int thickness = 2;

static int number_test = 0;
static int number_test_perCls[26];

static int accuracy_perCls_top1_box[26];
static int accuracy_perCls_top5_box[26];
static int accuracy_perCls_top1[26];
static int accuracy_perCls_top5[26];
static int pred_perCls[26][26];

int main() {

	string ROOT_DIR = "/home/user/rccn_for_cloth/data/clothesDataset/train";

	/* test for all the category in the dataset */
	for (int iCategory = 1; iCategory <= 26; iCategory ++) {
		int validFlag = 0;
		/* initialize the parameters */
		int nItem = 0;
		string* labelName = NULL;
		string* imageName = NULL;
		ofstream newGUID;
		newGUID.open(ROOT_DIR + "/" + to_string(iCategory) + "/newGUIDMapping.txt");

		/* read the detect result */
		/* read the true result */
		bool mappingflag = guidMapping::getImageAndLabelName(ROOT_DIR + "/" + 
				to_string(iCategory) + "/GUIDMapping.txt", &imageName, &labelName, &nItem);
		if (mappingflag == false) cerr<<"Fail reading the map files"<<endl;

		/* validate the number of images in the sub dir */
		for (int iImage = 0; iImage < nItem; iImage ++ ){

			/* read the true results */
			int true_classtype = 0;
			float true_gt_box[4] = {0, 0, 0, 0}; 
			bool clothflag = clothReader::readClothClass(
					ROOT_DIR + "/" + to_string(iCategory) + "/Label/" + labelName[iImage],
					&true_classtype, true_gt_box);
			if (clothflag == false) cerr<<"Error reading the map files"<<endl;
			if (true_classtype == INVALID_IMAGE) {
				//cout<<"An invalid label found at Category"<<iCategory<<""<<labelName[iImage]<<endl;
				continue;
			}
			else {
				newGUID<<"images/" + imageName[iImage] + "\"" + labelName[iImage].substr(0, labelName[iImage].length() - 10)<<endl;
				validFlag += 1;
			}

		}

		cout<<"Get "<<validFlag<<" valid image at "<<iCategory<<" Class"<<endl;
		/* release the data */
		delete[] labelName;
		delete[] imageName;
		newGUID.close();
	}

	return 0;
}

