/*
 * This file is used to exclude the invalid label files in the 
 * clothes data set
 *
 * Written by Tingwu Wang
 * 15.7.2015
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

int main() {

	string ROOT_DIR = "/media/Elements/twwang/fast-rcnn/data/clothesDataset/test";

	/* read all the GUIDMapping files in the dataset */
	for (int iCategory = 1; iCategory <= 26; iCategory ++) {
		int validFlag = 0;
		/* initialize the parameters */
		int nItem = 0;
		string* labelName = NULL;
		string* imageName = NULL;
		ofstream newGUID;
		newGUID.open(ROOT_DIR + "/" + to_string(iCategory) + "/newGUIDMapping.txt");

		/* read the original GUIDMapping files */
		bool mappingflag = guidMapping::getImageAndLabelName(ROOT_DIR + "/" + 
				to_string(iCategory) + "/GUIDMapping.txt", &imageName, &labelName, &nItem);
		if (mappingflag == false) cerr<<"Fail reading the map files"<<endl;

		/* take out the invalid files in the GUIDMapping files */
		for (int iImage = 0; iImage < nItem; iImage ++ ){

			/* read the results */
			int true_classtype = 0;
			float true_gt_box[4] = {0, 0, 0, 0}; 
			bool clothflag = clothReader::readClothClass(
					ROOT_DIR + "/" + to_string(iCategory) + "/Label/" + labelName[iImage],
					&true_classtype, true_gt_box);
			if (clothflag == false) cerr<<"Error reading the map files"<<endl;
			if (true_classtype == INVALID_IMAGE) {
				cout<<"An invalid label found at Category "<<iCategory<<" "<<labelName[iImage]<<endl;
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

