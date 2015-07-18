/* This file is the code that: 
 * 		1. read the precomputed proposals
 * 		2. generate the top5 results, box positions and class id
 * 		3. save the files into a boxes
 * */


#include <stdio.h>
#include "CaffeCls.hpp"
#include "imagehelper.hpp"
#include "readMapping.hpp"
#include "dbFeature.hpp"

using namespace std;
using namespace sdktest;
using namespace cv;

const int numCategory = 26;
const int threeNumCategory = 26;
const int fontFace = FONT_HERSHEY_SIMPLEX;
const float fontScale = 0.4;
const int thickness = 1;
const int showResult = 5;

int main(int argc, char** argv) {


	/* set the directories of the files */
	string testDataDir = "/media/Elements/twwang/fast-rcnn/data/clothesDataset/test";

	/* setting the number of class, containing the background */
	vector<int> tgt_cls(threeNumCategory);
	for (int i = 0; i < threeNumCategory; i++) {
		tgt_cls[i] = i + 1;
	}

	/* initial the edge detection for getting proposals */
	FastRCNNCls* fastRCNN = new FastRCNNCls;

	/* initial the caffe model */
	fastRCNN->set_model("/media/Elements/twwang/fast-rcnn/models/ClothCaffeNet/test.prototxt", 
			"/media/Elements/twwang/fast-rcnn/output/default/cloth_model/caffenet_fast_rcnn_iter_40000.caffemodel");

	for (int subCategory = 1; subCategory <= numCategory; subCategory ++) {

		/* read the mapping files */
		string subCategoryDir = testDataDir + "/" + to_string(subCategory);
		string imageSubDir = subCategoryDir + "/" + "images";
		string resultDir = subCategoryDir + "/" + "result";
		string* imageName = NULL;
		int itemNumber;

		bool mappingflag = guidMapping::getImageName(
				subCategoryDir + "/" + "newGUIDMapping.txt",
				&imageName, &itemNumber);
		if (mappingflag == false) {
			cout<<"Error reading the mapping files!"<<endl;
			return false;
		}

		/* initialize the pointers to record the data */
		dbFeature<float> floatWriters;
		dbFeature<int> intWriters;
		floatWriters.set_size(itemNumber,  showResult * (4 + 1)); /* 5 proposals, 4 coords
																	 + 1 confidence */
		intWriters.set_size(itemNumber, showResult * 1); /* 1(class) */
		float* floatPointer = floatWriters.get_data_pointer(); 
		int* intPointer = intWriters.get_data_pointer(); 

		/* processing each image */
		for (int nImage = 0; nImage< itemNumber; nImage ++) {

			float beginTime = clock();
			cout<<"Now processing the "<<nImage<<" in the "<<subCategory<<" th dir"<<endl;

			/* read the image */
			Mat img = imread(imageSubDir + "/" + imageName[nImage]);
            cout<<"Lets see the img name"<<endl;
            cout<<imageSubDir + "/" + imageName[nImage]<<endl;
            imwrite("/home/twwang/test.jpg",
					img);

            int debug;
            cin>>debug;
            if (img.empty()) return 0;
			string proposalFileName = subCategoryDir + "/proposals/" + imageName[nImage];
			
			/* read the proposals and set it in the fast RCNN */
			dbFeature<float> proposals;
			/* sorry i made a mistake, the 1 th sub file is still text file. Oops...*/
			if (subCategory == 1) proposals.read_from_text_file(proposalFileName.c_str());
			else proposals.read_from_file(proposalFileName.c_str());
            int debug2;
            cout<<"The size is "<<proposals.get_number()<<endl;
            cout<<"Continue? y/n"<<endl;
            cin>>debug2;

			fastRCNN->set_boxes(proposals.get_number(), proposals.get_dim(), 
					proposals.get_data_pointer());

			/* forward into the caffe net */
			vector<PropBox> randPboxes = fastRCNN->detect(img, tgt_cls);

			/* sort the results according to their confidence */
			vector<PropBox> pboxes(randPboxes.size());
            cout<<"The size is "<<randPboxes.size()<<endl;
            cout<<"Continue? y/n"<<endl;
            cin>>debug;
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

			float endTime = clock();
			cout << "Time used on this image: " << 
				(endTime - beginTime)/CLOCKS_PER_SEC << 's' << endl;

			/* write the results back into the output files */
			for (int iProposal = 0; iProposal < showResult; iProposal ++) {
                int debug;
                cout<<"The debug infomation is like this"<<endl;
                cout<<"The ith proposals of image "<<imageName[nImage]<< " has "<<endl;
                cout<<"     "<<pboxes.size()<<endl;
                if (pboxes.size() != 0) {
                    cout<<"     "<<pboxes[iProposal].confidence<<endl;
                    cout<<"     "<<pboxes[iProposal].cls_id<<endl;
                }
                cout<<"Continue? y/n"<<endl;
                cin>>debug;


				if ((unsigned int)iProposal >= pboxes.size()) { /* no more data */
					floatPointer[0] = -1;
					floatPointer[1] = -1;
					floatPointer[2] = -1;
					floatPointer[3] = -1;
					floatPointer[4] = -1;
					intPointer[0] = -1;
					floatPointer = floatPointer + 5;
					intPointer = intPointer + 1;
				} else{
					floatPointer[0] = pboxes[iProposal].x1;
					floatPointer[1] = pboxes[iProposal].y1;
					floatPointer[2] = pboxes[iProposal].x2;
					floatPointer[3] = pboxes[iProposal].y2;
					floatPointer[4] = pboxes[iProposal].confidence;
					intPointer[0] = pboxes[iProposal].cls_id;
					floatPointer = floatPointer + 5;
					intPointer = intPointer + 1;
				}
			}

			proposals.clear();
			img.release();

		}


		/* save the results and release what is necessary */
		intWriters.save_to_file((resultDir + "/" + "intResults").data());
		floatWriters.save_to_file((resultDir + "/" + "floatResults").data());
		
		intWriters.clear();
		floatWriters.clear();
		delete[] imageName;
	}


	//delete helper;
	delete fastRCNN;
	return 0;
}
