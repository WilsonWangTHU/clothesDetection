#include <stdio.h>
#include "CaffeCls.hpp"
#include "imagehelper.hpp"
#include "iostream"
#include "fstream"
#include <string>
#include "readMapping.hpp"
#include "dbFeature.hpp"
#include "time.h"

using namespace std;
using namespace sdktest;


int main(int argc, char** argv) {
	
	printf("Processing starts");

	if (argc <= 1) printf("Error, must have the root path\n");
	string ROOT_DIR = string(argv[1]);
	string IMAGE_DIR = ROOT_DIR + "/images";
	string LABEL_DIR = ROOT_DIR + "/Label";
	string PPS_DIR = ROOT_DIR + "/proposals";

	/* reading the mapping files */
	//string guidMappingFileName = string(argv[1]);
	printf("Reading mapping files\n");
	string guidMappingFileName = ROOT_DIR + "/GUIDMapping.txt";
	string* imageName = NULL;
	int itemNumber = 0;

	bool mappingflag = guidMapping::getImageName(guidMappingFileName,
			&imageName, &itemNumber);
	printf("The path of the images successfully extracted\n");

	if (mappingflag == false) {
		cout<<"Error during reading the file"<<endl;
		return false;
	}


	FastRCNNCls* fastRCNN = new FastRCNNCls; /* the proposal initializes */
	fastRCNN->init_box_model("../models/People/propose.model");
	Mat image; 								 /* the temp mat to store every image */
	vector<vector<vector<float> > > boxes(itemNumber);   /* the boxes[i][j][k] is the i_th image, 
											 	 the j_proposal, and k_th coodinates */

	for (int iImage = 0; iImage < itemNumber; iImage ++ ) { /* processing every image */
		image = imread(IMAGE_DIR + '/' + imageName[iImage]);

		fastRCNN->get_boxes(image, boxes[iImage]);

		if (iImage % 50 == 1) printf("Processing the %d image\n", iImage);

		image.release(); /* get ready for the next one */
	}
	
	/* write the binary files! */
	for(int i = 0;i < itemNumber; i++) {
		/* working on the ith image */

		int nPropNum = boxes[i].size();
		int nBoxSize = boxes[i][0].size(); /* nBoxSize = 4 */

		dbFeature<float> proposals;
		proposals.set_size(nPropNum, nBoxSize);
		float* pp = proposals.get_data_pointer(); 
		for(int j = 0 ;j < nPropNum; j++) {
			memcpy(pp, boxes[i][j].data(), sizeof(float) * nBoxSize);
			pp += nBoxSize;
		}
		proposals.save_to_file((PPS_DIR + "/" + imageName[i]).data());
	}

	delete[] imageName;
	delete fastRCNN;
	return 0;
}
