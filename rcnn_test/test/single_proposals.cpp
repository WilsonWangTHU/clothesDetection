#include <stdio.h>
#include "CaffeCls.hpp"
#include "imagehelper.hpp"
#include "iostream"
#include "fstream"
#include <string>
#include "readMapping.hpp"
#include "dbFeature.hpp"
#include "time.h"
#include "dirent.h"

using namespace std;
using namespace sdktest;


int main(int argc, char** argv) {
	
	printf("Processing starts\n");

	if (argc <= 1) printf("Error, must have the path for the \n");
	string IMAGE_DIR = string(argv[1]);
    //std::size_t found = IMAGE_DIR.find_last_of("/");
    //string OUTPUT_DIR = IMAGE_DIR.substr(0, found + 1) + "temp_proposal";
    string OUTPUT_DIR = "/home/twwang/temp_proposal";

	FastRCNNCls* fastRCNN = new FastRCNNCls; /* the proposal initializes */
	fastRCNN->init_box_model("../models/People/propose.model");
	Mat image; 								 /* the temp mat to store every image */
    cout<<"Eage Boxes model loaded"<<endl;

    /* the boxes[j][k] is the image with the j_proposal, and k_th coodinates */
	vector<vector<float> > boxes;   											 	

    cout<<"Loading image at "<<IMAGE_DIR<<endl;
    image = imread(IMAGE_DIR);
    
    /* set the boxes for the image */
    fastRCNN->get_boxes(image, boxes);

    image.release();

    /* write the binary files! */
    int nPropNum = boxes.size();
    int nBoxSize = boxes[0].size(); /* nBoxSize = 4 */

    cout<<"Processing the boxes"<<endl;
    dbFeature<float> proposals;
    proposals.set_size(nPropNum, nBoxSize);
    float* pp = proposals.get_data_pointer(); 
    for(int j = 0 ;j < nPropNum; j++) {
        memcpy(pp, boxes[j].data(), sizeof(float) * nBoxSize);
        pp += nBoxSize;
    }
    cout<<"Saving to file "<<OUTPUT_DIR<<endl;
    proposals.save_to_file((OUTPUT_DIR).data());

	delete fastRCNN;
	return 0;
}
