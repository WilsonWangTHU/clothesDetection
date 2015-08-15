/*
 * @Brief:
 * In this function we generate the detection proposals by dataset.
 * The dataset should have the directory tree like: /dataset/images,
 * and /dataset/proposals.
 * @Input:
 * ./CCPCFD_proposals.bin dataset_directory
 */

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

	if (argc <= 1) printf("Error, must have the root path\n");
	string ROOT_DIR = string(argv[1]);
	string IMAGE_DIR = ROOT_DIR;
	string PPS_DIR = ROOT_DIR + "/.." + "/proposals";

	/* reading the mapping files */
    DIR* dir;
    struct dirent* ent;
    int itemNumber = 0;
    int head = 0;
	string* imageName = new string[3000];
    if ((dir = opendir(ROOT_DIR.c_str())) != NULL) {
        /* print all the files and directories within directory */
        while ((ent = readdir (dir)) != NULL) {
            if (head == 0 || head == 1) {
                head ++;
                continue;
            }
            imageName[itemNumber] = ent->d_name;
            itemNumber ++;
        }
        closedir (dir);
    } else {
        /* could not open directory */
        perror ("");
        return EXIT_FAILURE;
    }

	printf("The path of the images successfully extracted\n");

	FastRCNNCls* fastRCNN = new FastRCNNCls; /* the proposal initializes */
	fastRCNN->init_box_model("../models/People/propose.model");
	Mat image; 								 /* the temp mat to store every image */
	vector<vector<vector<float> > > boxes(itemNumber);   /* the boxes[i][j][k] is the i_th image, 
											 	 the j_proposal, and k_th coodinates */

    cout<<"Eage Boxes model loaded"<<endl;
	for (int iImage = 0; iImage < itemNumber; iImage ++ ) { /* processing every image */

        cout<<"Loading image at "<<IMAGE_DIR + "/" + imageName[iImage]<<endl;
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
        cout<<"Saving to file "<<PPS_DIR + imageName[i]<<endl;
		proposals.save_to_text_file((PPS_DIR + '/' + imageName[i]).data());
	}

	delete[] imageName;
	delete fastRCNN;
	return 0;
}
