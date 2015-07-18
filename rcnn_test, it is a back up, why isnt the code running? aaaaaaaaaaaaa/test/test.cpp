/*
 *  * processing into usefull results, there are few things to considered,
 *   * 1. top 1 accuracy, top 1 bounding box 
 *    * 2. top 5 accuracy, top 5 bounding box
 *     * */
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

float get_overlaps(float* a, float* b);

int main() {

    string ROOT_DIR = "/media/Elements/twwang/fast-rcnn/data/clothesDataset/test";

    /* test for all the category in the dataset */
    for (int iCategory = 1; iCategory <= 26; iCategory ++) {
        /* initialize the parameters */
        number_test_perCls[iCategory] = 0;
        for (int i = 0; i< 26; i++) pred_perCls[iCategory][i] = 0;
        dbFeature<int> classResults;
        dbFeature<float> floatResults;
        int nItem = 0;
        string* labelName = NULL;
        string* imageName = NULL;

        /* read the detect result */
        classResults.read_from_file((ROOT_DIR + "/" + 
                    to_string(iCategory) + "/result/intResults").c_str());
        floatResults.read_from_file((ROOT_DIR + "/" + 
                    to_string(iCategory) + "/result/floatResults").c_str());
        float* fPointer = floatResults.get_data_pointer();
        int* iPointer = classResults.get_data_pointer();

        /* read the true result */
        bool mappingflag = guidMapping::getImageAndLabelName(ROOT_DIR + "/" + 
                to_string(iCategory) + "/newGUIDMapping.txt", &imageName, &labelName, &nItem);
        if (mappingflag == false) cerr<<"Fail reading the map files"<<endl;

        /* validate the number of images in the sub dir */
        if (classResults.get_number() != nItem || floatResults.get_number() != nItem)
            cerr<<"Number of images are not matched: "<<nItem<<" "<<floatResults.get_number()<<endl;

        for (int iImage = 0; iImage < nItem; iImage ++ ){

            /* read the true results */
            int true_classtype = 0;
            float true_gt_box[4] = {0, 0, 0, 0}; 
            bool clothflag = clothReader::readClothClass(
                    ROOT_DIR + "/" + to_string(iCategory) + "/Label/" + labelName[iImage],
                    &true_classtype, true_gt_box);
            if (clothflag == false) cerr<<"Error reading the map files"<<endl;
            if (true_classtype == INVALID_IMAGE) {
                cout<<"An invalid label found at Category"<<iCategory<<""<<labelName[iImage]<<endl;
                continue;
            }
            else {
                number_test ++;
                number_test_perCls[iCategory - 1] ++;
            }

            /* read the predict results */
            int pre_classtype[nPrediction];
            float pre_gt_box[nPrediction][4];
            float pre_confidence[nPrediction];

            for (int iPre = 0; iPre < nPrediction; iPre ++) {
                pre_classtype[iPre] = iPointer[0];

                pre_gt_box[iPre][0] = fPointer[0];
                pre_gt_box[iPre][1] = fPointer[1];
                pre_gt_box[iPre][2] = fPointer[2];
                pre_gt_box[iPre][3] = fPointer[3];
                pre_confidence[iPre] = fPointer[4];

                /* move the pointer */
                iPointer = iPointer + 1;
                fPointer = fPointer + 5;

            }

            /* get the 1th accuracy */
            if (get_overlaps(pre_gt_box[0], true_gt_box) > overlaps_threshold) {
                accuracy_perCls_top1_box[iCategory - 1] ++;
            }
            if (pre_classtype[0] == iCategory) {
                accuracy_perCls_top1[iCategory - 1] ++;
            }


            /* get the 5th accuracy and plot the image */
            Mat img = imread(ROOT_DIR + "/" + to_string(iCategory) + "/images/" + imageName[iImage]);
            int gotOne = 0;
            for (int iPre = 0; iPre < nPrediction; iPre ++) {
                if (pre_classtype[iPre] == -1) continue;
                if (pre_classtype[iPre] == iCategory &&
                        get_overlaps(pre_gt_box[iPre], true_gt_box) > overlaps_threshold &&
                        gotOne == 0) {
                    accuracy_perCls_top5_box[iCategory - 1] ++;
                    accuracy_perCls_top5[iCategory - 1] ++;
                    gotOne = 1;
                }
                if (pre_classtype[iPre] == iCategory && gotOne == 0) {
                    gotOne = 1;
                    accuracy_perCls_top5[iCategory - 1] ++;
                }

                string preText ="I:" + to_string(int(get_overlaps(pre_gt_box[0], true_gt_box) * 100)) + "P:" + to_string(pre_confidence[iPre]) +
                    "C:" + to_string(pre_classtype[iPre]);
                putText(img, preText,
                        Point(30, 30 * iPre + 20),
                        fontFace, fontScale, cvScalar(255 * (iPre != 0), 0, 255 * (iPre == 0)),
                        thickness, 8);
                putText(img, preText,
                        Point(350 - 200, 330- 30 * iPre),
                        fontFace, fontScale, cvScalar(255 * (iPre != 0), 0, 255 * (iPre == 0)),
                        thickness, 8);
                rectangle(img, 
                        Point(int(pre_gt_box[iPre][0]), int(pre_gt_box[iPre][1])),
                        Point(int(pre_gt_box[iPre][2]), int(pre_gt_box[iPre][3])),
                        cvScalar(255 * (iPre != 0), 0, 255 * (iPre == 0)), 2, 8, 0);
            }
            rectangle(img, 
                    Point(int(true_gt_box[0]), int(true_gt_box[1])),
                    Point(int(true_gt_box[2]), int(true_gt_box[3])),
                    cvScalar(0, 255, 0), 2, 8, 0);
            imwrite(ROOT_DIR + "/" + to_string(iCategory) + "/result/" + imageName[iImage],
                    img);
            img.release();

        }

        /* release the data */
        delete[] labelName;
        delete[] imageName;
        classResults.clear();
        floatResults.clear();

    }
    int top5 = 0;
    int top1 = 0;
    for (int i = 0; i < 26; i++) {
        cout<<"Number of test here: "<<number_test_perCls[i]<<endl;
        cout<<"Number of top1: "<<accuracy_perCls_top1[i]<<
            ", with the IoU 0.8: "<<accuracy_perCls_top1_box[i]<<endl;
        cout<<"Number of top5: "<<accuracy_perCls_top5[i]<<
            ", with the IoU 0.8: "<<accuracy_perCls_top5_box[i]<<endl;
        top1 = top1 + accuracy_perCls_top1[i];
        top5 = top5 + accuracy_perCls_top5[i];
    }
    cout<<"Overall test: "<<number_test<<endl;
    cout<<"Top1 accuracy is " <<1.0 * top1 / number_test<<endl;
    cout<<"Top5 accuracy is " <<1.0 * top5 / number_test<<endl;
    return 0;
}

float get_overlaps(float* a, float* b) {
    if (a == NULL || b == NULL) cerr<<"The boxes is not initialized!"<<endl;

    float sizeA = (a[2] - a[0]) * (a[3] - a[1]); 
    float sizeB = (b[2] - b[0]) * (b[3] - b[1]); 

    float x1 = max(a[0], b[0]);
    float x2 = min(a[2], b[2]);
    float y1 = max(a[1], b[1]);
    float y2 = min(a[3], b[3]);

    if (x2 <= x1 || y2 <= y1) return 0;
    float overlaps = (x2 - x1) * (y2 - y1);
    return overlaps / (sizeA + sizeB - overlaps);
}

