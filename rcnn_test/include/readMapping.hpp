#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
using namespace std;

class guidMapping
{
public:
static bool getImageAndLabelName(string mapFileName, string** imageName,
		string** labelName, int* itemNumber);
static bool getImageName(string mapFileName, string** imageName, int* itemNumber);
static bool getLabelName(string mapFileName, string** labelName, int* itemNumber);
static int getItemNumber(string mapFileName);
};
