#include "readMapping.hpp"
#define CLOTH_LABEL_EXT .clothInfo
string clothExt = string(".clothInfo");

int guidMapping::getItemNumber(string mapFileName) {
/* open the file for the first time to get the line number */
	ifstream mapFile(mapFileName);
	if (mapFile == NULL) {
		cout<<"Oops! No map file here at"<<mapFileName<<endl;
		return -1;
	} 
	long linesCount =0;
	string line;

	while (getline(mapFile , line)) ++linesCount;

	mapFile.close();
	return linesCount;
}

bool guidMapping::getImageAndLabelName(string mapFileName, string** imageName, 
		string** labelName, int* itemNumber) {

	/* get the item number */
	*itemNumber = guidMapping::getItemNumber(mapFileName);

	if (*itemNumber == -1) {
		printf("Failure in reading the file, exit");
		return false;
	}
	ifstream mapFile(mapFileName);
	if (mapFile == NULL) {
		cout<<"Oops! No map file here at"<<mapFileName<<endl;
		return false;
	} 

	*imageName = new string[*itemNumber];
	*labelName = new string[*itemNumber];

	string line;
	size_t found1 = 0;
	size_t found2 = 0;
	int linesCount = 0;

	while (getline(mapFile , line)) {
		linesCount ++;
		/* divide the files */
		found2 = line.find("\"");
		found1 = line.find("/");
		if (found1 == std::string::npos || found2 == std::string::npos) {
			cout<<"Oops! The file has a syntex error at line "<<linesCount<<endl;
			return false;
		}

		/* get the image name and label names out */
		(*imageName)[linesCount - 1] = line.substr(found1 + 1, found2 - found1 - 1);
		(*labelName)[linesCount - 1] = line.substr(found2 + 1, line.length()) + clothExt;

	}

	mapFile.close();
	return true;
}

bool guidMapping::getImageName(string mapFileName, string** imageName, int* itemNumber) {

	/* get the item number */
	*itemNumber = guidMapping::getItemNumber(mapFileName);

	if (*itemNumber == -1) {
		printf("Failure in reading the file, exit");
		return false;
	}
	ifstream mapFile(mapFileName);
	if (mapFile == NULL) {
		cout<<"Oops! No map file here at"<<mapFileName<<endl;
		return false;
	} 

	*imageName = new string[*itemNumber];

	string line;
	size_t found1 = 0;
	size_t found2 = 0;
	int linesCount = 0;

	while (getline(mapFile , line)) {
		linesCount ++;
		/* divide the files */
		found2 = line.find("\"");
		found1 = line.find("/");
		if (found1 == std::string::npos || found2 == std::string::npos) {
			cout<<"Oops! The file has a syntex error at line "<<linesCount<<endl;
			return false;
		}
		/* get the image name and label names out */
		(*imageName)[linesCount - 1] = line.substr(found1 + 1, found2 - found1 - 1);
	}

	mapFile.close();
	return true;
}

bool guidMapping::getLabelName(string mapFileName, string** labelName, int* itemNumber) {

	/* get the item number */
	*itemNumber = guidMapping::getItemNumber(mapFileName);

	if (*itemNumber == -1) {
		printf("Failure in reading the file, exit");
		return false;
	}
	ifstream mapFile(mapFileName);
	if (mapFile == NULL) {
		cout<<"Oops! No map file here at"<<mapFileName<<endl;
		return false;
	} 

	*labelName = new string[*itemNumber];

	string line;
	size_t found1 = 0;
	size_t found2 = 0;
	int linesCount = 0;

	while (getline(mapFile , line)) {
		linesCount ++;
		/* divide the files */
		found2 = line.find("\"");
		found1 = line.find("/");
		if (found1 == std::string::npos || found2 == std::string::npos) {
			cout<<"Oops! The file has a syntex error at line "<<linesCount<<endl;
			return false;
		}

		/* get the image name and label names out */
		(*labelName)[linesCount - 1] = line.substr(found2 + 1, line.length()) + clothExt;

	}

	mapFile.close();
	return true;
}

