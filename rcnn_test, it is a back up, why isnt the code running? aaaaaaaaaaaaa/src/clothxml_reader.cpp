#include "clothxml_reader.hpp"
const int numCategory = 26;
const string table[numCategory] = {
	"风衣", "毛呢大衣", "羊毛衫/羊绒衫",
	"棉服/羽绒服",  "小西装/短外套",
	"西服", "夹克", "旗袍", "皮衣", "皮草",
	"婚纱", "衬衫", "T恤", "Polo衫", "开衫",
	"马甲", "男女背心及吊带", "卫衣",
	"雪纺衫", "连衣裙", "半身裙",
	"打底裤", "休闲裤", "牛仔裤", "短裤",
	"卫裤/运动裤"
};

bool clothReader::readClothClass(string labelName, int* classtype, float* position) {
	/* initial the class type and check for the position avalability */
	(*classtype) = 0;

	if (position == NULL) {
		cout<<"The position is not initialized! Allocate memory before using!"<<endl;
		return false;
	}

	/* reading the xml files */
	XMLDocument doc;
	XMLError eResult = doc.LoadFile(const_cast<char*>(labelName.c_str()));
	if (eResult != XML_SUCCESS) {
		cout<<"Error opening the xml files! "<<labelName<<endl;
		return false;
	}

	XMLNode* pRoot = doc.FirstChildElement("attributes");
	if (pRoot == nullptr) {
		cout<<"No root XMLNode available!"<<endl;
		return false;
	}
	XMLElement* pElement = pRoot->FirstChildElement("clothClass");
	if (pElement == nullptr) {
		cout<<"No cloth class here!"<<endl;
		return false;
	}

	XMLElement* locationElement = pElement->FirstChildElement("Location");
	if (locationElement == nullptr) {
		cout<<"No class location here!"<<endl;
		return false;
	}

	/* record the infomation needed, note that the image could be invalid,
	 * check that */
	const char * szAttributeText = nullptr;
	float x1, x2, y1, y2; /* the four locations */

	szAttributeText = pElement->Attribute("type");
	if (szAttributeText == nullptr || 
			strcmp(locationElement->Attribute("SourceQuality"),
				"Invalid") == 0) {
		(*classtype) = INVALID_IMAGE; /* indicate that the image is invalid*/
		return true;
	}

	/* process the 4 points of the ground truth proposals */
	locationElement->QueryFloatAttribute("left", &x1);
	locationElement->QueryFloatAttribute("right", &x2);
	locationElement->QueryFloatAttribute("bottom", &y1);
	locationElement->QueryFloatAttribute("top", &y2);

	position[0] = min(x1, x2);
	position[1] = min(y1, y2);
	position[2] = max(x1, x2);
	position[3] = max(y1, y2);


	/* looking for the matching class number */
	for (int i = 0; i < numCategory; i++) {
		if (szAttributeText == table[i]) (*classtype) = i + 1;
	}

	if ((*classtype) == -1) {
		cout<<"No class match find for "<<szAttributeText<<" , error!"<<endl;
		return false;
	}

	doc.Clear();
	return true;
}
