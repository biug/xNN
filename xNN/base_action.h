#ifndef BASE_ACTION_H_
#define BASE_ACTION_H_

#include "token.h"

struct BaseAction {
	int LabelCount;
	vector<int> VecLabelMap;
	Token Words, POSes, Labels, RawLabels;

	BaseAction();
	BaseAction(const BaseAction & actions);
	virtual ~BaseAction();

	void loadLabels();
	void loadTokens(const std::string & file);
};

#endif