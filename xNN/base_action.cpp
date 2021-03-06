
#include "graph.h"
#include "base_action.h"

#include <fstream>

using std::ifstream;

BaseAction::BaseAction() : LabelCount(0) {}
BaseAction::BaseAction(const BaseAction & actions) : BaseAction() {}
BaseAction::~BaseAction() = default;

void BaseAction::loadLabels() {
	VecLabelMap.clear();
	LabelCount = static_cast<int>(Labels.count());

	for (int i = 0; i < Labels.count(); ++i) {
		const string & label = Labels[i];
		if (IS_LEFT_LABEL(label)) {
			RawLabels.add(DECODE_LEFT_LABEL(label));
			VecLabelMap.push_back(ENCODE_LABEL_ID(RawLabels.code(DECODE_LEFT_LABEL(label)), 0));
		}
		else if (IS_RIGHT_LABEL(label)) {
			RawLabels.add(DECODE_RIGHT_LABEL(label));
			VecLabelMap.push_back(ENCODE_LABEL_ID(0, RawLabels.code(DECODE_RIGHT_LABEL(label))));
		}
		else if (IS_TWOWAY_LABEL(label)) {
			RawLabels.add(DECODE_TWOWAY_LEFT_LABEL(label));
			RawLabels.add(DECODE_TWOWAY_RIGHT_LABEL(label));
			VecLabelMap.push_back(ENCODE_LABEL_ID(RawLabels.code(DECODE_TWOWAY_LEFT_LABEL(label)), RawLabels.code(DECODE_TWOWAY_RIGHT_LABEL(label))));
		}
		else {
			std::cout << "fuck label " << label << std::endl;
		}
	}
	std::cout << "LabelCount = " << LabelCount << std::endl;
	std::cout << "Labels.count() = " << Labels.count() << std::endl;
	std::cout << "VecLabelMap.size() = " << VecLabelMap.size() << std::endl;
}

void BaseAction::loadTokens(const std::string & file) {
	ifstream ifs(file);
	DepGraph graph;
	for (int i = 0, n = Labels.count(); i < n; ++i) {
		Labels[i] = ENCODE_LEFT_LABEL(Labels[i]);
	}
	while (ifs >> graph) {
		for (const auto & node : graph) {
			Words.add(node.m_sWord);
			POSes.add(node.m_sPOSTag);
			for (const auto & arc : node.m_vecRightArcs) {
				Labels.add(arc.second);
			}
		}
	}
	loadLabels();
	ifs.close();
}