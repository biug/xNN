#include "twostack_action.h"

TwoStackAction::TwoStackAction() : BaseAction() {}

void TwoStackAction::loadActions(const std::string & file) {
	loadTokens(file);

	A_SH_FIRST = ACTION_END;	A_SH_END = A_SH_FIRST + LabelCount;
	A_RE_FIRST = A_SH_END;		A_RE_END = A_RE_FIRST + LabelCount;
	A_MM_FIRST = A_RE_END;		A_MM_END = A_MM_FIRST + LabelCount;
	A_RC_FIRST = A_MM_END;		A_RC_END = A_RC_FIRST + LabelCount;
	MAX_ACTION = A_RC_END;

	std::cout << "MAX_ACTION is " << MAX_ACTION << std::endl;
}

TwoStackAction::TwoStackAction(const TwoStackAction & actions) : TwoStackAction() {}

TwoStackAction::~TwoStackAction() = default;

bool TwoStackAction::extractOracle(TwoStackState & item, const DepGraph & graph) const {
	int rightNodeSeeks[MAX_SENTENCE_SIZE];
	memset(rightNodeSeeks, 0, sizeof(rightNodeSeeks));
	while (followOneAction(item, rightNodeSeeks, graph)) {
	}
		;
	return item == graph;
}

/*
action priority is : "reduce" > "arc *" > "swap" > "shift"
for example
| 0 | 1 |					| 2 | 3 | . . .
if we have arc 0 - 2, 1 - 3
we use swap, get state
| 1 |						| 2 | 3 | . . .
| 0 |
we use shift, get state
| 1 | 0 |					| 2 | 3 | . . .
we use arc-reduce, get state
| 1 |						| 2 | 3 | . . .		"0 - 2"
we use shift + reduce, get state
| 1 |							| 3 | . . .
finally we use arc-reduce, get state
|								| 3 | . . .		"1 - 3"
*/
bool TwoStackAction::followOneAction(TwoStackState & item, int(&seeks)[MAX_SENTENCE_SIZE], const DepGraph & graph,
	const pair<int, pair<int, int>> & labels) const {
	if (!item.stackEmpty()) {
		int & seek = seeks[item.stackTop()];
		const DepNode & node = graph[item.stackTop()];
		int size = node.m_vecRightArcs.size();
		while (seek < size && node.m_vecRightArcs[seek].first < item.size()) {
			++seek;
		}
		if (seek >= size) {
			switch (labels.first) {
			case -1:
				item.reduce(REDUCE);
				return true;
			default:
				item.arcReduce(labels.first, labels.second.first, labels.second.second, A_RE_FIRST + labels.first);
				return true;
			}
		}
		const auto & rightArc = node.m_vecRightArcs[seek];
		if (rightArc.first == item.size()) {
			++seek;
			return followOneAction(item, seeks, graph, node.m_vecRightLabels[seek - 1]);
		}
	}
	// swap after reduce/arc
	for (int i = item.stackBack() - 1; i >= 0; --i) {
		const DepNode &node = graph[item.stack(i)];
		const int & seek = seeks[item.stack(i)];
		if (seek < node.m_vecRightArcs.size() && node.m_vecRightArcs[seek].first == item.size()) {
			switch (labels.first) {
			case -1:
				item.mem(MEM);
				return true;
			default:
				item.arcMem(labels.first, labels.second.first, labels.second.second, A_MM_FIRST + labels.first);
				return true;
			}
		}
	}
	for (int i = item.secondStackBack(); i >= 0; --i) {
		switch (labels.first) {
		case -1:
			item.recall(RECALL);
			return true;
		default:
			item.arcRecall(labels.first, labels.second.first, labels.second.second, A_RC_FIRST + labels.first);
			return true;
		}
	}
	// shfit after swap
	if (item.size() < graph.size()) {
		switch (labels.first) {
		case -1:
			item.shift(SHIFT);
			return true;
		default:
			item.arcShift(labels.first, labels.second.first, labels.second.second, A_SH_FIRST + labels.first);
			return true;
		}
	}
	return false;
}

void TwoStackAction::doAction(TwoStackState & item, const int & action) const {
	if (action < ACTION_END) {
		switch (action) {
		case SHIFT:
			item.shift(action);
			return;
		case REDUCE:
			item.reduce(action);
			return;
		case MEM:
			item.mem(action);
			return;
		case RECALL:
			item.recall(action);
			return;
		default:
			return;
		}
	}
	else if (action < A_SH_END) {
		int label = action - A_SH_FIRST;
		int labelId = VecLabelMap[label];
		item.arcShift(label, LEFT_LABEL_ID(labelId), RIGHT_LABEL_ID(labelId), action);
	}
	else if (action < A_RE_END) {
		int label = action - A_RE_FIRST;
		int labelId = VecLabelMap[label];
		item.arcReduce(label, LEFT_LABEL_ID(labelId), RIGHT_LABEL_ID(labelId), action);
	}
	else if (action < A_MM_END) {
		int label = action - A_MM_FIRST;
		int labelId = VecLabelMap[label];
		item.arcMem(label, LEFT_LABEL_ID(labelId), RIGHT_LABEL_ID(labelId), action);
	}
	else if (action < A_RC_END) {
		int label = action - A_RC_FIRST;
		int labelId = VecLabelMap[label];
		item.arcRecall(label, LEFT_LABEL_ID(labelId), RIGHT_LABEL_ID(labelId), action);
	}
}

string TwoStackAction::printAction(const int & action) const {
	if (action < ACTION_END) {
		switch (action) {
		case SHIFT:
			return "SHIFT";
		case REDUCE:
			return "REDUCE";
		case MEM:
			return "MEM";
		case RECALL:
			return "RECALL";
		default:
			return "BAD ACTION";
		}
	}
	else if (action < A_SH_END) {
		return "ARC SHIFT";
	}
	else if (action < A_RE_END) {
		return "ARC REDUCE";
	}
	else if (action < A_MM_END) {
		return "ARC MEM";
	}
	else if (action < A_RC_END) {
		return "ARC RECALL";
	}
	else {
		return "BAD ACTION";
	}
}

bool TwoStackAction::testAction(const TwoStackState & item, const DepGraph & graph, const int & action) const {
	if (action < ACTION_END) {
		switch (action) {
		case SHIFT:
			return item.size() < graph.size() && item.canShift();
		case REDUCE:
			return !item.stackEmpty();
		case MEM:
			return item.canMem();
		case RECALL:
			return item.canRecall();
		default:
			return false;
		}
	}
	else if (action < A_SH_END) {
		return item.canArc() && item.size() < graph.size() && item.canShift();
	}
	else if (action < A_RE_END) {
		return item.canArc() && item.size() < graph.size() && !item.stackEmpty();
	}
	else if (action < A_MM_END) {
		return item.canArc() && item.size() < graph.size() && item.canMem();
	}
	else if (action < A_RC_END) {
		return item.canArc() && item.size() < graph.size() && item.canRecall();
	}
	else {
		return false;
	}
}