#include "twostack_state.h"
#include "twostack_action.h"

TwoStackState::TwoStackState() {
	clear();
}

TwoStackState::~TwoStackState() = default;

void TwoStackState::shift(const int & action) {
	m_bCanMem = true;
	m_lStack[++m_nStackBack] = m_nNextWord++;
	m_lActionList[++m_nActionBack] = action;
	clearNext();
}

void TwoStackState::reduce(const int & action) {
	--m_nStackBack;
	m_lActionList[++m_nActionBack] = action;
}

void TwoStackState::mem(const int & action) {
	m_lSecondStack[++m_nSecondStackBack] = m_lStack[m_nStackBack--];
	m_lActionList[++m_nActionBack] = action;
}

void TwoStackState::recall(const int & action) {
	m_bCanMem = false;
	m_lStack[++m_nStackBack] = m_lSecondStack[m_nSecondStackBack--];
	m_lActionList[++m_nActionBack] = action;
}

void TwoStackState::arc(const int & label, const int & leftLabel, const int & rightLabel, const int & action) {
	BaseState::arc(label, leftLabel, rightLabel);
	m_lActionList[++m_nActionBack] = action;
}

void TwoStackState::arcShift(const int & label, const int & leftLabel, const int & rightLabel, const int & action) {
	BaseState::arc(label, leftLabel, rightLabel);
	shift(0);
	m_lActionList[m_nActionBack] = action;
}


void TwoStackState::arcReduce(const int & label, const int & leftLabel, const int & rightLabel, const int & action) {
	BaseState::arc(label, leftLabel, rightLabel);
	reduce(action);
	m_lActionList[m_nActionBack] = action;
}

void TwoStackState::arcMem(const int & label, const int & leftLabel, const int & rightLabel, const int & action) {
	BaseState::arc(label, leftLabel, rightLabel);
	mem(0);
	m_lActionList[m_nActionBack] = action;
}

void TwoStackState::arcRecall(const int & label, const int & leftLabel, const int & rightLabel, const int & action) {
	BaseState::arc(label, leftLabel, rightLabel);
	recall(0);
	m_lActionList[m_nActionBack] = action;
}

vector<vector<int>> TwoStackState::features(const BaseAction * action, const DepGraph & graph) const {
	int st[3] = { -2, -2, -2 }, bf[3] = { -1, -1, -1 };
	if (m_nStackBack >= 0) {
		if (m_nStackBack >= 1) {
			if (m_nStackBack >= 2) {
				st[2] = m_lStack[m_nStackBack - 2];
			}
			st[1] = m_lStack[m_nStackBack - 1];
		}
		st[0] = m_lStack[m_nStackBack];
	}
	if (m_nNextWord < graph.size()) {
		if (m_nNextWord < graph.size() - 1) {
			if (m_nNextWord < graph.size() - 2) {
				bf[2] = m_nNextWord + 2;
			}
			bf[1] = m_nNextWord + 1;
		}
		bf[0] = m_nNextWord;
	}
	int lh[2] = { -2, -2 }, rh[2] = { -1, -1 };
	int lhl[2] = { -2, -2 }, rhl[2] = { -1, -1 };
	int lc1[2] = { -2, -2 }, rc1[2] = { -1, -1 }, lc2[2] = { -2, -2 }, rc2[2] = { -1, -1 };
	int lcl1[2] = { -2, -2 }, rcl1[2] = { -1, -1 }, lcl2[2] = { -2, -2 }, rcl2[2] = { -1, -1 };
	if (st[0] >= 0) {
		if (m_lHeadL[st[0]] >= 0) {
			lh[0] = m_lHeadL[st[0]];
			lhl[0] = m_lHeadLabelL[st[0]];
		}
		if (m_lHeadR[st[0]] >= 0) {
			rh[0] = m_lHeadR[st[0]];
			rhl[0] = m_lHeadLabelR[st[0]];
		}
		if (m_lPredL[st[0]] >= 0) {
			lc1[0] = m_lPredL[st[0]];
			lcl1[0] = m_lPredLabelL[st[0]];
		}
		if (m_lPredR[st[0]] >= 0) {
			rc1[0] = m_lPredR[st[0]];
			rcl1[0] = m_lPredLabelR[st[0]];
		}
		if (m_lSubPredL[st[0]] >= 0) {
			lc2[0] = m_lSubPredL[st[0]];
			lcl2[0] = m_lSubPredLabelL[st[0]];
		}
		if (m_lSubPredR[st[0]] >= 0) {
			rc2[0] = m_lSubPredR[st[0]];
			rcl2[0] = m_lSubPredLabelR[st[0]];
		}
	}
	if (st[1] >= 0) {
		if (m_lHeadL[st[1]] >= 0) {
			lh[1] = m_lHeadL[st[1]];
			lhl[1] = m_lHeadLabelL[st[1]];
		}
		if (m_lHeadR[st[1]] >= 0) {
			rh[1] = m_lHeadR[st[1]];
			rhl[1] = m_lHeadLabelR[st[1]];
		}
		if (m_lPredL[st[1]] >= 0) {
			lc1[1] = m_lPredL[st[1]];
			lcl1[1] = m_lPredLabelL[st[1]];
		}
		if (m_lPredR[st[1]] >= 0) {
			rc1[1] = m_lPredR[st[1]];
			rcl1[1] = m_lPredLabelR[st[1]];
		}
		if (m_lSubPredL[st[1]] >= 0) {
			lc2[1] = m_lSubPredL[st[1]];
			lcl2[1] = m_lSubPredLabelL[st[1]];
		}
		if (m_lSubPredR[st[1]] >= 0) {
			rc2[1] = m_lSubPredR[st[1]];
			rcl2[1] = m_lSubPredLabelR[st[1]];
		}
	}
	int llh[2] = { -2, -2 }, rrh[2] = { -1, -1 };
	int llhl[2] = { -2, -2 }, rrhl[2] = { -1, -1 };
	if (lh[0] >= 0 && m_lPredL[lh[0]] >= 0) {
		llh[0] = m_lPredL[lh[0]];
		llhl[0] = m_lPredLabelL[lh[0]];
	}
	if (lh[1] >= 0 && m_lPredL[lh[1]] >= 0) {
		llh[1] = m_lPredL[lh[1]];
		llhl[1] = m_lPredLabelL[lh[1]];
	}
	if (rh[0] < m_nNextWord && rh[0] >= 0 && m_lPredR[rh[0]] >= 0) {
		rrh[0] = m_lPredR[rh[0]];
		rrhl[0] = m_lPredLabelR[rh[0]];
	}
	if (rh[1] < m_nNextWord && rh[1] >= 0 && m_lPredR[rh[1]] >= 0) {
		rrh[1] = m_lPredR[rh[1]];
		rrhl[1] = m_lPredLabelR[rh[1]];
	}
	int llc[2] = { -2, -2 }, rrc[2] = { -1, -1 };
	int llcl[2] = { -2, -2 }, rrcl[2] = { -1, -1 };
	if (lc1[0] >= 0 && m_lPredL[lc1[0]] >= 0) {
		llc[0] = m_lPredL[lc1[0]];
		llcl[0] = m_lPredLabelL[lc1[0]];
	}
	if (lc1[1] >= 0 && m_lPredL[lc1[1]] >= 0) {
		llc[1] = m_lPredL[lc1[1]];
		llcl[1] = m_lPredLabelL[lc1[1]];
	}
	if (rc1[0] < m_nNextWord && rc1[0] >= 0 && m_lPredR[rc1[0]] >= 0) {
		rrc[0] = m_lPredR[rc1[0]];
		rrcl[0] = m_lPredLabelR[rc1[0]];
	}
	if (rc1[1] < m_nNextWord && rc1[1] >= 0 && m_lPredR[rc1[1]] >= 0) {
		rrc[1] = m_lPredR[rc1[1]];
		rrcl[1] = m_lPredLabelR[rc1[1]];
	}
	std::vector<int> words, poses, deps;
	words.push_back(st[0]);		words.push_back(st[1]);		words.push_back(st[2]);
	words.push_back(bf[0]);		words.push_back(bf[1]);		words.push_back(bf[2]);
	words.push_back(lh[0]);		words.push_back(lh[1]);		words.push_back(rh[0]);		words.push_back(rh[1]);
	words.push_back(lc1[0]);	words.push_back(lc1[1]);	words.push_back(rc1[0]);	words.push_back(rc1[1]);
	words.push_back(lc2[0]);	words.push_back(lc2[1]);	words.push_back(rc2[0]);	words.push_back(rc2[1]);
	words.push_back(llh[0]);	words.push_back(llh[1]);	words.push_back(rrh[0]);	words.push_back(rrh[1]);
	words.push_back(llc[0]);	words.push_back(llc[1]);	words.push_back(rrc[0]);	words.push_back(rrc[1]);
	poses = words;
	deps.push_back(lhl[0]);		deps.push_back(lhl[1]);		deps.push_back(rhl[0]);		deps.push_back(rhl[1]);
	deps.push_back(lcl1[0]);	deps.push_back(lcl1[1]);	deps.push_back(rcl1[0]);	deps.push_back(rcl1[1]);
	deps.push_back(lcl2[0]);	deps.push_back(lcl2[1]);	deps.push_back(rcl2[0]);	deps.push_back(rcl2[1]);
	deps.push_back(llhl[0]);	deps.push_back(llhl[1]);	deps.push_back(rrhl[0]);	deps.push_back(rrhl[1]);
	deps.push_back(llcl[0]);	deps.push_back(llcl[1]);	deps.push_back(rrcl[0]);	deps.push_back(rrcl[1]);
	int ii = 0;
	for (auto && word : words) {
		if (word >= 0) {
			word = action->Words.code(graph[word].m_sWord);
		}
		else {
			word += 2;
		}
	}
	for (auto && pos : poses) {
		if (pos >= 0) {
			pos = action->POSes.code(graph[pos].m_sPOSTag);
		}
		else {
			pos += 2;
		}
	}
	for (auto && label : deps) {
		if (label < 0) {
			label += 2;
		}
	}
	return{ words, poses, deps };
}

void TwoStackState::clear() {
	//reset buffer seek
	m_nNextWord = 0;
	//reset stack seek
	m_nStackBack = -1;
	m_nSecondStackBack = -1;
	m_bCanMem = false;
	//reset action
	m_nActionBack = -1;
	clearNext();
}

void TwoStackState::clearNext() {
	m_lHeadL[m_nNextWord] = -1;
	m_lHeadLabelL[m_nNextWord] = 0;
	m_lHeadLNum[m_nNextWord] = 0;
	m_lHeadR[m_nNextWord] = -1;
	m_lHeadLabelR[m_nNextWord] = 0;
	m_lHeadRNum[m_nNextWord] = 0;
	m_lPredL[m_nNextWord] = -1;
	m_lSubPredL[m_nNextWord] = -1;
	m_lPredLabelL[m_nNextWord] = 0;
	m_lSubPredLabelL[m_nNextWord] = 0;
	m_lPredLNum[m_nNextWord] = 0;
	m_lPredR[m_nNextWord] = -1;
	m_lSubPredR[m_nNextWord] = -1;
	m_lPredLabelR[m_nNextWord] = 0;
	m_lSubPredLabelR[m_nNextWord] = 0;
	m_lPredRNum[m_nNextWord] = 0;
	m_vecRightNodes[m_nNextWord].clear();
}

bool TwoStackState::operator==(const TwoStackState & item) const {
	if (m_nActionBack != item.m_nActionBack) {
		return false;
	}
	for (int i = m_nActionBack; i >= 0; --i) {
		if (m_lActionList[i] != item.m_lActionList[i]) {
			return false;
		}
	}
	return true;
}

bool TwoStackState::operator==(const DepGraph & graph) const {
	if (m_nNextWord != graph.size()) {
		return false;
	}
	for (int i = 0; i < m_nNextWord; ++i) {
		if (m_vecRightNodes[i].size() != graph[i].m_vecRightArcs.size()) {
			return false;
		}
		for (int j = 0; j < m_vecRightNodes[i].size(); ++j) {
			if (m_vecRightNodes[i][j].head != graph[i].m_vecRightArcs[j].first ||
				m_vecRightNodes[i][j].label != graph[i].m_vecRightLabels[j].first) {
				return false;
			}
		}
	}
	return true;
}

TwoStackState & TwoStackState::operator=(const TwoStackState & item) {
	m_nStackBack = item.m_nStackBack;
	m_nSecondStackBack = item.m_nSecondStackBack;
	m_nActionBack = item.m_nActionBack;
	m_nNextWord = item.m_nNextWord;
	m_bCanMem = item.m_bCanMem;

	int len = sizeof(int) * (m_nNextWord + 1);
	if (m_nStackBack >= 0) {
		memcpy(m_lStack, item.m_lStack, sizeof(int) * (m_nStackBack + 1));
	}
	if (m_nSecondStackBack >= 0) {
		memcpy(m_lSecondStack, item.m_lSecondStack, sizeof(int) * (m_nSecondStackBack + 1));
	}
	if (m_nActionBack >= 0) {
		memcpy(m_lActionList, item.m_lActionList, sizeof(int) * (m_nActionBack + 1));
	}

	memcpy(m_lHeadL, item.m_lHeadL, len);
	memcpy(m_lHeadLabelL, item.m_lHeadLabelL, len);
	memcpy(m_lHeadLNum, item.m_lHeadLNum, len);
	memcpy(m_lHeadR, item.m_lHeadR, len);
	memcpy(m_lHeadLabelR, item.m_lHeadLabelR, len);
	memcpy(m_lHeadRNum, item.m_lHeadRNum, len);
	memcpy(m_lPredL, item.m_lPredL, len);
	memcpy(m_lSubPredL, item.m_lSubPredL, len);
	memcpy(m_lPredLabelL, item.m_lPredLabelL, len);
	memcpy(m_lSubPredLabelL, item.m_lSubPredLabelL, len);
	memcpy(m_lPredLNum, item.m_lPredLNum, len);
	memcpy(m_lPredR, item.m_lPredR, len);
	memcpy(m_lSubPredR, item.m_lSubPredR, len);
	memcpy(m_lPredLabelR, item.m_lPredLabelR, len);
	memcpy(m_lSubPredLabelR, item.m_lSubPredLabelR, len);
	memcpy(m_lPredRNum, item.m_lPredRNum, len);

	return *this;
}