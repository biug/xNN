#ifndef TWOSTACK_STATE_H_
#define TWOSTACK_STATE_H_

#include "base_state.h"

class TwoStackState : public BaseState {
	bool m_bCanMem;
	int m_nSecondStackBack;
	int m_lSecondStack[MAX_SENTENCE_SIZE];

public:
	TwoStackState();
	~TwoStackState();

	const int & secondStackTop() const { return m_lSecondStack[m_nSecondStackBack]; }
	const int & secondStackBack() const { return m_nSecondStackBack; }

	bool canMem() const { return m_nStackBack > 0 && m_bCanMem; }
	bool canRecall() const { return m_nSecondStackBack >= 0; }
	bool canArc() const { return m_nStackBack != -1 && (m_vecRightNodes[stackTop()].empty() || ((m_vecRightNodes[stackTop()].back().head != m_nNextWord))); }
	bool canShift() const { return m_nSecondStackBack == -1; }

	void shift(const int & action);
	void reduce(const int & action);
	void mem(const int & action);
	void recall(const int & action);
	void arc(const int & label, const int & leftLabel, const int & rightLabel, const int & action);
	void arcShift(const int & label, const int & leftLabel, const int & rightLabel, const int & action);
	void arcReduce(const int & label, const int & leftLabel, const int & rightLabel, const int & action);
	void arcMem(const int & label, const int & leftLabel, const int & rightLabel, const int & action);
	void arcRecall(const int & label, const int & leftLabel, const int & rightLabel, const int & action);

	void clear();
	void clearNext();
	void print(const BaseAction * action, const DepGraph & graph) const override;

	bool operator==(const TwoStackState & item) const;
	bool operator==(const DepGraph & graph) const;

	TwoStackState & operator=(const TwoStackState & i);
};

#endif