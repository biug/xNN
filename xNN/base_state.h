#ifndef BASE_STATE_H_
#define BASE_STATE_H_

#include "graph.h"
#include "base_action.h"

#include <vector>

using std::vector;

struct RightNode {
	int head, label;
	RightNode(int h, int l) : head(h), label(l) {}
	RightNode(const RightNode & rn) : head(rn.head), label(rn.label) {}
	~RightNode() = default;
};

class BaseState {
protected:
	int m_nNextWord;
	int m_nStackBack;

	int m_nActionBack;
	int m_lActionList[MAX_SENTENCE_SIZE << MAX_SENTENCE_BITS];

	int m_lStack[MAX_SENTENCE_SIZE];
	int m_lHeadL[MAX_SENTENCE_SIZE];	//heads for every node
	int m_lHeadLabelL[MAX_SENTENCE_SIZE];	//label for every node
	int m_lHeadLNum[MAX_SENTENCE_SIZE];
	int m_lHeadR[MAX_SENTENCE_SIZE];
	int m_lHeadLabelR[MAX_SENTENCE_SIZE];
	int m_lHeadRNum[MAX_SENTENCE_SIZE];
	int m_lPredL[MAX_SENTENCE_SIZE];		//left dependency children
	int m_lSubPredL[MAX_SENTENCE_SIZE];
	int m_lPredLabelL[MAX_SENTENCE_SIZE];
	int m_lSubPredLabelL[MAX_SENTENCE_SIZE];
	int m_lPredLNum[MAX_SENTENCE_SIZE];
	int m_lPredR[MAX_SENTENCE_SIZE];		//right dependency children
	int m_lSubPredR[MAX_SENTENCE_SIZE];
	int m_lPredLabelR[MAX_SENTENCE_SIZE];
	int m_lSubPredLabelR[MAX_SENTENCE_SIZE];
	int m_lPredRNum[MAX_SENTENCE_SIZE];

	vector<RightNode> m_vecRightNodes[MAX_SENTENCE_SIZE];

public:
	BaseState() : m_nNextWord(0), m_nStackBack(-1), m_nActionBack(-1) {}
	virtual ~BaseState() {};

	const int & action(const int & index) const				{ return m_lActionList[index]; }
	const int & lastAction() const							{ return m_lActionList[m_nActionBack]; }
	const int & stack(const int & index) const				{ return m_lStack[index]; }
	const int & leftHead(const int & index) const			{ return m_lHeadL[index]; }
	const int & rightHead(const int & index) const			{ return m_lHeadR[index]; }
	const int & leftHeadLabel(const int & index) const		{ return m_lHeadLabelL[index]; }
	const int & rightHeadLabel(const int & index) const		{ return m_lHeadLabelR[index]; }
	const int & leftPred(const int & index) const			{ return m_lPredL[index]; }
	const int & rightPred(const int & index) const			{ return m_lPredR[index]; }
	const int & leftSubPred(const int & index) const		{ return m_lSubPredL[index]; }
	const int & rightSubPred(const int & index) const		{ return m_lSubPredR[index]; }
	const int & leftPredLabel(const int & index) const		{ return m_lPredLabelL[index]; }
	const int & rightPredLabel(const int & index) const		{ return m_lPredLabelR[index]; }
	const int & leftSubPredLabel(const int & index) const	{ return m_lSubPredLabelL[index]; }
	const int & rightSubPredLabel(const int & index) const	{ return m_lSubPredLabelR[index]; }
	const int & leftHeadArity(const int & index) const		{ return m_lHeadLNum[index]; }
	const int & leftPredArity(const int & index) const		{ return m_lPredLNum[index]; }
	const int & rightHeadArity(const int & index) const		{ return m_lHeadRNum[index]; }
	const int & rightPredArity(const int & index) const		{ return m_lPredRNum[index]; }

	const int & size() const								{ return m_nNextWord; };
	const int & stackBack() const							{ return m_nStackBack; }
	const int & stackTop() const							{ return m_lStack[m_nStackBack]; }
	const int & stackSubTop() const							{ return m_lStack[m_nStackBack - 1]; }
	const int & actionBack() const							{ return m_nActionBack; }
	bool stackEmpty() const									{ return m_nStackBack == -1; }

	void arc(const int & label, const int & leftLabel, const int & rightLabel);
	void arcLeft(const int & label);
	void arcRight(const int & label);
	void generateGraph(const DepGraph & sent, DepGraph & graph, const Token & labels) const;

	virtual void print(const BaseAction * action, const DepGraph & graph) const = 0;
};

#endif
