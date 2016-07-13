#ifndef TWOSTACK_ACTION_H_
#define TWOSTACK_ACTION_H_

#include "base_action.h"
#include "twostack_state.h"

#include <vector>

using std::vector;
using std::string;

struct TwoStackAction : public BaseAction {

	enum Action {
		SHIFT,
		REDUCE,
		MEM,
		RECALL,
		ACTION_END,
	};

	int A_SH_FIRST, A_SH_END;
	int A_RE_FIRST, A_RE_END;
	int A_MM_FIRST, A_MM_END;
	int A_RC_FIRST, A_RC_END;
	int MAX_ACTION;

	TwoStackAction();
	TwoStackAction(const TwoStackAction & actions);
	~TwoStackAction();

	void loadActions(const std::string & file);
	void doAction(TwoStackState & item, const int & action) const;
	std::string printAction(const int & action) const;
	bool extractOracle(TwoStackState & item, const DepGraph & graph) const;
	bool followOneAction(TwoStackState & item, int(&seeks)[MAX_SENTENCE_SIZE], const DepGraph & graph,
		const pair<int, pair<int, int>> & labels = { -1,{ 0, 0 } }) const;
	bool testAction(const TwoStackState & item, const DepGraph & graph, const int & action) const;
};

#endif