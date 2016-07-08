#ifndef GRAPH_H_
#define GRAPH_H_

#include <vector>
#include <string>
#include <cstring>
#include <fstream>
#include <unordered_map>

#include "token.h"

using std::pair;
using std::size_t;
using std::vector;
using std::string;
using std::istream;
using std::ostream;
using std::ifstream;

struct CoNLL08DepNode {
	typedef	pair<int, string>			RightArc;
	typedef pair<int, pair<int, int>>	RightLabel;
	
	string m_sWord;
	string m_sPOSTag;
	int m_nTreeHead;

	vector<RightArc> m_vecRightArcs;
	vector<RightLabel> m_vecRightLabels;

	CoNLL08DepNode();
	~CoNLL08DepNode();
	CoNLL08DepNode(const CoNLL08DepNode & node);
};

class CoNLL08DepGraph {
	typedef	vector<CoNLL08DepNode>::iterator		iterator;
	typedef vector<CoNLL08DepNode>::const_iterator	const_iterator;
private:
	std::vector<CoNLL08DepNode> m_vecNodes;

public:
	CoNLL08DepGraph();
	~CoNLL08DepGraph();
	CoNLL08DepGraph(const CoNLL08DepGraph & graph);

	void clear()											{ m_vecNodes.clear(); }
	const int size() const									{ return m_vecNodes.size(); }
	void add(const CoNLL08DepNode & node)					{ m_vecNodes.push_back(node); }
	CoNLL08DepNode & back()									{ return m_vecNodes.back(); }
	CoNLL08DepNode & operator[](const int & i)				{ return m_vecNodes[i]; }
	const CoNLL08DepNode & operator[](const int & i) const	{ return m_vecNodes[i]; }

	iterator begin()										{ return m_vecNodes.begin(); }
	iterator end()											{ return m_vecNodes.end(); }
	const_iterator begin() const							{ return m_vecNodes.cbegin(); }
	const_iterator end() const								{ return m_vecNodes.cend(); }

	CoNLL08DepGraph & operator=(const CoNLL08DepGraph & g);
	void setLabels(const Token & labels, const vector<int> & vecLabels);

	bool checkArc(const CoNLL08DepGraph & g);

	friend bool operator==(const CoNLL08DepGraph & g1, const CoNLL08DepGraph & g2);
	friend bool operator!=(const CoNLL08DepGraph & g1, const CoNLL08DepGraph & g2);
	friend std::istream & operator >> (istream & is, CoNLL08DepGraph & graph);
	friend std::ostream & operator<<(ostream & os, const CoNLL08DepGraph & graph);
};

bool operator==(const CoNLL08DepGraph & g1, const CoNLL08DepGraph & g2);
bool operator!=(const CoNLL08DepGraph & g1, const CoNLL08DepGraph & g2);
std::istream & operator >> (istream & is, CoNLL08DepGraph & graph);
std::ostream & operator<<(ostream & os, const CoNLL08DepGraph & graph);

typedef	CoNLL08DepNode	DepNode;
typedef	CoNLL08DepGraph	DepGraph;

#endif