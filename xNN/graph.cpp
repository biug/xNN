#include "graph.h"

#include <set>
#include <map>
#include <tuple>
#include <sstream>
#include <algorithm>

using std::set;
using std::map;
using std::pair;
using std::tuple;
using std::istringstream;

CoNLL08DepNode::CoNLL08DepNode() : m_sWord(""), m_sPOSTag(""), m_nTreeHead(-1) {}

CoNLL08DepNode::~CoNLL08DepNode() = default;

CoNLL08DepNode::CoNLL08DepNode(const CoNLL08DepNode & node) :
	m_sWord(node.m_sWord), m_sPOSTag(node.m_sPOSTag), m_nTreeHead(node.m_nTreeHead),
	m_vecRightArcs(node.m_vecRightArcs), m_vecRightLabels(node.m_vecRightLabels) {}

bool operator==(const CoNLL08DepNode & n1, const CoNLL08DepNode & n2) {
	return n1.m_sWord == n2.m_sWord && n1.m_sPOSTag == n2.m_sPOSTag && n1.m_nTreeHead == n2.m_nTreeHead && n1.m_vecRightArcs == n2.m_vecRightArcs;
}

bool operator!=(const CoNLL08DepNode & n1, const CoNLL08DepNode & n2) {
	return !(n1 == n2);
}

CoNLL08DepGraph::CoNLL08DepGraph() = default;

CoNLL08DepGraph::~CoNLL08DepGraph() = default;

CoNLL08DepGraph::CoNLL08DepGraph(const CoNLL08DepGraph & graph) : m_vecNodes(graph.m_vecNodes) {}

// add crossing edge
CoNLL08DepGraph & CoNLL08DepGraph::operator=(const CoNLL08DepGraph & g) {
	m_vecNodes = g.m_vecNodes;
	return *this;
}

void CoNLL08DepGraph::setLabels(const Token & labels, const vector<int> & vecLabels) {
	for (auto && node : m_vecNodes) {
		node.m_vecRightLabels.clear();
		for (const auto & arc : node.m_vecRightArcs) {
			const int & label = labels.code(arc.second);
			node.m_vecRightLabels.push_back(
				CoNLL08DepNode::RightLabel(label, { LEFT_LABEL_ID(vecLabels[label]), RIGHT_LABEL_ID(vecLabels[label]) }));
		}
	}
}

bool CoNLL08DepGraph::checkArc(const CoNLL08DepGraph & g) {
	set<string> arcs1, arcs2;
	for (const auto & node : m_vecNodes) {
		for (const auto & arc : node.m_vecRightArcs) {
			if (IS_LEFT_LABEL(arc.second)) {
				arcs1.insert(m_vecNodes[arc.first].m_sWord + DECODE_LEFT_LABEL(arc.second) + node.m_sWord);
			}
			else if (IS_RIGHT_LABEL(arc.second)) {
				arcs1.insert(node.m_sWord + DECODE_RIGHT_LABEL(arc.second) + m_vecNodes[arc.first].m_sWord);
			}
			else {
				arcs1.insert(m_vecNodes[arc.first].m_sWord + DECODE_TWOWAY_LEFT_LABEL(arc.second) + node.m_sWord);
				arcs1.insert(node.m_sWord + DECODE_TWOWAY_RIGHT_LABEL(arc.second) + m_vecNodes[arc.first].m_sWord);
			}
		}
	}
	for (const auto & node : g.m_vecNodes) {
		for (const auto & arc : node.m_vecRightArcs) {
			if (IS_LEFT_LABEL(arc.second)) {
				arcs2.insert(g.m_vecNodes[arc.first].m_sWord + DECODE_LEFT_LABEL(arc.second) + node.m_sWord);
			}
			else if (IS_RIGHT_LABEL(arc.second)) {
				arcs2.insert(node.m_sWord + DECODE_RIGHT_LABEL(arc.second) + g.m_vecNodes[arc.first].m_sWord);
			}
			else {
				arcs2.insert(g.m_vecNodes[arc.first].m_sWord + DECODE_TWOWAY_LEFT_LABEL(arc.second) + node.m_sWord);
				arcs2.insert(node.m_sWord + DECODE_TWOWAY_RIGHT_LABEL(arc.second) + g.m_vecNodes[arc.first].m_sWord);
			}
		}
	}
	bool equal = true;
	for (const auto arc : arcs1) {
		if (arcs2.find(arc) == arcs2.end()) {
			equal = false;
		}
	}
	for (const auto arc : arcs2) {
		if (arcs1.find(arc) == arcs1.end()) {
			equal = false;
		}
	}
	return equal;
}

// equal
bool operator==(const CoNLL08DepGraph & g1, const CoNLL08DepGraph & g2) {
	if (g1.size() != g2.size()) {
		return false;
	}
	for (int i = 0, n = g1.size(); i < n; ++i) {
		if (g1[i] != g2[i]) {
			return false;
		}
	}
	return true;
}

bool operator!=(const CoNLL08DepGraph & g1, const CoNLL08DepGraph & g2) {
	return !(g1 == g2);
}

istream & operator >> (istream & is, CoNLL08DepGraph & graph) {
	string line, token;
	vector<int> heads;
	typedef tuple<int, int, string> tArcs;
	vector<tArcs> arcs;

	graph.clear();
	while (true) {
		std::getline(is, line);
		if (line.empty()) {
			break;
		}
		CoNLL08DepNode node;
		istringstream issLine(line);
		issLine >> token >> node.m_sWord >> token >> node.m_sPOSTag >> token >> token >> token >> token >> token;
		if (token == "_") {
			node.m_nTreeHead = -1;
		}
		else {
			istringstream issToken(token);
			issToken >> node.m_nTreeHead;
		}
		issLine >> token >> token;
		if (token != "_") {
			heads.push_back(graph.size());
		}
		int i = 0;
		while (issLine >> token) {
			if (token != "_") {
				if (i < heads.size() && heads[i] != graph.size()) {
					arcs.push_back(tArcs(i, graph.size(), ENCODE_RIGHT_LABEL(token)));
				}
				else if (i >= heads.size()) {
					arcs.push_back(tArcs(graph.size(), i, ENCODE_LEFT_LABEL(token)));
				}
			}
			++i;
		}
		graph.add(node);
	}
	for (auto && arc : arcs) {
		if (IS_LEFT_LABEL(std::get<2>(arc))) {
			std::get<1>(arc) = heads[std::get<1>(arc)];
		}
		else {
			std::get<0>(arc) = heads[std::get<0>(arc)];
		}
	}
	std::sort(arcs.begin(), arcs.end(), [](const tArcs & arc1, const tArcs & arc2) {
		if (std::get<0>(arc1) != std::get<0>(arc2)) {
			return std::get<0>(arc1) < std::get<0>(arc2);
		}
		else if (std::get<1>(arc1) != std::get<1>(arc2)) {
			return std::get<1>(arc1) < std::get<1>(arc2);
		}
		else {
			return IS_LEFT_LABEL(std::get<2>(arc1));
		}
	});
	for (const auto & arc : arcs) {
		auto & rightArcs = graph[std::get<0>(arc)].m_vecRightArcs;
		if (rightArcs.size() > 0 && rightArcs.back().first == std::get<1>(arc)) {
			rightArcs.back().second = ENCODE_TWOWAY_LABEL(DECODE_LEFT_LABEL(rightArcs.back().second), DECODE_RIGHT_LABEL(std::get<2>(arc)));
		}
		else {
			rightArcs.push_back(pair<int, string>(std::get<1>(arc), std::get<2>(arc)));
		}
	}
	return is;
}

ostream & operator<<(ostream & os, const CoNLL08DepGraph & graph) {
	set<int> heads;
	map<int, map<int, string>> arcs;
	int i = 0;
	for (const auto & node : graph) {
		for (const auto & arc : node.m_vecRightArcs) {
			if (IS_LEFT_LABEL(arc.second)) {
				heads.insert(arc.first);
				arcs[i][arc.first] = DECODE_LEFT_LABEL(arc.second);
			}
			else if (IS_RIGHT_LABEL(arc.second)) {
				heads.insert(i);
				arcs[arc.first][i] = DECODE_RIGHT_LABEL(arc.second);
			}
			else {
				heads.insert(arc.first);
				heads.insert(i);
				arcs[i][arc.first] = DECODE_TWOWAY_LEFT_LABEL(arc.second);
				arcs[arc.first][i] = DECODE_TWOWAY_RIGHT_LABEL(arc.second);
			}
		}
		++i;
	}
	i = 0;
	vector<int> headOrders;
	for (const auto & head : heads) {
		headOrders.push_back(head);
	}
	for (const auto & node : graph.m_vecNodes) {
		os << i + 1 << " " << node.m_sWord << " " << node.m_sWord << " " << node.m_sPOSTag << " " << node.m_sPOSTag << " _ _ s " << node.m_nTreeHead << " _ " << (heads.find(i) == heads.end() ? "_" : node.m_sWord);
		for (int j = 0; j < heads.size(); ++j) {
			os << " " << (arcs.find(i) == arcs.end() || arcs[i].find(headOrders[j]) == arcs[i].end() ? "_" : arcs[i][headOrders[j]]);
		}
		os << std::endl;
		++i;
	}
	os << std::endl;
	return os;
}
