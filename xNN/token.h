#ifndef TOKEN_H_
#define TOKEN_H_

#include <vector>
#include <string>
#include <iostream>
#include <unordered_map>

#include "macros.h"
#include "global.h"

using std::size_t;
using std::vector;
using std::string;
using std::unordered_map;

extern const vector<string> g_vecSpecialTokens;

class Token {
	int m_nWaterMark;

	vector<string> m_vecKeys;
	unordered_map<string, int> m_mapTokens;

public:
	Token() : m_nWaterMark(0) {
		for (const auto & token : g_vecSpecialTokens) {
			add(token);
		}
	}
	Token(const Token & token) :
		m_nWaterMark(token.m_nWaterMark), m_vecKeys(token.m_vecKeys), m_mapTokens(token.m_mapTokens) {}
	~Token() = default;

	void add(const string & key) {
		if (m_mapTokens.find(key) == m_mapTokens.end()) {
			m_mapTokens[key] = m_nWaterMark++;
			m_vecKeys.push_back(key);
		}
	}

	const int code(const string & key) {
		add(key);
		return m_mapTokens[key];
	}

	const int code(const string & key) const			{ return m_mapTokens.at(key); }

	const string & operator[](const int code) const	{ return m_vecKeys[code]; }
	
	int count() const								{ return m_nWaterMark; }

	friend std::istream & operator >> (std::istream & is, Token & token) {
		int count;
		is >> count;
		string t;
		while (count--) {
			is >> t;
			token.add(t);
		}
		return is;
	}

	friend std::ostream & operator<<(std::ostream & os, const Token & token) {
		os << token.m_vecKeys.size() << std::endl;
		for (auto && t : token.m_vecKeys) {
			os << t << std::endl;
		}
		return os;
	}
};

#endif
