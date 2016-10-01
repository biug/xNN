#include <random>
#include <string>
#include <fstream>

#include "graph.h"
#include "data_generator.h"

void generateEmbeddings(const std::string & file, const std::string & embeddings_file, int vec_len) {
	std::default_random_engine dre;
	std::normal_distribution<float> dis(static_cast<float>(0.0), static_cast<float>(0.01));
	std::ifstream ifs(file);
	DepGraph graph;
	unordered_map<string, int> words, poses, labels;
	for (const auto & token : g_vecSpecialTokens) {
		words.insert({ token, words.size() });
		poses.insert({ token, poses.size() });
		labels.insert({ token, labels.size() });
	}
	while (ifs >> graph) {
		for (const auto & node : graph) {
			if (words.find(node.m_sWord) == words.end()) {
				words.insert({ node.m_sWord, words.size() });
			}
			if (poses.find(node.m_sPOSTag) == poses.end()) {
				poses.insert({ node.m_sPOSTag, poses.size() });
			}
			for (const auto & arc : node.m_vecRightArcs) {
				if (IS_LEFT_LABEL(arc.second)) {
					string label = DECODE_LEFT_LABEL(arc.second);
					if (labels.find(label) == labels.end()) {
						labels.insert({ label, labels.size() });
					}
				}
				else if (IS_RIGHT_LABEL(arc.second)) {
					string label = DECODE_RIGHT_LABEL(arc.second);
					if (labels.find(label) == labels.end()) {
						labels.insert({ label, labels.size() });
					}
				}
				else {
					string label = DECODE_TWOWAY_LEFT_LABEL(arc.second);
					if (labels.find(label) == labels.end()) {
						labels.insert({ label, labels.size() });
					}
					label = DECODE_TWOWAY_RIGHT_LABEL(arc.second);
					if (labels.find(label) == labels.end()) {
						labels.insert({ label, labels.size() });
					}
				}
			}
		}
	}
	ifs.close();
	vector<string> vecWords(words.size()), vecPOSes(poses.size()), vecLabels(labels.size());
	for (const auto & word : words) {
		vecWords[word.second] = word.first;
	}
	for (const auto & pos : poses) {
		vecPOSes[pos.second] = pos.first;
	}
	for (const auto & label : labels) {
		vecLabels[label.second] = label.first;
	}
	std::ofstream ofs(embeddings_file);
	ofs << vec_len << std::endl;
	ofs << words.size() << std::endl;
	for (const auto & word : vecWords) {
		ofs << word << std::endl;
		for (int j = 0; j < vec_len; ++j) {
			ofs << dis(dre) << ' ';
		}
		ofs << std::endl;
	}
	ofs << poses.size() << std::endl;
	for (const auto & pos : vecPOSes) {
		ofs << pos << std::endl;
		for (int j = 0; j < vec_len; ++j) {
			ofs << dis(dre) << ' ';
		}
		ofs << std::endl;
	}
	ofs << labels.size() << std::endl;
	for (const auto & label : vecLabels) {
		ofs << label << std::endl;
		for (int j = 0; j < vec_len; ++j) {
			ofs << dis(dre) << ' ';
		}
		ofs << std::endl;
	}
}