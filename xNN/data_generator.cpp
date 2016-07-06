#include <random>
#include <string>
#include <fstream>

#include "data_generator.h"

void generateEmbeddings(const std::string & token_file, const std::string & embeddings_file, int vec_len) {
	int word_num, pos_num, label_num;
	std::string word, pos, label;
	std::default_random_engine dre;
	std::normal_distribution<float> dis(static_cast<float>(0.0), static_cast<float>(0.01));
	std::ifstream ifs(token_file);
	std::ofstream ofs(embeddings_file);
	ofs << vec_len << std::endl;
	ifs >> word_num;
	ofs << word_num << std::endl;
	for (int i = 0; i < word_num; ++i) {
		ifs >> word;
		ofs << word << std::endl;
		for (int j = 0; j < vec_len; ++j) {
			ofs << dis(dre) << ' ';
		}
		ofs << std::endl;
	}
	ifs >> pos_num;
	ofs << pos_num << std::endl;
	for (int i = 0; i < pos_num; ++i) {
		ifs >> pos;
		ofs << pos << std::endl;
		for (int j = 0; j < vec_len; ++j) {
			ofs << dis(dre) << ' ';
		}
		ofs << std::endl;
	}
	ifs >> label_num;
	ofs << label_num << std::endl;
	for (int i = 0; i < label_num; ++i) {
		ifs >> label;
		ofs << label << std::endl;
		for (int j = 0; j < vec_len; ++j) {
			ofs << dis(dre) << ' ';
		}
		ofs << std::endl;
	}
}