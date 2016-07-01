#include <random>
#include <fstream>

#include "data_generator.h"

void generateEmbeddings(const std::string & embeddings_file, int vec_len, int word_num, int pos_num, int label_num) {
	std::default_random_engine dre;
	std::normal_distribution<float> dis(static_cast<float>(0.0), static_cast<float>(0.01));
	std::ofstream ofs(embeddings_file);
	ofs << vec_len << std::endl;
	ofs << word_num << std::endl;
	for (int i = 0; i < word_num; ++i) {
		ofs << i << std::endl;
		ofs << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " ";
		ofs << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " ";
		ofs << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " ";
		ofs << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " ";
		ofs << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << std::endl;
	}
	ofs << pos_num << std::endl;
	for (int i = 0; i < pos_num; ++i) {
		ofs << i << std::endl;
		ofs << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " ";
		ofs << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " ";
		ofs << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " ";
		ofs << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " ";
		ofs << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << std::endl;
	}
	ofs << label_num << std::endl;
	for (int i = 0; i < label_num; ++i) {
		ofs << i << std::endl;
		ofs << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " ";
		ofs << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " ";
		ofs << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " ";
		ofs << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " ";
		ofs << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << " " << (dis(dre)) << std::endl;
	}
}