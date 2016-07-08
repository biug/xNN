/*
 * main.cpp
 *
 *  Created on: 2016Äê6ÔÂ7ÈÕ
 *      Author: Administrator
 */
#include <random>
#include <iostream>

#include "loss.hpp"
#include "activation.hpp"
#include "parser_net.hpp"
#include "normal_generator.hpp"

#include "data_generator.h"

#include "sgd_updator.hpp"
#include "adagrad_updator.hpp"

#include "graph.h"
#include "twostack_action.h"

#pragma comment(lib, "lib\\libopenblas.lib")

int main(int argc, char * argv[]) {
	std::ios_base::sync_with_stdio(false);
	std::cin.tie(nullptr);
	const std::string dir = "D:\\cpp\\vs\\xNN\\xNN\\";

	//generateEmbeddings("pas.batchpas.token", "E:\\pas.embeddings", 50);
	RandomGenerator<float> *generator = new NormalGenerator<float>(static_cast<float>(0.0), static_cast<float>(0.1));

	ParserNet<float, Cubic, PartialCubic, Softmax, PartialSoftmax, AdaGradUpdator> parser(
		EMBEDDING_LEN,
		{ {26 * EMBEDDING_LEN, 26 * EMBEDDING_LEN, 20 * EMBEDDING_LEN }, {300}, {188} },
		dir + "pas.embeddings",
		generator
	);

	parser.train({ dir + "pas.batch" }, dir + "pas.model", dir + "pas.new_embeddings", 200, static_cast<float>(ADAGRAD_THRESHOLD));

	//parser.test(dir + "pas.batch", dir + "pas.model");
	//parser.parse(dir + "pas.conll08.small", dir + "pas.batch.out", dir + "pas.model");

	//TwoStackAction action;
	//action.loadActions("c:\\users\\zhangxun\\desktop\\xnn\\xnn\\pas.conll08.small");
	//std::cout << action.Words << std::endl;
	//std::cout << action.POSes << std::endl;
	//std::cout << action.RawLabels << std::endl;
	//std::cout << "max action is " << action.MAX_ACTION << std::endl;
	//std::ifstream ifs("c:\\users\\zhangxun\\desktop\\xnn\\xnn\\pas.conll08.small");
	//DepGraph graph;
	//TwoStackState state;
	//int id = 0;
	//while (ifs >> graph) {
	//	graph.setLabels(action.Labels, action.VecLabelMap);
	//	if (!action.extractOracle(state, graph)) std::cout << "error" << std::endl;
	//	state.clear();
	//}
}
