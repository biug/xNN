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

	//generateEmbeddings("E:\\pas.token", "E:\\pas.embeddings", 50);
	//RandomGenerator<float> *generator = new NormalGenerator<float>(static_cast<float>(0.0), static_cast<float>(0.1));
	//ParserNet<float, Cubic, PartialCubic, Softmax, PartialSoftmax, AdaGradUpdator> parser(50, { {18 * 50, 18 * 50, 12 * 50}, {200}, {69} }, "E:\\newEmbeddings", generator);
	//parser.train({ "E:\\batch1" }, 300, static_cast<float>(ADAGRAD_THRESHOLD));
	//parser.test("E:\\batch1");

	TwoStackAction action;
	action.loadActions("E:\\pas.conll08.small");
	//std::cout << action.Words << std::endl;
	//std::cout << action.POSes << std::endl;
	//std::cout << action.RawLabels << std::endl;
	//std::cout << "max action is " << action.MAX_ACTION << std::endl;
	std::ifstream ifs("E:\\pas.conll08.small");
	DepGraph graph;
	TwoStackState state;
	int id = 0;
	while (ifs >> graph) {
		graph.setLabels(action.Labels, action.VecLabelMap);
		if (!action.extractOracle(state, graph)) std::cout << "error" << std::endl;
		state.clear();
	}
}
