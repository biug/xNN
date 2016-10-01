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

typedef ParserNet<float, Cubic, PartialCubic, Softmax, PartialSoftmax, AdaGradUpdator> Parser;

void trainData(const string & filename, const string & testfile, const string& embeddingname, const string & modelname) {
	RandomGenerator<float> *generator = new NormalGenerator<float>(static_cast<float>(0.0), static_cast<float>(0.1));
	Parser parser(
		EMBEDDING_LEN,
		{ { 26 * EMBEDDING_LEN, 26 * EMBEDDING_LEN, 20 * EMBEDDING_LEN },{ HIDDEN_WIDTH },{ LOSS_WIDTH } },
		embeddingname,
		generator
	);
	parser.train(filename, testfile, modelname, embeddingname + ".new", 50, static_cast<float>(ADAGRAD_THRESHOLD));
	delete generator;
}

void parseData(const string & trainname, const string & filename, const string & embeddingname, const string & modelname, const string & outputname) {
	RandomGenerator<float> *generator = new NormalGenerator<float>(static_cast<float>(0.0), static_cast<float>(0.1));
	Parser parser(
		EMBEDDING_LEN,
		{ { 26 * EMBEDDING_LEN, 26 * EMBEDDING_LEN, 20 * EMBEDDING_LEN },{ HIDDEN_WIDTH },{ LOSS_WIDTH } },
		embeddingname,
		generator
	);
	parser.parse(trainname, filename, outputname, modelname);
	delete generator;
}

int main(int argc, char * argv[]) {
	std::ios_base::sync_with_stdio(false);
	std::cin.tie(nullptr);

	if (strcmp(argv[1], "gen") == 0) {
		TwoStackAction action;
		action.loadActions(argv[2]);
		generateEmbeddings(argv[2], argv[3], EMBEDDING_LEN);
	}
	else if (strcmp(argv[1], "train") == 0) {
		trainData(argv[2], argv[3], argv[4], argv[5]);
	}
	else if (strcmp(argv[1], "parse") == 0) {
		parseData(argv[2], argv[3], argv[4], argv[5], argv[6]);
	}

}
