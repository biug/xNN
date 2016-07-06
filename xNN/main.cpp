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

#pragma comment(lib, "lib\\libopenblas.lib")

int main(int argc, char * argv[]) {
	std::ios_base::sync_with_stdio(false);
	std::cin.tie(nullptr);
	//generateEmbeddings("E:\\token", "E:\\embeddings", 50);
	RandomGenerator<float> *generator = new NormalGenerator<float>((float)0.0, (float)0.1);
	ParserNet<float, Cubic, PartialCubic, Softmax, PartialSoftmax, AdaGradUpdator> parser(50, { {18 * 50, 18 * 50, 12 * 50}, {200}, {69} }, "E:\\embeddings", generator);
	//ParserNet<float, Cubic, PartialCubic, Softmax, PartialSoftmax, SGDUpdator> parser(50, { { 18 * 50, 18 * 50, 12 * 50 },{ 200 },{ 69 } }, "E:\\embeddings", generator);
	parser.train({ "E:\\batch1" }, 30, static_cast<float>(ADAGRAD_THRESHOLD));
}
