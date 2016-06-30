/*
 * main.cpp
 *
 *  Created on: 2016Äê6ÔÂ7ÈÕ
 *      Author: Administrator
 */
#include <iostream>

#include "loss.hpp"
#include "activation.hpp"
#include "parser_net.hpp"

#pragma comment(lib, "lib\\libopenblas.lib")

int main(int argc, char * argv[]) {
	ParserNet<float, Cubic, PartialCubic, Softmax, PartialSoftmax> parser(50, { {18 * 50, 18 * 50, 12 * 50}, {50 * 200}, {200 * 5} }, "embeddings");
	parser.train({ "batches" }, 20, (float)1e-5);
}
