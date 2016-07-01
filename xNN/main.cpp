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

#pragma comment(lib, "lib\\libopenblas.lib")

int main(int argc, char * argv[]) {
	RandomGenerator<float> *generator = new NormalGenerator<float>(0.0, 0.1);
	ParserNet<float, Cubic, PartialCubic, Softmax, PartialSoftmax> parser(50, { {18 * 50, 18 * 50, 12 * 50}, {200}, {5} }, "..\\x64\\Release\\embeddings", generator);
	parser.train({ "..\\x64\\Release\\batch1" }, 1000, static_cast<float>(1e-10));
}
