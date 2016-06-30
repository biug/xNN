/*
 * main.cpp
 *
 *  Created on: 2016��6��7��
 *      Author: Administrator
 */
#include <random>
#include <iostream>

#include "loss.hpp"
#include "activation.hpp"
#include "parser_net.hpp"

#pragma comment(lib, "lib\\libopenblas.lib")

int main(int argc, char * argv[]) {
	ParserNet<float, Cubic, PartialCubic, Softmax, PartialSoftmax, std::default_random_engine, std::normal_distribution> parser(50, { {18 * 50, 18 * 50, 12 * 50}, {50 * 200}, {200 * 5} }, "embeddings");
	parser.train({ "batches" }, 20, static_cast<float>(1e-5));
}
