/*
* random_generator.hpp
*
*  Created on: 2016Äê6ÔÂ5ÈÕ
*      Author: Administrator
*/

#ifndef RANDOM_GENERATOR_HPP_
#define RANDOM_GENERATOR_HPP_

#include <ctime>
#include <random>

template<typename DType, typename RandomEngine = std::default_random_engine>
class RandomGenerator {
protected:
	RandomEngine m_Engine;
public:
	RandomGenerator() : m_Engine((unsigned int)std::time(nullptr)) {}
	~RandomGenerator() {}

	virtual DType generate() = 0;
};

#endif