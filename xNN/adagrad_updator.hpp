/*
* adagrad_updator.hpp
*
*  Created on: 2016Äê6ÔÂ5ÈÕ
*      Author: Administrator
*/

#ifndef ADAGRAD_UPDATOR_HPP_
#define ADAGRAD_UPDATOR_HPP_

#include <cmath>
#include <vector>

#include "macros.h"
#include "myblas.hpp"

using std::vector;

template<typename DType, template<typename> class Neuron>
class AdaGradUpdator {
protected:
	int m_nDownNum;
	DType * m_pBiasVec;
	DType * m_pWeightVec;
	Neuron<DType> * m_pNeuron;
public:
	AdaGradUpdator(int downNum = 0, Neuron<DType> * neuron = nullptr);
	AdaGradUpdator(const AdaGradUpdator<DType, Neuron> & updator);
	~AdaGradUpdator() {}

	void update(int batch);
	void update(DType * args, const DType * args_diff, int size, int batch);
};

template<typename DType, template<typename> class Neuron>
AdaGradUpdator<DType, Neuron>::AdaGradUpdator(int downNum = 0, Neuron<DType> * neuron = nullptr) : m_nDownNum(downNum), m_pNeuron(neuron) {
	const DType epsilon = static_cast<DType>(GLOBAL_EPSILON);

	if (neuron != nullptr) {
		m_pBiasVec = new DType[neuron->getVecLen()];
		m_pWeightVec = new DType[neuron->getDownWeightOffset(downNum)];
		for (int i = 0, n = neuron->getVecLen(); i < n; ++i) {
			m_pBiasVec[i] = epsilon;
		}
		for (int i = 0, n = neuron->getDownWeightOffset(m_nDownNum); i < n; ++i) {
			m_pWeightVec[i] = epsilon;
		}
	}
	else {
		m_pBiasVec = new DType[downNum];
		for (int i = 0; i < downNum; ++i) {
			m_pBiasVec[i] = epsilon;
		}
	}
}

template<typename DType, template<typename> class Neuron>
AdaGradUpdator<DType, Neuron>::AdaGradUpdator(const AdaGradUpdator<DType, Neuron> & updator) :
	m_vecWeightSums(updator.m_vecWeightSums), m_dBiasSum(updator.m_dBiasSum), m_pNeuron(updator.m_pNeuron) {}

template<typename DType, template<typename> class Neuron>
void AdaGradUpdator<DType, Neuron>::update(int batch) {
	const DType yita = static_cast<DType>(ADAGRAD_ALPHA);
	const DType alpha = static_cast<DType>(1 - REGULAR_LAMDA);

	const DType * biasDiff = m_pNeuron->getBiasDiff();
	const DType * weightDiff = m_pNeuron->getWeightDiff(0);
	DType * bias = m_pNeuron->getMutableBias();
	DType * weight = m_pNeuron->getMutableWeight(0);

	for (int i = 0, n = m_pNeuron->getVecLen(); i < n; ++i) {
		DType diff = biasDiff[i] / batch;
		m_pBiasVec[i] += diff * diff;
		bias[i] = alpha * bias[i] - yita / sqrt(m_pBiasVec[i]) * diff;
	}

	for (int i = 0, n = m_pNeuron->getDownWeightOffset(m_nDownNum); i < n; ++i) {
		DType diff = weightDiff[i] / batch;
		m_pWeightVec[i] += diff * diff;
		weight[i] = alpha * weight[i] - yita / sqrt(m_pWeightVec[i]) * diff;
	}
}

template<typename DType, template<typename> class Neuron>
void AdaGradUpdator<DType, Neuron>::update(DType * args, const DType * args_diff, int size, int batch) {
	const DType yita = static_cast<DType>(ADAGRAD_ALPHA);
	const DType alpha = static_cast<DType>(1 - REGULAR_LAMDA);

	for (int i = 0; i < size; ++i) {
		DType diff = args_diff[i] / batch;
		m_pBiasVec[i] += diff * diff;
		args[i] = alpha * args[i] - yita / sqrt(m_pBiasVec[i]) * diff;
	}
}

#endif