/*
* adagrad_updator.hpp
*
*  Created on: 2016��6��5��
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
	size_t m_nDownNum;
	DType * m_pBiasVec;
	DType * m_pWeightVec;
	Neuron<DType> * m_pNeuron;
public:
	AdaGradUpdator(size_t downNum = 0, Neuron<DType> * neuron = nullptr);
	AdaGradUpdator(const AdaGradUpdator<DType, Neuron> & updator);
	~AdaGradUpdator() {}

	void update(int batch);
	void update(DType * args, const DType * args_diff, int size, int batch);
};

template<typename DType, template<typename> class Neuron>
AdaGradUpdator<DType, Neuron>::AdaGradUpdator(size_t downNum = 0, Neuron<DType> * neuron = nullptr) : m_nDownNum(downNum), m_pNeuron(neuron) {
	const DType epsilon = 1e-10;

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
	const DType yita = (DType)ADAGRAD_ALPHA;
	const DType alpha = (DType)(1 - REGULA_LAMDA);

	const DType * biasDiff = m_pNeuron->getBiasDiff();
	const DType * weightDiff = m_pNeuron->getWeightDiff(0);
	DType * bias = m_pNeuron->getMutableBias();
	DType * weight = m_pNeuron->getMutableWeight(0);

	for (int i = 0, n = m_pNeuron->getVecLen(); i < n; ++i) {
		m_pBiasVec[i] += biasDiff[i] * biasDiff[i];
		bias[i] = alpha * bias[i] - yita / sqrt(m_pBiasVec[i]) * biasDiff[i];
	}

	for (int i = 0, n = m_pNeuron->getDownWeightOffset(m_nDownNum); i < n; ++i) {
		m_pWeightVec[i] += weightDiff[i] * weightDiff[i];
		weight[i] = alpha * weight[i] - yita / sqrt(m_pWeightVec[i]) * weightDiff[i];
	}
}
template<typename DType, template<typename> class Neuron>
void AdaGradUpdator<DType, Neuron>::update(DType * args, const DType * args_diff, int size, int batch) {
	const DType yita = (DType)ADAGRAD_ALPHA;
	const DType alpha = (DType)(1 - REGULA_LAMDA);

	for (int i = 0; i < size; ++i) {
		m_pBiasVec[i] += args_diff[i] * args_diff[i];
		args[i] = alpha * args[i] - yita / sqrt(m_pBiasVec[i]) * args_diff[i];
	}
}

#endif