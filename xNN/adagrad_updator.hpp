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
	vector<DType> m_vecWeightSums;
	DType m_dBiasSum;
	Neuron<DType> * m_pNeuron;
public:
	AdaGradUpdator(size_t downNum = 0, Neuron<DType> * neuron = nullptr);
	AdaGradUpdator(const AdaGradUpdator<DType, Neuron> & updator);
	~AdaGradUpdator() {}

	DType update(int batch);
	void update(DType * args, const DType * args_diff, DType norm, int size, int batch);
};

template<typename DType, template<typename> class Neuron>
AdaGradUpdator<DType, Neuron>::AdaGradUpdator(size_t downNum = 0, Neuron<DType> * neuron = nullptr) :
	m_vecWeightSums(downNum, (DType)ADAGRAD_EPSILON), m_dBiasSum((DType)ADAGRAD_EPSILON), m_pNeuron(neuron) {}

template<typename DType, template<typename> class Neuron>
AdaGradUpdator<DType, Neuron>::AdaGradUpdator(const AdaGradUpdator<DType, Neuron> & updator) :
	m_vecWeightSums(updator.m_vecWeightSums), m_dBiasSum(updator.m_dBiasSum), m_pNeuron(updator.m_pNeuron) {}

template<typename DType, template<typename> class Neuron>
DType AdaGradUpdator<DType, Neuron>::update(int batch) {
	DType norm = 0;
	for (size_t i = 0, n = m_vecWeightSums.size(); i < n; ++i) {
		DType weightNorm = m_pNeuron->getWeightNorm1(i);

		norm += weightNorm;
		m_vecWeightSums[i] += weightNorm;
		alpha_vector_add_beta_vector(m_pNeuron->getMutableWeight(i), m_pNeuron->getWeightDiff(i), -(DType)ADAGRAD_ALPHA / sqrt(m_vecWeightSums[i]) / (DType)batch, (DType)(1 - REGULA_LAMDA), m_pNeuron->getDownWeightSize(i));
	}
	DType biasNorm = m_pNeuron->getBiasNorm1();

	norm += biasNorm;
	m_dBiasSum += biasNorm;
	alpha_vector_add_beta_vector(m_pNeuron->getMutableBias(), m_pNeuron->getBiasDiff(), -(DType)ADAGRAD_ALPHA / sqrt(m_dBiasSum) / (DType)batch, (DType)(1 - REGULA_LAMDA), m_pNeuron->getVecLen());

	return norm;
}
template<typename DType, template<typename> class Neuron>
void AdaGradUpdator<DType, Neuron>::update(DType * args, const DType * args_diff, DType norm, int size, int batch) {
	m_dBiasSum += norm;
	alpha_vector_add_beta_vector(args, args_diff, -(DType)ADAGRAD_ALPHA / sqrt(m_dBiasSum) / (DType)batch, (DType)(1 - REGULA_LAMDA), size);
}

#endif