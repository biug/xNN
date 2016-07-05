/*
* sgd_updator.hpp
*
*  Created on: 2016Äê6ÔÂ5ÈÕ
*      Author: Administrator
*/

#ifndef SGD_UPDATOR_HPP_
#define SGD_UPDATOR_HPP_

#include <cmath>
#include <vector>

#include "macros.h"
#include "myblas.hpp"

using std::vector;

template<typename DType, template<typename> class Neuron>
class SGDUpdator {
protected:
	size_t m_nDownNum;
	DType * m_pBiasVec;
	DType * m_pWeightVec;
	Neuron<DType> * m_pNeuron;
public:
	SGDUpdator(size_t downNum = 0, Neuron<DType> * neuron = nullptr);
	SGDUpdator(const SGDUpdator<DType, Neuron> & updator);
	~SGDUpdator();

	DType update(int batch);
	void update(DType * args, const DType * args_diff, DType norm, int size, int batch);
};

template<typename DType, template<typename> class Neuron>
SGDUpdator<DType, Neuron>::SGDUpdator(size_t downNum = 0, Neuron<DType> * neuron = nullptr) : m_nDownNum(downNum), m_pNeuron(neuron) {
	if (neuron != nullptr) {
		m_pBiasVec = new DType[neuron->getVecLen()];
		m_pWeightVec = new DType[neuron->getDownWeightOffset(downNum)];
		memset(m_pBiasVec, 0, sizeof(DType) * neuron->getVecLen());
		memset(m_pWeightVec, 0, sizeof(DType) * neuron->getDownWeightOffset(downNum));
	}
	else {
		m_pBiasVec = new DType[downNum];
		memset(m_pBiasVec, 0, sizeof(DType) * downNum);
	}
}

template<typename DType, template<typename> class Neuron>
SGDUpdator<DType, Neuron>::SGDUpdator(const SGDUpdator<DType, Neuron> & updator) :
	m_nDownNum(updator.m_nDownNum), m_pBiasVec(updator.m_pBiasVec), m_pWeightVec(updator.m_pWeightVec), m_pNeuron(updator.m_pNeuron) {}

template<typename DType, template<typename> class Neuron>
SGDUpdator<DType, Neuron>::~SGDUpdator() {
	delete[] m_pBiasVec;
	delete[] m_pWeightVec;
}

template<typename DType, template<typename> class Neuron>
DType SGDUpdator<DType, Neuron>::update(int batch) {
	DType norm = 0;
	for (size_t i = 0; i < m_nDownNum; ++i) {
		DType weightNorm = m_pNeuron->getWeightNorm1(i);

		norm += weightNorm;
		DType * vec = &m_pWeightVec[m_pNeuron->getDownWeightOffset(i)];
		int vecSize = m_pNeuron->getDownWeightSize(i);
		alpha_vector_add_beta_vector(vec, m_pNeuron->getWeightDiff(i), -(DType)SGD_ALPHA / (DType)batch, (DType)SGD_MOMENTUM, vecSize);
		alpha_vector_add_beta_vector(m_pNeuron->getMutableWeight(i), vec, (DType)1, (DType)(1 - REGULA_LAMDA), vecSize);
	}
	DType biasNorm = m_pNeuron->getBiasNorm1();

	norm += biasNorm;
	alpha_vector_add_beta_vector(m_pBiasVec, m_pNeuron->getBiasDiff(), -(DType)SGD_ALPHA / (DType)batch, (DType)SGD_MOMENTUM, m_pNeuron->getVecLen());
	alpha_vector_add_beta_vector(m_pNeuron->getMutableBias(), m_pBiasVec, (DType)1, (DType)(1 - REGULA_LAMDA), m_pNeuron->getVecLen());
	return norm;
}
template<typename DType, template<typename> class Neuron>
void SGDUpdator<DType, Neuron>::update(DType * args, const DType * args_diff, DType norm, int size, int batch) {
	alpha_vector_add_beta_vector(m_pBiasVec, args_diff, -(DType)SGD_ALPHA / (DType)batch, (DType)SGD_MOMENTUM, size);
	alpha_vector_add_beta_vector(args, m_pBiasVec, (DType)1, (DType)(1 - REGULA_LAMDA), size);
}

#endif