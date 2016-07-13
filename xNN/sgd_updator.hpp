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
	typedef Neuron<DType>	Blob;

protected:
	int m_nDownNum;
	DType * m_pBiasVec;
	DType * m_pWeightVec;
	Blob * m_pNeuron;
public:
	SGDUpdator(int downNum = 0, Blob * neuron = nullptr);
	SGDUpdator(const SGDUpdator<DType, Neuron> & updator);
	~SGDUpdator();

	void update(int batch);
	void update(DType * args, const DType * args_diff, int size, int batch);
};

template<typename DType, template<typename> class Neuron>
SGDUpdator<DType, Neuron>::SGDUpdator(int downNum = 0, Blob * neuron = nullptr) : m_nDownNum(downNum), m_pNeuron(neuron) {
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
void SGDUpdator<DType, Neuron>::update(int batch) {
	const DType miu = -static_cast<DType>(SGD_ALPHA) / static_cast<DType>(batch);

	static const DType momentum = static_cast<DType>(SGD_MOMENTUM);
	static const DType alpha = static_cast<DType>(1);
	static const DType beta = static_cast<DType>(1 - REGULAR_LAMDA);

	// bias
	alpha_vector_add_beta_vector(m_pBiasVec, m_pNeuron->getBiasDiff(), miu, momentum, m_pNeuron->getVecLen());
	alpha_vector_add_beta_vector(m_pNeuron->getMutableBias(), m_pBiasVec, alpha, beta, m_pNeuron->getVecLen());
	// weight
	alpha_vector_add_beta_vector(m_pWeightVec, m_pNeuron->getWeightDiff(0), miu, momentum, m_pNeuron->getDownWeightOffset(m_nDownNum));
	alpha_vector_add_beta_vector(m_pNeuron->getMutableWeight(0), m_pWeightVec, alpha, beta, m_pNeuron->getDownWeightOffset(m_nDownNum));
}

template<typename DType, template<typename> class Neuron>
void SGDUpdator<DType, Neuron>::update(DType * args, const DType * args_diff, int size, int batch) {
	alpha_vector_add_beta_vector(m_pBiasVec, args_diff, -static_cast<DType>(SGD_ALPHA) / static_cast<DType>(batch), static_cast<DType>(SGD_MOMENTUM), size);
	alpha_vector_add_beta_vector(args, m_pBiasVec, static_cast<DType>(1), static_cast<DType>(1 - REGULAR_LAMDA), size);
}

#endif