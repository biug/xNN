/*
* input_neuron.hpp
*
*  Created on: 2016Äê6ÔÂ3ÈÕ
*      Author: Administrator
*/

#ifndef INPUT_NEURON_HPP_
#define INPUT_NEURON_HPP_

#include <vector>

using std::vector;

template<typename DType>
class InputNeuron {
	// vector length
	int m_nEmbeddingLen;
	int m_nVecLen;

	// vector z
	DType * m_pInput;
	// dLoss / dz
	DType * m_pInputDiff;

public:
	InputNeuron(int embeddingLen, int vecLen);
	~InputNeuron();

	void loadEmbedding(const DType * embeddingMatrix, const vector<int> & ids);
	void updateEmbedding(DType * embeddingMatrixDiff, const vector<int> & ids);

	inline int getVecLen() const {
		return m_nVecLen;
	}

	inline DType * getMutableInput() {
		return m_pInput;
	}
	inline DType * getMutableInputDiff() {
		return m_pInputDiff;
	}

	inline const DType * const getInput() const {
		return m_pInput;
	}
	inline const DType * const getInputDiff() const {
		return m_pInputDiff;
	}
};

// definitions

/**
*	a neuron contains up-layers' size and a vector size
*	we call up-layers' size ni, and vector size m
*	so weight wi is matrix which size is m * ni
*/
template<typename DType>
InputNeuron<DType>::InputNeuron(int embeddingLen, int vecLen) : m_nEmbeddingLen(embeddingLen), m_nVecLen(vecLen) {
	m_pInput = new DType[m_nVecLen];
	m_pInputDiff = new DType[m_nVecLen];
}

template<typename DType>
InputNeuron<DType>::~InputNeuron() {
	delete[] m_pInput;
	delete[] m_pInputDiff;
}

template<typename DType>
void InputNeuron<DType>::loadEmbedding(const DType * embeddingMatrix, const vector<int> & ids) {
	int offset = 0;
	for (const auto & id : ids) {
		vector_copy_vector(&m_pInput[offset], &embeddingMatrix[id * m_nEmbeddingLen], m_nEmbeddingLen);
		offset += m_nEmbeddingLen;
	}
}

template<typename DType>
void InputNeuron<DType>::updateEmbedding(DType * embeddingMatrixDiff, const vector<int> & ids) {
	int offset = 0;
	for (const auto & id : ids) {
		vector_add_vector(&embeddingMatrixDiff[id * m_nEmbeddingLen], &m_pInputDiff[offset], m_nEmbeddingLen);
		offset += m_nEmbeddingLen;
	}
}

#endif /* INPUT_NEURON_HPP_ */
