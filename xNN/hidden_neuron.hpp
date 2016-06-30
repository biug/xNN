/*
 * hidden_neuron.hpp
 *
 *  Created on: 2016Äê6ÔÂ3ÈÕ
 *      Author: Administrator
 */

#ifndef HIDDEN_NEURON_HPP_
#define HIDDEN_NEURON_HPP_

#include <vector>

using std::vector;
using std::size_t;

template<typename DType>
class HiddenNeuron {
	// vector length
	int m_nVecLen;

	// vector z
	DType * m_pOutput;
	// dLoss / dz
	DType * m_pOutputDiff;
	// vector a
	DType * m_pActive;
	// da / dz
	DType * m_pActivePartial;
	// bias
	DType * m_pBias;
	// dLoss / dbias
	DType * m_pBiasDiff;
	
	int * m_pWeightOffsets;
	int * m_pWeightSizes;

	// matrix w
	DType * m_pWeights;
	// dLoss / dw
	DType * m_pWeightDiffs;

public:
	HiddenNeuron(int vecLen, const vector<int> & downLens);
	~HiddenNeuron();

	inline int getVecLen() const {
		return m_nVecLen;
	}

	inline int getDownWeightSize(int down_id) const {
		return m_pWeightSizes[down_id];
	}

	inline DType * getMutableOutput() {
		return m_pOutput;
	}
	inline DType * getMutableOutputDiff() {
		return m_pOutputDiff;
	}
	inline DType * getMutableActive() {
		return m_pActive;
	}
	inline DType * getMutableActivePartial() {
		return m_pActivePartial;
	}
	inline DType * getMutableBias() {
		return m_pBias;
	}
	inline DType * getMutableBiasDiff() {
		return m_pBiasDiff;
	}
	inline DType * getMutableWeight(size_t down_id) {
		return &m_pWeights[m_pWeightOffsets[down_id]];
	}
	inline DType * getMutableWeightDiff(size_t down_id) {
		return &m_pWeightDiffs[m_pWeightOffsets[down_id]];
	}

	inline const DType * const getOutput() const {
		return m_pOutput;
	}
	inline const DType * const getOutputDiff() const {
		return m_pOutputDiff;
	}
	inline const DType * const getActive() const {
		return m_pActive;
	}
	inline const DType * const getActivePartial() const {
		return m_pActivePartial;
	}
	inline const DType * const getBias() const {
		return m_pBias;
	}
	inline const DType * const getBiasDiff() const {
		return m_pBiasDiff;
	}
	inline const DType * const getWeight(size_t down_id) const {
		return &m_pWeights[m_pWeightOffsets[down_id]];
	}
	inline const DType * const getWeightDiff(size_t down_id) const {
		return &m_pWeightDiffs[m_pWeightOffsets[down_id]];
	}

	DType norm2(int downNum) const;
};

// definitions

/*
*	a neuron contains up-layers' size and a vector size
*	we call up-layers' size ni, and vector size m
*	so weight wi is matrix which size is m * ni
*/
template<typename DType>
HiddenNeuron<DType>::HiddenNeuron(int vecLen, const vector<int> & downLens) : m_nVecLen(vecLen) {
	int downNum = (int)downLens.size();
	m_pOutput = new DType[m_nVecLen];
	m_pOutputDiff = new DType[m_nVecLen];
	m_pActive = new DType[m_nVecLen];
	m_pActivePartial = new DType[m_nVecLen];
	m_pBias = new DType[m_nVecLen];
	m_pBiasDiff = new DType[m_nVecLen];

	m_pWeightOffsets = new int[downNum + 1];
	m_pWeightSizes = new int[downNum];
	m_pWeightOffsets[0] = 0;
	for (int i = 1; i <= downNum; ++i) {
		m_pWeightOffsets[i] = m_pWeightOffsets[i - 1] + downLens[i] * m_nVecLen;
		m_pWeightSizes[i - 1] = m_pWeightOffsets[i] - m_pWeightOffsets[i - 1];
	}

	m_pWeights = new DType[m_pWeightOffsets[downNum]];
	m_pWeightDiffs = new DType[m_pWeightOffsets[downNum]];
}

template<typename DType>
HiddenNeuron<DType>::~HiddenNeuron() {
	delete[] m_pOutput;
	delete[] m_pOutputDiff;
	delete[] m_pActive;
	delete[] m_pActivePartial;
	delete[] m_pBias;
	delete[] m_pBiasDiff;

	delete[] m_pWeightOffsets;
	delete[] m_pWeightSizes;

	delete[] m_pWeights;
	delete[] m_pWeightDiffs;
}

template<typename DType>
DType HiddenNeuron<DType>::norm2(int downNum) const {
	DType norm = 0;
	for (int i = 0; i < m_nVecLen; ++i) {
		norm += m_pBiasDiff[i] * m_pBiasDiff[i];
	}
	for (int i = 0; i < downNum; ++i) {
		for (int j = m_pWeightOffsets[i]; j < m_pWeightSizes[i]; ++j)
			norm += m_pWeightDiffs[j] * m_pWeightDiffs[j];
	}
	return norm;
}

#endif /* HIDDEN_NEURON_HPP_ */
