/*
 * hidden_neuron.hpp
 *
 *  Created on: 2016Äê6ÔÂ3ÈÕ
 *      Author: Administrator
 */

#ifndef HIDDEN_NEURON_HPP_
#define HIDDEN_NEURON_HPP_

#include <cmath>
#include <vector>
#include <iostream>

#include "random_generator.hpp"

using std::vector;
using std::size_t;
using std::iostream;

template<typename DType>
class HiddenNeuron {
	// vector length
	int m_nVecLen;
	// down num
	int m_nDownNum;

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
	HiddenNeuron(int vecLen, const vector<int> & downLens, RandomGenerator<DType> * generator);
	~HiddenNeuron();

	void initDiff(int downNum);

	inline int getVecLen() const {
		return m_nVecLen;
	}

	inline int getDownNum() const {
		return m_nDownNum;
	}

	inline int & getMutableVecLen() {
		return m_nVecLen;
	}

	inline int & getMutableDownNum() {
		return m_nDownNum;
	}

	inline int getDownWeightSize(int down_id) const {
		return m_pWeightSizes[down_id];
	}

	inline int getDownWeightOffset(int down_id) const {
		return m_pWeightOffsets[down_id];
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
	inline DType * getMutableWeight(int down_id) {
		return &m_pWeights[m_pWeightOffsets[down_id]];
	}
	inline DType * getMutableWeightDiff(int down_id) {
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
	inline const DType * const getWeight(int down_id) const {
		return &m_pWeights[m_pWeightOffsets[down_id]];
	}
	inline const DType * const getWeightDiff(int down_id) const {
		return &m_pWeightDiffs[m_pWeightOffsets[down_id]];
	}

	DType norm1(int down_num) const;
	DType norm2(int down_num) const;

	DType getWeightNorm1(int down_id) const;
	DType getBiasNorm1() const;

	DType getWeightNorm2(int down_id) const;
	DType getBiasNorm2() const;

	friend std::istream & operator >> (std::istream & is, HiddenNeuron<DType> & neuron) {
		int num;
		is >> num;
		for (int i = 0, n = neuron.getVecLen(); i < n; ++i) {
			is >> neuron.getMutableBias()[i];
		}
		is >> num;
		for (int i = 0, n = neuron.getDownWeightOffset(neuron.getDownNum()); i < n; ++i) {
			is >> neuron.getMutableWeight(0)[i];
		}
		return is;
	}

	friend std::ostream & operator << (std::ostream & os, HiddenNeuron<DType> & neuron) {
		os << neuron.getVecLen() << std::endl;
		for (int i = 0; i < neuron.getVecLen(); ++i) {
			os << neuron.getBias()[i] << ' ';
		}
		os << std::endl;
		os << neuron.getDownWeightOffset(neuron.getDownNum()) << std::endl;
		for (int i = 0, n = neuron.getDownWeightOffset(neuron.getDownNum()); i < n; ++i) {
			os << neuron.getWeight(0)[i] << ' ';
		}
		os << std::endl;
		return os;
	}
};

// definitions

/*
*	a neuron contains up-layers' size and a vector size
*	we call up-layers' size ni, and vector size m
*	so weight wi is matrix which size is m * ni
*/
template<typename DType>
HiddenNeuron<DType>::HiddenNeuron(int vecLen, const vector<int> & downLens, RandomGenerator<DType> * generator) : m_nVecLen(vecLen), m_nDownNum(downLens.size()) {
	int downNum = downLens.size();
	m_pOutput = new DType[m_nVecLen];
	m_pOutputDiff = new DType[m_nVecLen];
	m_pActive = new DType[m_nVecLen];
	m_pActivePartial = new DType[m_nVecLen];
	m_pBias = new DType[m_nVecLen];
	m_pBiasDiff = new DType[m_nVecLen];

	for (int i = 0; i < m_nVecLen; ++i) {
		m_pBias[i] = generator->generate();
	}

	m_pWeightOffsets = new int[downNum + 1];
	m_pWeightSizes = new int[downNum];
	m_pWeightOffsets[0] = 0;
	for (int i = 1; i <= downNum; ++i) {
		m_pWeightOffsets[i] = m_pWeightOffsets[i - 1] + downLens[i - 1] * m_nVecLen;
		m_pWeightSizes[i - 1] = m_pWeightOffsets[i] - m_pWeightOffsets[i - 1];
	}

	m_pWeights = new DType[m_pWeightOffsets[downNum]];
	m_pWeightDiffs = new DType[m_pWeightOffsets[downNum]];

	for (int i = 0, n = m_pWeightOffsets[downNum]; i < n; ++i) {
		m_pWeights[i] = generator->generate();
	}
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
void HiddenNeuron<DType>::initDiff(int down_num) {
	memset(m_pBiasDiff, 0, sizeof(DType) * m_nVecLen);
	memset(m_pWeightDiffs, 0, sizeof(DType) * m_pWeightOffsets[down_num]);
}

template<typename DType>
DType HiddenNeuron<DType>::norm1(int down_num) const {
	DType norm = 0;
	for (int i = 0; i < m_nVecLen; ++i) {
		norm += std::abs(m_pBiasDiff[i]);
	}
	for (int i = 0, n = m_pWeightOffsets[down_num]; i < n; ++i) {
		norm += std::abs(m_pWeightDiffs[i]);
	}
	return norm;
}

template<typename DType>
DType HiddenNeuron<DType>::norm2(int down_num) const {
	DType norm = 0;
	for (int i = 0; i < m_nVecLen; ++i) {
		norm += m_pBiasDiff[i] * m_pBiasDiff[i];
	}
	for (int i = 0, n = m_pWeightOffsets[down_num]; i < n; ++i) {
		norm += m_pWeightDiffs[i] * m_pWeightDiffs[i];
	}
	return norm;
}

template<typename DType>
DType HiddenNeuron<DType>::getWeightNorm1(int down_id) const {
	DType norm = 0;
	for (int i = m_pWeightOffsets[down_id], n = m_pWeightOffsets[down_id + 1]; i < n; ++i) {
		norm += std::abs(m_pWeightDiffs[i]);
	}
	return norm;
}

template<typename DType>
DType HiddenNeuron<DType>::getBiasNorm1() const {
	DType norm = 0;
	for (int i = 0; i < m_nVecLen; ++i) {
		norm += std::abs(m_pBiasDiff[i]);
	}
	return norm;
}

template<typename DType>
DType HiddenNeuron<DType>::getWeightNorm2(int down_id) const {
	DType norm = 0;
	for (int i = m_pWeightOffsets[down_id], n = m_pWeightOffsets[down_id + 1]; i < n; ++i) {
		norm += m_pWeightDiffs[i] * m_pWeightDiffs[i];
	}
	return norm;
}

template<typename DType>
DType HiddenNeuron<DType>::getBiasNorm2() const {
	DType norm = 0;
	for (int i = 0; i < m_nVecLen; ++i) {
		norm += m_pBiasDiff[i] * m_pBiasDiff[i];
	}
	return norm;
}

#endif /* HIDDEN_NEURON_HPP_ */
