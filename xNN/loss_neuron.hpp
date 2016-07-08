/*
 * loss_neuron.hpp
 *
 *  Created on: 2016Äê6ÔÂ6ÈÕ
 *      Author: Administrator
 */

#ifndef LOSS_NEURON_HPP_
#define LOSS_NEURON_HPP_

template<typename DType>
class LossNeuron {
	int m_nVecLen;
	DType * m_pOutput;
	DType * m_pOutputPartial;
	DType * m_pCorrectLabel;

public:
	LossNeuron(int vec_len);
	~LossNeuron();

	inline int getVecLen() const {
		return m_nVecLen;
	}

	inline DType * getMutableOutput() {
		return m_pOutput;
	}
	inline DType * getMutableOutputPartial() {
		return m_pOutputPartial;
	}
	inline DType * getMutableCorrectLabel() {
		return m_pCorrectLabel;
	}

	inline const DType * const getOutput() const {
		return m_pOutput;
	}
	inline const DType * const getOutputPartial() const {
		return m_pOutputPartial;
	}
	inline const DType * const getCorrectLabel() const {
		return m_pCorrectLabel;
	}
};

#endif /* LOSS_NEURON_HPP_ */
