/*
 * loss_layer.hpp
 *
 *  Created on: 2016Äê6ÔÂ5ÈÕ
 *      Author: Administrator
 */

#ifndef LOSS_LAYER_HPP_
#define LOSS_LAYER_HPP_

#include <vector>

#include "hidden_neuron.hpp"

using std::vector;

template<typename DType, template <typename> class Loss, template <typename> class PartialLoss>
class LossLayer {
	typedef HiddenNeuron<DType>		HiddenBlob;

	Loss<DType> m_foLoss;
	PartialLoss<DType> m_foPartialLoss;
public:
	LossLayer();
	~LossLayer();

	inline void foreward(HiddenBlob * down) {
		// down.active = Loss(down.output)
		m_foLoss(down->getMutableActive(), down->getOutput(), down->getVecLen());
	}
	
	inline void backward(HiddenBlob * down, int correctLabel) {
		// down.output_diff = partialLoss(down.output, up.correct)
		m_foPartialLoss(down->getMutableOutputDiff(), down->getOutput(), correctLabel, down->getVecLen());
	}
};

// definitions

template<typename DType, template <typename> class Loss, template <typename> class PartialLoss>
LossLayer<DType, Loss, PartialLoss>::LossLayer() {

}

template<typename DType, template <typename> class Loss, template <typename> class PartialLoss>
LossLayer<DType, Loss, PartialLoss>::~LossLayer() {

}
#endif /* LOSS_LAYER_HPP_ */
