/*
* hidden_layer.hpp
*
*  Created on: 2016Äê6ÔÂ5ÈÕ
*      Author: Administrator
*/

#ifndef HIDDEN_LAYER_HPP_
#define HIDDEN_LAYER_HPP_

#include <vector>

#include "myblas.hpp"
#include "hidden_neuron.hpp"

using std::vector;

template<typename DType, template <typename> class Activation, template <typename> class PartialActivation>
class HiddenLayer {
	Activation<DType> m_foActivation;
	PartialActivation<DType> m_foPartialActivation;
public:
	HiddenLayer();
	~HiddenLayer();

	void active(const vector<HiddenNeuron<DType> *> & downs);

	void foreward(const vector<HiddenNeuron<DType> *> & ups, const vector<HiddenNeuron<DType> *> & downs);
	
	void backward(const vector<HiddenNeuron<DType> *> & ups, const vector<HiddenNeuron<DType> *> & downs);
};

// definitions

template<typename DType, template <typename> class Activation, template <typename> class PartialActivation>
HiddenLayer<DType, Activation, PartialActivation>::HiddenLayer() {

}

template<typename DType, template <typename> class Activation, template <typename> class PartialActivation>
HiddenLayer<DType, Activation, PartialActivation>::~HiddenLayer() {

}

/*
*	activate :
*		downs
*/
template<typename DType, template <typename> class Activation, template <typename> class PartialActivation>
void HiddenLayer<DType, Activation, PartialActivation>::active(const vector<HiddenNeuron<DType> *> & downs) {
	for (HiddenNeuron<DType> * down : downs) {
		// downs[i].active = sigma(downs[i].output)
		m_foActivation(down->getMutableActive(), down->getOutput(), down->getVecLen());
	}
}

/*
 *	calculate :
 *		ups
*/
template<typename DType, template <typename> class Activation, template <typename> class PartialActivation>
void HiddenLayer<DType, Activation, PartialActivation>::foreward(const vector<HiddenNeuron<DType> *> & ups, const vector<HiddenNeuron<DType> *> & downs) {
	int downNum = downs.size();
	for (HiddenNeuron<DType> * up : ups) {
		int upLen = up->getVecLen();
		// ups[j].output = ups[j].bias
		vector_copy_vector(up->getMutableOutput(), up->getBias(), upLen);
		for (int downId = 0; downId < downNum; ++downId) {
			HiddenNeuron<DType> * down = downs[downId];
			// ups[j].output += downs[i].active * ups[j].weight[i]
			vector_mul_matrix_add_output(up->getMutableOutput(), down->getActive(), up->getWeight(downId), down->getVecLen(), upLen);
		}
	}
}

/*
 *	calculate :
 *		dLoss / dZ		of	downs
 *		dLoss / dBias	of	downs
 *		dLoss / dWeight	of	ups		( for downs )
*/
template<typename DType, template <typename> class Activation, template <typename> class PartialActivation>
void HiddenLayer<DType, Activation, PartialActivation>::backward(const vector<HiddenNeuron<DType> *> & ups, const vector<HiddenNeuron<DType> *> & downs) {
	int downNum = downs.size();
	for (int downId = 0; downId < downNum; ++downId) {
		HiddenNeuron<DType> * down = downs[downId];
		int downLen = down->getVecLen();
		// downs[i].output_diff = 0
		memset(down->getMutableOutputDiff(), 0, sizeof(DType) * downLen);
		// downs[i].active_partial = sigma'(downs[i].output)
		m_foPartialActivation(down->getMutableActivePartial(), down->getOutput(), downLen);
		for (HiddenNeuron<DType> * up : ups) {
			// downs[i].output_diff += transpose(ups[j].output_diff) * ups[j].weight[i];
			transpose_vector_mul_matrix_add_output(down->getMutableOutputDiff(), up->getOutputDiff(), up->getWeight(downId), downLen, up->getVecLen());
		}
		// downs[i].output_diff = downs[i].output_diff .* sigma'(downs[i].output)
		vector_hadamard_product(down->getMutableOutputDiff(), down->getActivePartial(), downLen);
		// downs[i].bias_diff += downs[i].output_diff
		vector_add_vector(down->getMutableBiasDiff(), down->getOutputDiff(), downLen);
		for (HiddenNeuron<DType> * up : ups) {
			// ups[j].weight_diff[i] += trans(downs[i].active) * ups[j].output_diff
			vector_mul_vector_add_matrix(up->getMutableWeightDiff(downId), down->getActive(), up->getOutputDiff(), downLen, up->getVecLen());
		}
	}
}

#endif /* HIDDEN_LAYER_HPP_ */
