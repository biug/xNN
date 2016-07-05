/*
* input_layer.hpp
*
*  Created on: 2016Äê6ÔÂ5ÈÕ
*      Author: Administrator
*/

#ifndef INPUT_LAYER_HPP_
#define INPUT_LAYER_HPP_

#include <vector>

#include "myblas.hpp"
#include "input_neuron.hpp"
#include "hidden_neuron.hpp"

using std::vector;
using std::size_t;

template<typename DType>
class InputLayer {
public:
	InputLayer();
	~InputLayer();

	void foreward(const vector<HiddenNeuron<DType> *> & ups, const vector<InputNeuron<DType> *> & downs);
	
	void backward(const vector<HiddenNeuron<DType> *> & ups, const vector<InputNeuron<DType> *> & downs);
};

// definitions

template<typename DType>
InputLayer<DType>::InputLayer() {

}

template<typename DType>
InputLayer<DType>::~InputLayer() {

}

/*
 *	calculate :
 *		ups
*/
template<typename DType>
void InputLayer<DType>::foreward(const vector<HiddenNeuron<DType> *> & ups, const vector<InputNeuron<DType> *> & downs) {
	size_t downNum = downs.size();
	for (HiddenNeuron<DType> * up : ups) {
		int upLen = up->getVecLen();
		// ups[j].output = ups[j].bias
		vector_copy_vector(up->getMutableOutput(), up->getBias(), upLen);
		for (size_t downId = 0; downId < downNum; ++downId) {
			InputNeuron<DType> * down = downs.at(downId);
			// ups[j].output += downs[i].input * ups[j].weight[i]
			vector_mul_matrix_add_output(up->getMutableOutput(), down->getInput(), up->getWeight(downId), down->getVecLen(), upLen);
		}
	}
}

/*
 *	calculate :
 *		dLoss / dZ		of	downs
 *		dLoss / dWeight	of	ups		( for downs )
*/
template<typename DType>
void InputLayer<DType>::backward(const vector<HiddenNeuron<DType> *> & ups, const vector<InputNeuron<DType> *> & downs) {
	size_t downNum = downs.size();
	for (size_t downId = 0; downId < downNum; ++downId) {
		InputNeuron<DType> * down = downs[downId];
		int downLen = down->getVecLen();
		// downs[i].input_diff = 0
		memset(down->getMutableInputDiff(), 0, sizeof(DType) * downLen);
		for (HiddenNeuron<DType> * up : ups) {
			// downs[i].input_diff += transpose(ups[j].output_diff) * ups[j].weight[i];
			transpose_vector_mul_matrix_add_output(down->getMutableInputDiff(), up->getOutputDiff(), up->getWeight(downId), downLen, up->getVecLen());
		}
		for (HiddenNeuron<DType> * up : ups) {
			// ups[j].weight_diff[i] += trans(downs[i].input) * ups[j].output_diff
			vector_mul_vector_add_matrix(up->getMutableWeightDiff(downId), down->getInput(), up->getOutputDiff(), downLen, up->getVecLen());
		}
	}
}

#endif /* INPUT_LAYER_HPP_ */
