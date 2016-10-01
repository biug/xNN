/*
* parser_net.hpp
*
*  Created on: 2016Äê6ÔÂ5ÈÕ
*      Author: Administrator
*/

#ifndef PARSER_NET_HPP_
#define PARSER_NET_HPP_

#include <ctime>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <unordered_map>

#include "graph.h"
#include "twostack_action.h"
#include "input_layer.hpp"
#include "hidden_layer.hpp"
#include "loss_layer.hpp"

#include "random_generator.hpp"

using std::vector;
using std::string;
using std::ifstream;
using std::ofstream;
using std::stringstream;
using std::unordered_map;

template<typename DType, template <typename> class Activation, template <typename> class PartialActivation, template <typename> class Loss, template <typename> class PartialLoss, template <typename Type, template <typename> class Neuron> class Updator>
class ParserNet {
	int m_nBatchSize;
	DType m_dLastValue;
	int m_nHiddenSize;

	vector<InputNeuron<DType> *> m_vecInputNeurons;
	vector<vector<HiddenNeuron<DType> *>> m_vecsHiddenNeurons;

	Updator<DType, HiddenNeuron> *m_WordUpdator, *m_POSUpdator, *m_LabelUpdator;
	vector<vector<Updator<DType, HiddenNeuron> *>> m_vecsHiddenUpdators;

	InputLayer<DType> m_lyrInputLayer;
	HiddenLayer<DType, Activation, PartialActivation> m_lyrHiddenLayers;
	LossLayer<DType, Loss, PartialLoss> m_lyrLossLayer;

	int m_nEmbeddingLen, m_nWordNum, m_nPOSNum, m_nLabelNum;
	DType *m_pWordMatrix, *m_pPOSMatrix, *m_pLabelMatrix;
	DType *m_pWordMatrixDiff, *m_pPOSMatrixDiff, *m_pLabelMatrixDiff;
	Token m_tWords, m_tPOSes, m_tLabels;
	DType m_dLossVal;

	int m_nCorrectLabel;
	vector<int> m_vecBatchWords, m_vecBatchPOSes, m_vecBatchLabels;

	int m_nTotalAction;
	int m_nCorrectAction;

	void updateEmbeddingDiffs();

	void initBatch();
	void foreward();
	void backward();
	void update();

	void readOneAction(const vector<vector<int>> & feat);
	void trainOneBatch(const vector<vector<vector<int>>> & feature);

	void loadModel(const string & model_file);
	void saveModel(const string & model_file, const string & embedding_file);
public:
	ParserNet(int embedding_num, const vector<vector<int>> & neuron_lens, const string & embedding_file, RandomGenerator<DType> * generator);
	~ParserNet();

	void train(const string & input_file, const string & test_file, const string & model_file, const string & embedding_file, int max_iter, DType threshold);
	void parse(const string & train, const string & input, const string & output, const string & model_file);
	void test(const string & file, const string & model_file);
	void generateNNData(DepGraph & graph, TwoStackAction & action, vector<vector<vector<int>>> & feature);
};

// definitions

template<typename DType, template <typename> class Activation, template <typename> class PartialActivation, template <typename> class Loss, template <typename> class PartialLoss, template <typename Type, template <typename> class Neuron> class Updator>
ParserNet<DType, Activation, PartialActivation, Loss, PartialLoss, Updator>::ParserNet(int embedding_num, const vector<vector<int>> & neuron_lens, const string & embedding_file, RandomGenerator<DType> * generator)
{
	m_nHiddenSize = neuron_lens.size() - 1;
	// init input neurons
	for (int i = 0; i < neuron_lens.front().size(); ++i) {
		m_vecInputNeurons.push_back(new InputNeuron<DType>(embedding_num, neuron_lens[0][i]));
	}
	// init hidden neurons
	for (int i = 1; i < neuron_lens.size(); ++i) {
		vector<HiddenNeuron<DType> *> vecHiddenNeurons;
		vector<Updator<DType, HiddenNeuron> *> vecHiddenUpdators;
		for (int j = 0; j < neuron_lens[i].size(); ++j) {
			HiddenNeuron<DType> * pNeuron = new HiddenNeuron<DType>(neuron_lens[i][j], neuron_lens[i - 1], generator);
			vecHiddenNeurons.push_back(pNeuron);
			vecHiddenUpdators.push_back(new Updator<DType, HiddenNeuron>(neuron_lens[i - 1].size(), pNeuron));
		}
		m_vecsHiddenNeurons.push_back(vecHiddenNeurons);
		m_vecsHiddenUpdators.push_back(vecHiddenUpdators);
	}
	// zero last layer bias
	memset(m_vecsHiddenNeurons.back().back()->getMutableBias(), 0, sizeof(DType) * m_vecsHiddenNeurons.back().back()->getVecLen());

	string line;
	ifstream ifs(embedding_file);
	// init embedding veclen
	ifs >> m_nEmbeddingLen;
	// init words
	ifs >> m_nWordNum;
	m_pWordMatrix = new DType[m_nWordNum * m_nEmbeddingLen];
	m_pWordMatrixDiff = new DType[m_nWordNum * m_nEmbeddingLen];
	m_WordUpdator = new Updator<DType, HiddenNeuron>(m_nWordNum * m_nEmbeddingLen);
	for (int i = 0, offset = 0; i < m_nWordNum; ++i) {
		string word;
		ifs >> word;
		m_tWords.add(word);
		for (int j = 0; j < m_nEmbeddingLen; ++j) {
			ifs >> m_pWordMatrix[offset++];
		}
	}
	// init postags
	ifs >> m_nPOSNum;
	m_pPOSMatrix = new DType[m_nPOSNum * m_nEmbeddingLen];
	m_pPOSMatrixDiff = new DType[m_nPOSNum * m_nEmbeddingLen];
	m_POSUpdator = new Updator<DType, HiddenNeuron>(m_nPOSNum * m_nEmbeddingLen);
	for (int i = 0, offset = 0; i < m_nPOSNum; ++i) {
		string pos;
		ifs >> pos;
		m_tPOSes.add(pos);
		for (int j = 0; j < m_nEmbeddingLen; ++j) {
			ifs >> m_pPOSMatrix[offset++];
		}
	}
	// init Labeltags
	ifs >> m_nLabelNum;
	m_pLabelMatrix = new DType[m_nLabelNum * m_nEmbeddingLen];
	m_pLabelMatrixDiff = new DType[m_nLabelNum * m_nEmbeddingLen];
	m_LabelUpdator = new Updator<DType, HiddenNeuron>(m_nLabelNum * m_nEmbeddingLen);
	for (int i = 0, offset = 0; i < m_nLabelNum; ++i) {
		string label;
		ifs >> label;
		m_tLabels.add(label);
		for (int j = 0; j < m_nEmbeddingLen; ++j) {
			ifs >> m_pLabelMatrix[offset++];
		}
	}
	std::cout << "net init complete" << std::endl;
}

template<typename DType, template <typename> class Activation, template <typename> class PartialActivation, template <typename> class Loss, template <typename> class PartialLoss, template <typename Type, template <typename> class Neuron> class Updator>
ParserNet<DType, Activation, PartialActivation, Loss, PartialLoss, Updator>::~ParserNet() {
	// free input neurons
	for (int i = 0; i < m_vecInputNeurons.size(); ++i) {
		delete m_vecInputNeurons[i];
	}
	// free hidden neurons
	for (int i = 0; i < m_vecsHiddenNeurons.size(); ++i) {
		for (int j = 0; j < m_vecsHiddenNeurons[i].size(); ++j) {
			delete m_vecsHiddenNeurons[i][j];
			delete m_vecsHiddenUpdators[i][j];
		}
	}
	// free words
	delete[] m_pWordMatrix;
	delete m_WordUpdator;
	// free POSs
	delete[] m_pPOSMatrix;
	delete m_POSUpdator;
	// free labels
	delete[] m_pLabelMatrix;
	delete m_LabelUpdator;
}

template<typename DType, template <typename> class Activation, template <typename> class PartialActivation, template <typename> class Loss, template <typename> class PartialLoss, template <typename Type, template <typename> class Neuron> class Updator>
void ParserNet<DType, Activation, PartialActivation, Loss, PartialLoss, Updator>::readOneAction(const vector<vector<int>> & feat) {
	// increase batch size
	++m_nBatchSize;
	// read word
	m_vecBatchWords.clear();
	for (const auto & word : feat[0]) {
		m_vecBatchWords.push_back(word);
	}
	// read poses
	m_vecBatchPOSes.clear();
	for (const auto & pos : feat[1]) {
		m_vecBatchPOSes.push_back(pos);
	}
	// read labels
	m_vecBatchLabels.clear();
	for (const auto & label : feat[2]) {
		m_vecBatchLabels.push_back(label);
	}
	// load embeddings
	m_vecInputNeurons[0]->loadEmbedding((const DType*)m_pWordMatrix, m_vecBatchWords);
	m_vecInputNeurons[1]->loadEmbedding((const DType*)m_pPOSMatrix, m_vecBatchPOSes);
	m_vecInputNeurons[2]->loadEmbedding((const DType*)m_pLabelMatrix, m_vecBatchLabels);
	// read action
	m_nCorrectLabel = feat[3].front();
}

template<typename DType, template <typename> class Activation, template <typename> class PartialActivation, template <typename> class Loss, template <typename> class PartialLoss, template <typename Type, template <typename> class Neuron> class Updator>
void ParserNet<DType, Activation, PartialActivation, Loss, PartialLoss, Updator>::updateEmbeddingDiffs() {
	m_vecInputNeurons[0]->updateEmbedding(m_pWordMatrixDiff, m_vecBatchWords);
	m_vecInputNeurons[1]->updateEmbedding(m_pPOSMatrixDiff, m_vecBatchPOSes);
	m_vecInputNeurons[2]->updateEmbedding(m_pLabelMatrixDiff, m_vecBatchLabels);
}

template<typename DType, template <typename> class Activation, template <typename> class PartialActivation, template <typename> class Loss, template <typename> class PartialLoss, template <typename Type, template <typename> class Neuron> class Updator>
void ParserNet<DType, Activation, PartialActivation, Loss, PartialLoss, Updator>::initBatch() {
	// zero batch size
	m_nBatchSize = 0;
	// zero neuron diffs
	for (const auto & neuron : m_vecsHiddenNeurons[0]) {
		neuron->initDiff(m_vecInputNeurons.size());
	}
	for (int i = 1; i < m_vecsHiddenNeurons.size(); ++i) {
		for (const auto & neuron : m_vecsHiddenNeurons[i]) {
			neuron->initDiff(m_vecsHiddenNeurons[i - 1].size());
		}
	}
	// zero embedding diffs
	memset(m_pWordMatrixDiff, 0, m_nWordNum * m_nEmbeddingLen * sizeof(DType));
	memset(m_pPOSMatrixDiff, 0, m_nPOSNum * m_nEmbeddingLen * sizeof(DType));
	memset(m_pLabelMatrixDiff, 0, m_nLabelNum * m_nEmbeddingLen * sizeof(DType));
}

template<typename DType, template <typename> class Activation, template <typename> class PartialActivation, template <typename> class Loss, template <typename> class PartialLoss, template <typename Type, template <typename> class Neuron> class Updator>
void ParserNet<DType, Activation, PartialActivation, Loss, PartialLoss, Updator>::foreward() {
	// foreward
	m_lyrInputLayer.foreward(m_vecsHiddenNeurons[0], m_vecInputNeurons);
	for (int i = 1; i < m_nHiddenSize; ++i) {
		m_lyrHiddenLayers.active(m_vecsHiddenNeurons[i - 1]);
		m_lyrHiddenLayers.foreward(m_vecsHiddenNeurons[i], m_vecsHiddenNeurons[i - 1]);
	}
	m_lyrLossLayer.foreward(m_vecsHiddenNeurons.back().back());
}

template<typename DType, template <typename> class Activation, template <typename> class PartialActivation, template <typename> class Loss, template <typename> class PartialLoss, template <typename Type, template <typename> class Neuron> class Updator>
void ParserNet<DType, Activation, PartialActivation, Loss, PartialLoss, Updator>::backward() {
	// backward
	m_lyrLossLayer.backward(m_vecsHiddenNeurons.back().back(), m_nCorrectLabel);
	for (int i = m_nHiddenSize - 1; i > 0; --i) {
		m_lyrHiddenLayers.backward(m_vecsHiddenNeurons[i], m_vecsHiddenNeurons[i - 1]);
	}
	m_lyrInputLayer.backward(m_vecsHiddenNeurons[0], m_vecInputNeurons);
	// update embeddings
	updateEmbeddingDiffs();

}

template<typename DType, template <typename> class Activation, template <typename> class PartialActivation, template <typename> class Loss, template <typename> class PartialLoss, template <typename Type, template <typename> class Neuron> class Updator>
void ParserNet<DType, Activation, PartialActivation, Loss, PartialLoss, Updator>::update() {
	if (m_nBatchSize == 0) return;
	// word embedding
	m_WordUpdator->update(m_pWordMatrix, m_pWordMatrixDiff, m_nWordNum * m_nEmbeddingLen, m_nBatchSize);
	// pos embedding
	m_POSUpdator->update(m_pPOSMatrix, m_pPOSMatrixDiff, m_nPOSNum * m_nEmbeddingLen, m_nBatchSize);
	// label embedding
	m_LabelUpdator->update(m_pLabelMatrix, m_pLabelMatrixDiff, m_nLabelNum * m_nEmbeddingLen, m_nBatchSize);
	// hidden neurons norm
	for (auto && vecUpdators : m_vecsHiddenUpdators) {
		for (auto && updator : vecUpdators) {
			updator->update(m_nBatchSize);
		}
	}
}

template<typename DType, template <typename> class Activation, template <typename> class PartialActivation, template <typename> class Loss, template <typename> class PartialLoss, template <typename Type, template <typename> class Neuron> class Updator>
void ParserNet<DType, Activation, PartialActivation, Loss, PartialLoss, Updator>::generateNNData(DepGraph & graph, TwoStackAction & action, vector<vector<vector<int>>> & feature) {
	TwoStackState state, cstate;
	state.clear();
	cstate.clear();
	graph.setLabels(action.Labels, action.VecLabelMap);
	action.extractOracle(state, graph);
	for (int i = 0, n = state.actionBack(); i <= n; ++i) {
		feature.push_back(cstate.features(&action, graph));
		feature.back().push_back({ state.action(i) });
		action.doAction(cstate, state.action(i));
	}
}

template<typename DType, template <typename> class Activation, template <typename> class PartialActivation, template <typename> class Loss, template <typename> class PartialLoss, template <typename Type, template <typename> class Neuron> class Updator>
void ParserNet<DType, Activation, PartialActivation, Loss, PartialLoss, Updator>::loadModel(const string & model_file) {
	// load models
	ifstream ifs(model_file);
	for (int i = 0, n = m_vecsHiddenNeurons.size(); i < n; ++i) {
		for (int j = 0, m = m_vecsHiddenNeurons[i].size(); j < m; ++j) {
			ifs >> *m_vecsHiddenNeurons[i][j];
		}
	}
}

template<typename DType, template <typename> class Activation, template <typename> class PartialActivation, template <typename> class Loss, template <typename> class PartialLoss, template <typename Type, template <typename> class Neuron> class Updator>
void ParserNet<DType, Activation, PartialActivation, Loss, PartialLoss, Updator>::saveModel(const string & model_file, const string & embedding_file) {
	// save models
	ofstream ofs(model_file);
	for (int i = 0, n = m_vecsHiddenNeurons.size(); i < n; ++i) {
		for (int j = 0, m = m_vecsHiddenNeurons[i].size(); j < m; ++j) {
			ofs << *m_vecsHiddenNeurons[i][j] << std::endl;
		}
	}
	// save embeddings
	ofs.close();
	ofs.open(embedding_file);
	ofs << m_nEmbeddingLen << std::endl;
	ofs << m_nWordNum << std::endl;
	for (int i = 0, offset = 0; i < m_nWordNum; ++i) {
		ofs << m_tWords[i] << std::endl;
		for (int j = 0; j < m_nEmbeddingLen; ++j) ofs << m_pWordMatrix[offset++] << ' '; ofs << std::endl;
	}
	ofs << m_nPOSNum << std::endl;
	for (int i = 0, offset = 0; i < m_nPOSNum; ++i) {
		ofs << m_tPOSes[i] << std::endl;
		for (int j = 0; j < m_nEmbeddingLen; ++j) ofs << m_pPOSMatrix[offset++] << ' '; ofs << std::endl;
	}
	ofs << m_nLabelNum << std::endl;
	for (int i = 0, offset = 0; i < m_nLabelNum; ++i) {
		ofs << m_tLabels[i] << std::endl;
		for (int j = 0; j < m_nEmbeddingLen; ++j) ofs << m_pLabelMatrix[offset++] << ' '; ofs << std::endl;
	}
	ofs.close();
}

template<typename DType, template <typename> class Activation, template <typename> class PartialActivation, template <typename> class Loss, template <typename> class PartialLoss, template <typename Type, template <typename> class Neuron> class Updator>
void ParserNet<DType, Activation, PartialActivation, Loss, PartialLoss, Updator>::train(const string & input_file, const string & test_file, const string & model_file, const string & embedding_file, int max_iter, DType threshold) {
	// read one batch
	TwoStackAction action;
	action.loadActions(input_file);
	DepGraph graph;
	int batch_size = 0;
	m_nTotalAction = 0;
	m_nCorrectAction = 0;
	vector<vector<vector<int>>> feature;
	m_dLossVal = DType();
	while (max_iter--) {
		ifstream graphfs(input_file);
		while (graphfs >> graph) {
			++batch_size;
			generateNNData(graph, action, feature);
			if (batch_size % ONE_BATCH == 0) {
				trainOneBatch(feature);
				feature.clear();
			}
			if (batch_size % OUTPUT_BATCH == 0) {
				std::cout << "Loss Val = " << m_dLossVal << std::endl;
				std::cout << "Correct Rate = " << (double)m_nCorrectAction / (double)m_nTotalAction << std::endl;
				m_dLossVal = DType();
				m_nTotalAction = 0;
				m_nCorrectAction = 0;
				saveModel(model_file, embedding_file);
			}
		}
		trainOneBatch(feature);
		saveModel(model_file, embedding_file);
		std::cout << "round " << max_iter << std::endl;
	}
}

template<typename DType, template <typename> class Activation, template <typename> class PartialActivation, template <typename> class Loss, template <typename> class PartialLoss, template <typename Type, template <typename> class Neuron> class Updator>
void ParserNet<DType, Activation, PartialActivation, Loss, PartialLoss, Updator>::trainOneBatch(const vector<vector<vector<int>>> & feature) {
	while (true) {
		int dropcount = 0;
		srand(time(NULL));
		for (InputNeuron<DType> * blob : m_vecInputNeurons) {
			blob->setDropout((double)(rand() % 100) / (double)100 < DROP_OUT_RATE);
			if (blob->getDropout()) ++dropcount;
		}
		if (dropcount < m_vecInputNeurons.size()) {
			break;
		}
	}

	initBatch();
	for (const auto & feat : feature) {
		readOneAction(feat);
		foreward();
		backward();
		m_dLossVal += m_vecsHiddenNeurons.back().back()->getActive()[m_nCorrectLabel];

		++m_nTotalAction;
		int maxLabel = 0;
		for (int i = 0; i < LOSS_WIDTH; ++i) {
			if (m_vecsHiddenNeurons.back().back()->getActive()[maxLabel] > m_vecsHiddenNeurons.back().back()->getActive()[i]) {
				maxLabel = i;
			}
		}
		if (maxLabel == m_nCorrectLabel) {
			++m_nCorrectAction;
		}
	}
	update();
}

template<typename DType, template <typename> class Activation, template <typename> class PartialActivation, template <typename> class Loss, template <typename> class PartialLoss, template <typename Type, template <typename> class Neuron> class Updator>
void ParserNet<DType, Activation, PartialActivation, Loss, PartialLoss, Updator>::parse(const string & train, const string & input, const string & output, const string & model_file) {
	loadModel(model_file);
	TwoStackAction action;
	TwoStackState state;
	action.loadActions(train);
	ifstream ifs(input);
	ofstream ofs(output);
	DepGraph graph;
	DepGraph out;
	m_vecInputNeurons[0]->setDropout(false);
	m_vecInputNeurons[1]->setDropout(false);
	m_vecInputNeurons[2]->setDropout(false);

	while (ifs >> graph) {
		state.clear();
		graph.setLabels(action.Labels, action.VecLabelMap);
		while (!state.stackEmpty() || state.size() < graph.size()) {
			auto features = state.features(&action, graph);
			m_vecBatchWords = features[0];
			m_vecBatchPOSes = features[1];
			m_vecBatchLabels = features[2];
			// load embeddings
			m_vecInputNeurons[0]->loadEmbedding((const DType*)m_pWordMatrix, m_vecBatchWords);
			m_vecInputNeurons[1]->loadEmbedding((const DType*)m_pPOSMatrix, m_vecBatchPOSes);
			m_vecInputNeurons[2]->loadEmbedding((const DType*)m_pLabelMatrix, m_vecBatchLabels);
			foreward();
			vector<pair<DType, int>> scores;
			for (int i = 0, n = m_vecsHiddenNeurons.back().back()->getVecLen(); i < n; ++i) {
				scores.push_back({ m_vecsHiddenNeurons.back().back()->getActive()[i], i });
			}
			std::sort(scores.begin(), scores.end());
			for (const auto & score : scores) {
				if (action.testAction(state, graph, score.second)) {
					//std::cout << "action is " << action.printAction(score.second) << std::endl;
					action.doAction(state, score.second);
					break;
				}
			}
		}
		out.clear();
		state.generateGraph(graph, out, action.Labels);
		ofs << out;
	}
}

template<typename DType, template <typename> class Activation, template <typename> class PartialActivation, template <typename> class Loss, template <typename> class PartialLoss, template <typename Type, template <typename> class Neuron> class Updator>
void ParserNet<DType, Activation, PartialActivation, Loss, PartialLoss, Updator>::test(const string & file, const string & model_file) {
	loadModel(model_file);
	ifstream ifs(file);
	int total = 0, correct = 0;
	std::cout << "load complete" << std::endl;
	while (readOneAction(ifs)) {
		foreward();
		const DType * const output = m_vecsHiddenNeurons.back().back()->getActive();
		int maxLabel = 0, len = m_vecsHiddenNeurons.back().back()->getVecLen();
		for (int i = 0; i < len; ++i) {
			if (output[maxLabel] > output[i]) maxLabel = i;
		}
		++total;
		if (m_nCorrectLabel == maxLabel) ++correct;
	}
	std::cout << "rate is " << (double)correct / (double)total;
}

#endif