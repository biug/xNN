/*
* parser_net.hpp
*
*  Created on: 2016Äê6ÔÂ5ÈÕ
*      Author: Administrator
*/

#ifndef PARSER_NET_HPP_
#define PARSER_NET_HPP_

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>

#include "input_layer.hpp"
#include "hidden_layer.hpp"
#include "loss_layer.hpp"

using std::vector;
using std::string;
using std::ifstream;
using std::stringstream;
using std::unordered_map;

template<typename DType, template <typename> class Activation, template <typename> class PartialActivation, template <typename> class Loss, template <typename> class PartialLoss>
class ParserNet {
	const DType g_dMomentum = DType(0.95);
	const DType g_dLearningRate = DType(0.001);

	vector<InputNeuron<DType> *> m_vecInputNeurons;
	vector<vector<HiddenNeuron<DType> *>> m_vecsHiddenNeurons;

	InputLayer<DType> m_lyrInputLayer;
	HiddenLayer<DType, Activation, PartialActivation> m_lyrHiddenLayers;
	LossLayer<DType, Loss, PartialLoss> m_lyrLossLayer;

	int m_nEmbeddingLen, m_nWordNum, m_nPOSNum, m_nLabelNum;
	DType **m_pWordMatrix, **m_pPOSMatrix, **m_pLabelMatrix;
	DType **m_pWordMatrixDiff, **m_pPOSMatrixDiff, **m_pLabelMatrixDiff;
	unordered_map<string, int> m_mapWordOrder, m_mapPOSOrder, m_mapLabelOrder;
public:
	ParserNet(int embeddingNum, const vector<vector<int>> & neuronLens, const string & embedding_file);
	~ParserNet();

	void train(const vector<string> & batchFiles, int max_iter, DType threshold);
};

// definitions

template<typename DType, template <typename> class Activation, template <typename> class PartialActivation, template <typename> class Loss, template <typename> class PartialLoss>
ParserNet<DType, Activation, PartialActivation, Loss, PartialLoss>::ParserNet(int embeddingNum, const vector<vector<int>> & neuronLens, const string & embedding_file) {
	// init input neurons
	for (int i = 0; i < neuronLens.front().size(); ++i) {
		m_vecInputNeurons.push_back(new InputNeuron<DType>(embeddingNum, neuronLens[0][i]));
	}
	// init hidden neurons
	for (int i = 1; i < neuronLens.size(); ++i) {
		vector<HiddenNeuron<DType> *> vecHiddenNeurons;
		for (int j = 0; j < neuronLens[i].size(); ++j) {
			vecHiddenNeurons.push_back(new HiddenNeuron<DType>(neuronLens[i][j], neuronLens[i - 1]));
		}
		m_vecsHiddenNeurons.push_back(vecHiddenNeurons);
	}

	string line;
	ifstream ifs(embedding_file);
	// init embedding veclen
	ifs >> m_nEmbeddingLen;
	// init words
	ifs >> m_nWordNum;
	m_pWordMatrix = new DType*[m_nWordNum];
	m_pWordMatrixDiff = new DType*[m_nWordNum];
	for (int i = 0; i < m_nWordNum; ++i) {
		string word;
		ifs >> word;
		m_pWordMatrix[m_mapWordOrder[word] = i] = new DType[m_nEmbeddingLen];
		m_pWordMatrixDiff[m_mapWordOrder[word] = i] = new DType[m_nEmbeddingLen];
		for (int j = 0; j < m_nEmbeddingLen; ++j) {
			ifs >> m_pWordMatrix[i][j];
		}
	}
	// init postags
	ifs >> m_nPOSNum;
	m_pPOSMatrix = new DType*[m_nPOSNum];
	m_pPOSMatrixDiff = new DType*[m_nPOSNum];
	for (int i = 0; i < m_nPOSNum; ++i) {
		string POS;
		ifs >> POS;
		m_pPOSMatrix[m_mapPOSOrder[POS] = i] = new DType[m_nEmbeddingLen];
		m_pPOSMatrixDiff[m_mapPOSOrder[POS] = i] = new DType[m_nEmbeddingLen];
		for (int j = 0; j < m_nEmbeddingLen; ++j) {
			ifs >> m_pPOSMatrix[i][j];
		}
	}
	// init Labeltags
	ifs >> m_nLabelNum;
	m_pLabelMatrix = new DType*[m_nLabelNum];
	m_pLabelMatrixDiff = new DType*[m_nLabelNum];
	for (int i = 0; i < m_nLabelNum; ++i) {
		string label;
		ifs >> label;
		m_pLabelMatrix[m_mapLabelOrder[label] = i] = new DType[m_nEmbeddingLen];
		m_pLabelMatrixDiff[m_mapLabelOrder[label] = i] = new DType[m_nEmbeddingLen];
		for (int j = 0; j < m_nEmbeddingLen; ++j) {
			ifs >> m_pLabelMatrix[i][j];
		}
	}
}

template<typename DType, template <typename> class Activation, template <typename> class PartialActivation, template <typename> class Loss, template <typename> class PartialLoss>
ParserNet<DType, Activation, PartialActivation, Loss, PartialLoss>::~ParserNet() {
	// free input neurons
	for (int i = 0; i < m_vecInputNeurons.size(); ++i) {
		delete m_vecInputNeurons[i];
	}
	// free hidden neurons
	for (int i = 0; i < m_vecsHiddenNeurons.size(); ++i) {
		for (int j = 0; j < m_vecsHiddenNeurons[i].size(); ++j) {
			delete m_vecsHiddenNeurons[i][j];
		}
	}
	// free words
	for (int i = 0; i < m_nWordNum; ++i) {
		delete[] m_pWordMatrix[i];
	}
	delete[] m_pWordMatrix;
	// free POSs
	for (int i = 0; i < m_nPOSNum; ++i) {
		delete[] m_pPOSMatrix[i];
	}
	delete[] m_pPOSMatrix;
	// free labels
	for (int i = 0; i < m_nLabelNum; ++i) {
		delete[] m_pLabelMatrix[i];
	}
	delete[] m_pLabelMatrix;
	// free maps
	m_mapWordOrder.clear();
	m_mapPOSOrder.clear();
	m_mapLabelOrder.clear();
}

template<typename DType, template <typename> class Activation, template <typename> class PartialActivation, template <typename> class Loss, template <typename> class PartialLoss>
void ParserNet<DType, Activation, PartialActivation, Loss, PartialLoss>::train(const vector<string> & batchFiles, int max_iter, DType threshold) {
	// read one batch
	int hiddenSize = (int)m_vecsHiddenNeurons.size();
	for (const auto & batchFile : batchFiles) {
		for (int iter = 0; iter < max_iter; ++iter) {
			ifstream ifs(batchFile);
			string line;
			stringstream ss;
			int id;
			// zero embedding diffs
			memset(m_pWordMatrixDiff, 0, m_nWordNum * m_nEmbeddingLen * sizeof(DType));
			memset(m_pPOSMatrixDiff, 0, m_nWordNum * m_nEmbeddingLen * sizeof(DType));
			memset(m_pLabelMatrixDiff, 0, m_nWordNum * m_nEmbeddingLen * sizeof(DType));
			while (true) {
				int correctLabel;
				vector<int> words, poses, labels;
				getline(ifs, line);
				if (line.empty()) break;
				ss << line;
				while (ss >> id) {
					words.push_back(id);
				}
				getline(ifs, line);
				ss << line;
				while (ss >> id) {
					poses.push_back(id);
				}
				getline(ifs, line);
				ss << line;
				while (ss >> id) {
					labels.push_back(id);
				}
				getline(ifs, line);
				ss << line;
				ss >> correctLabel;
				// load embeddings
				m_vecInputNeurons[0]->loadEmbedding((const DType**)m_pWordMatrix, words);
				m_vecInputNeurons[1]->loadEmbedding((const DType**)m_pPOSMatrix, poses);
				m_vecInputNeurons[2]->loadEmbedding((const DType**)m_pLabelMatrix, labels);
				// foreward
				m_lyrInputLayer.foreward(m_vecsHiddenNeurons[0], m_vecInputNeurons);
				for (int i = 1; i < hiddenSize; ++i) {
					m_lyrHiddenLayers.foreward(m_vecsHiddenNeurons[i], m_vecsHiddenNeurons[i - 1]);
				}
				m_lyrLossLayer.foreward(m_vecsHiddenNeurons.back().back());
				// backward
				m_lyrLossLayer.backward(m_vecsHiddenNeurons.back().back(), correctLabel);
				for (int i = hiddenSize - 1; i > 0; --i) {
					m_lyrHiddenLayers.backward(m_vecsHiddenNeurons[i], m_vecsHiddenNeurons[i - 1]);
				}
				m_lyrInputLayer.backward(m_vecsHiddenNeurons[0], m_vecInputNeurons);
				m_vecInputNeurons[0]->updateEmbedding(m_pWordMatrixDiff, words);
				m_vecInputNeurons[1]->updateEmbedding(m_pPOSMatrixDiff, poses);
				m_vecInputNeurons[2]->updateEmbedding(m_pLabelMatrixDiff, labels);
			}
			// update batch
			for (int i = hiddenSize - 1; i > 0; --i) {
				m_lyrHiddenLayers.update(m_vecsHiddenNeurons[i], (int)m_vecsHiddenNeurons[i - 1].size(), g_dMomentum, g_dLearningRate);
			}
			m_lyrInputLayer.update(m_vecsHiddenNeurons.front(), (int)m_vecInputNeurons.size(), g_dMomentum, g_dLearningRate);
			DType norm = 0;
			for (int i = 0; i < m_nWordNum; ++i) {
				// word[i] = (-learning_rate) * word_diff[i] + momentum * word[i]
				alpha_vector_add_beta_vector(m_pWordMatrix[i], m_pWordMatrixDiff[i], -g_dLearningRate, g_dMomentum, m_nEmbeddingLen);
				for (int j = 0; j < m_nEmbeddingLen; ++j) {
					norm += m_pWordMatrixDiff[i][j] * m_pWordMatrixDiff[i][j];
				}
			}
			for (int i = 0; i < m_nPOSNum; ++i) {
				// pos[i] = (-learning_rate) * pos_diff[i] + momentum * pos[i]
				alpha_vector_add_beta_vector(m_pPOSMatrix[i], m_pPOSMatrixDiff[i], -g_dLearningRate, g_dMomentum, m_nEmbeddingLen);
				for (int j = 0; j < m_nEmbeddingLen; ++j) {
					norm += m_pPOSMatrix[i][j] * m_pPOSMatrix[i][j];
				}
			}
			for (int i = 0; i < m_nWordNum; ++i) {
				// label[i] = (-learning_rate) * label_diff[i] + momentum * label[i]
				alpha_vector_add_beta_vector(m_pLabelMatrix[i], m_pLabelMatrixDiff[i], -g_dLearningRate, g_dMomentum, m_nEmbeddingLen);
				for (int j = 0; j < m_nEmbeddingLen; ++j) {
					norm += m_pLabelMatrix[i][j] * m_pLabelMatrix[i][j];
				}
			}
			for (const auto & neuron : m_vecsHiddenNeurons[0]) {
				norm += neuron->norm2((int)m_vecInputNeurons.size());
			}
			for (int i = 1; i < m_vecsHiddenNeurons.size(); ++i) {
				for (const auto & neuron : m_vecsHiddenNeurons[i]) {
					norm += neuron->norm2((int)m_vecsHiddenNeurons[i - 1].size());
				}
			}
			if (norm < threshold) {
				break;
			}
		}
	}
}

#endif