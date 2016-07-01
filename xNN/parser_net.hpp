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
#include <iostream>
#include <unordered_map>

#include "input_layer.hpp"
#include "hidden_layer.hpp"
#include "loss_layer.hpp"

#include "random_generator.hpp"

using std::vector;
using std::string;
using std::ifstream;
using std::stringstream;
using std::unordered_map;

template<typename DType, template <typename> class Activation, template <typename> class PartialActivation, template <typename> class Loss, template <typename> class PartialLoss>
class ParserNet {
	const DType g_dMomentum = static_cast<DType>(0.99 - 1e-8);
	const DType g_dLearningRate = static_cast<DType>(0.001);

	vector<InputNeuron<DType> *>	m_vecInputNeurons;
	vector<vector<HiddenNeuron<DType> *>> m_vecsHiddenNeurons;

	InputLayer<DType> m_lyrInputLayer;
	HiddenLayer<DType, Activation, PartialActivation> m_lyrHiddenLayers;
	LossLayer<DType, Loss, PartialLoss> m_lyrLossLayer;

	int m_nEmbeddingLen, m_nWordNum, m_nPOSNum, m_nLabelNum;
	DType *m_pWordMatrix, *m_pPOSMatrix, *m_pLabelMatrix;
	DType *m_pWordMatrixDiff, *m_pPOSMatrixDiff, *m_pLabelMatrixDiff;
	unordered_map<string, int> m_mapWordOrder, m_mapPOSOrder, m_mapLabelOrder;
public:
	ParserNet(int embedding_num, const vector<vector<int>> & neuron_lens, const string & embedding_file, RandomGenerator<DType> * generator);
	~ParserNet();

	void train(const vector<string> & batchFiles, int max_iter, DType threshold);
};

// definitions

template<typename DType, template <typename> class Activation, template <typename> class PartialActivation, template <typename> class Loss, template <typename> class PartialLoss>
ParserNet<DType, Activation, PartialActivation, Loss, PartialLoss>::ParserNet(int embedding_num, const vector<vector<int>> & neuron_lens, const string & embedding_file, RandomGenerator<DType> * generator) {
	// init input neurons
	for (int i = 0; i < neuron_lens.front().size(); ++i) {
		m_vecInputNeurons.push_back(new InputNeuron<DType>(embedding_num, neuron_lens[0][i]));
	}
	// init hidden neurons
	for (int i = 1; i < neuron_lens.size(); ++i) {
		vector<HiddenNeuron<DType> *> vecHiddenNeurons;
		for (int j = 0; j < neuron_lens[i].size(); ++j) {
			vecHiddenNeurons.push_back(new HiddenNeuron<DType>(neuron_lens[i][j], neuron_lens[i - 1], generator));
		}
		m_vecsHiddenNeurons.push_back(vecHiddenNeurons);
	}

	string line;
	ifstream ifs(embedding_file);
	// init embedding veclen
	ifs >> m_nEmbeddingLen;
	// init words
	ifs >> m_nWordNum;
	m_pWordMatrix = new DType[m_nWordNum * m_nEmbeddingLen];
	m_pWordMatrixDiff = new DType[m_nWordNum * m_nEmbeddingLen];
	for (int i = 0, offset = 0; i < m_nWordNum; ++i) {
		string word;
		ifs >> word;
		m_mapWordOrder[word] = i;
		for (int j = 0; j < m_nEmbeddingLen; ++j) {
			ifs >> m_pWordMatrix[offset++];
		}
	}
	// init postags
	ifs >> m_nPOSNum;
	m_pPOSMatrix = new DType[m_nPOSNum * m_nEmbeddingLen];
	m_pPOSMatrixDiff = new DType[m_nPOSNum * m_nEmbeddingLen];
	for (int i = 0, offset = 0; i < m_nPOSNum; ++i) {
		string pos;
		ifs >> pos;
		m_mapPOSOrder[pos] = i;
		for (int j = 0; j < m_nEmbeddingLen; ++j) {
			ifs >> m_pPOSMatrix[offset++];
		}
	}
	// init Labeltags
	ifs >> m_nLabelNum;
	m_pLabelMatrix = new DType[m_nLabelNum * m_nEmbeddingLen];
	m_pLabelMatrixDiff = new DType[m_nLabelNum * m_nEmbeddingLen];
	for (int i = 0, offset = 0; i < m_nLabelNum; ++i) {
		string label;
		ifs >> label;
		m_mapLabelOrder[label] = i;
		for (int j = 0; j < m_nEmbeddingLen; ++j) {
			ifs >> m_pLabelMatrix[offset++];
		}
	}
	std::cout << "net init complete" << std::endl;
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
	delete[] m_pWordMatrix;
	// free POSs
	delete[] m_pPOSMatrix;
	// free labels
	delete[] m_pLabelMatrix;
	// free maps
	m_mapWordOrder.clear();
	m_mapPOSOrder.clear();
	m_mapLabelOrder.clear();
}

template<typename DType, template <typename> class Activation, template <typename> class PartialActivation, template <typename> class Loss, template <typename> class PartialLoss>
void ParserNet<DType, Activation, PartialActivation, Loss, PartialLoss>::train(const vector<string> & batch_files, int max_iter, DType threshold) {
	// read one batch
	size_t hiddenSize = m_vecsHiddenNeurons.size();
	for (const auto & batchFile : batch_files) {
		for (int iter = 0; iter < max_iter; ++iter) {
			ifstream ifs(batchFile);
			string line;
			stringstream ss;
			int id, size;
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
			while (true) {
				int correctLabel;
				vector<int> words, poses, labels;
				getline(ifs, line);
				if (line.empty()) break;
				ss << line;
				while (ss >> id) {
					words.push_back(id);
				}
				ss.clear();
				getline(ifs, line);
				ss << line;
				while (ss >> id) {
					poses.push_back(id);
				}
				ss.clear();
				getline(ifs, line);
				ss << line;
				while (ss >> id) {
					labels.push_back(id);
				}
				ss.clear();
				getline(ifs, line);
				ss << line;
				ss >> correctLabel;
				// load embeddings
				m_vecInputNeurons[0]->loadEmbedding((const DType*)m_pWordMatrix, words);
				m_vecInputNeurons[1]->loadEmbedding((const DType*)m_pPOSMatrix, poses);
				m_vecInputNeurons[2]->loadEmbedding((const DType*)m_pLabelMatrix, labels);
				// foreward
				m_lyrInputLayer.foreward(m_vecsHiddenNeurons[0], m_vecInputNeurons);
				for (size_t i = 1; i < hiddenSize; ++i) {
					m_lyrHiddenLayers.foreward(m_vecsHiddenNeurons[i], m_vecsHiddenNeurons[i - 1]);
				}
				m_lyrLossLayer.foreward(m_vecsHiddenNeurons.back().back());
				std::cout << "loss vector len is " << m_vecsHiddenNeurons.back().back()->getVecLen() << std::endl;
				for (size_t i = 0; i < m_vecsHiddenNeurons.back().back()->getVecLen(); ++i) {
					std::cout << m_vecsHiddenNeurons.back().back()->getActive()[i] << ' ';
				}
				std::cout << std::endl;
				// backward
				m_lyrLossLayer.backward(m_vecsHiddenNeurons.back().back(), correctLabel);
				std::cout << "partial loss vector len is " << m_vecsHiddenNeurons.back().back()->getVecLen() << std::endl;
				for (size_t i = 0; i < m_vecsHiddenNeurons.back().back()->getVecLen(); ++i) {
					std::cout << m_vecsHiddenNeurons.back().back()->getOutputDiff()[i] << ' ';
				}
				std::cout << std::endl;
				for (size_t i = hiddenSize - 1; i > 0; --i) {
					m_lyrHiddenLayers.backward(m_vecsHiddenNeurons[i], m_vecsHiddenNeurons[i - 1]);
				}
				m_lyrInputLayer.backward(m_vecsHiddenNeurons[0], m_vecInputNeurons);
				m_vecInputNeurons[0]->updateEmbedding(m_pWordMatrixDiff, words);
				m_vecInputNeurons[1]->updateEmbedding(m_pPOSMatrixDiff, poses);
				m_vecInputNeurons[2]->updateEmbedding(m_pLabelMatrixDiff, labels);
			}
			// update batch
			for (size_t i = hiddenSize - 1; i > 0; --i) {
				m_lyrHiddenLayers.update(m_vecsHiddenNeurons[i], m_vecsHiddenNeurons[i - 1].size(), g_dMomentum, g_dLearningRate);
			}
			m_lyrInputLayer.update(m_vecsHiddenNeurons.front(), m_vecInputNeurons.size(), g_dMomentum, g_dLearningRate);
			DType norm = 0;
			size = m_nWordNum * m_nEmbeddingLen;
			// word = (-learning_rate) * word_diff + momentum * word
			alpha_vector_add_beta_vector(m_pWordMatrix, m_pWordMatrixDiff, -g_dLearningRate, g_dMomentum, size);
			for (int i = 0; i < size; ++i) {
				norm += std::abs(m_pWordMatrixDiff[i]);
			}
			size = m_nPOSNum * m_nEmbeddingLen;
			// pos = (-learning_rate) * pos_diff + momentum * pos
			alpha_vector_add_beta_vector(m_pPOSMatrix, m_pPOSMatrixDiff, -g_dLearningRate, g_dMomentum, size);
			for (int i = 0; i < size; ++i) {
				norm += std::abs(m_pPOSMatrixDiff[i]);
			}
			size = m_nLabelNum * m_nEmbeddingLen;
			// label = (-learning_rate) * label_diff + momentum * label
			alpha_vector_add_beta_vector(m_pLabelMatrix, m_pLabelMatrixDiff, -g_dLearningRate, g_dMomentum, size);
			for (int i = 0; i < size; ++i) {
				norm += std::abs(m_pLabelMatrixDiff[i]);
			}
			for (const auto & neuron : m_vecsHiddenNeurons[0]) {
				norm += neuron->norm1(m_vecInputNeurons.size());
			}
			for (int i = 1; i < m_vecsHiddenNeurons.size(); ++i) {
				for (const auto & neuron : m_vecsHiddenNeurons[i]) {
					norm += neuron->norm1(m_vecsHiddenNeurons[i - 1].size());
				}
			}
			std::cout << "iterator is " << iter << " norm is " << norm << std::endl;
			if (norm < threshold) {
				break;
			}
		}
	}
}

#endif