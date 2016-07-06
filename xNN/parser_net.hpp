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
#include <unordered_map>

#include "macros.h"
#include "input_layer.hpp"
#include "hidden_layer.hpp"
#include "loss_layer.hpp"

#include "random_generator.hpp"

using std::vector;
using std::string;
using std::ifstream;
using std::stringstream;
using std::unordered_map;

template<typename DType, template <typename> class Activation, template <typename> class PartialActivation, template <typename> class Loss, template <typename> class PartialLoss, template <typename Type, template <typename> class Neuron> class Updator>
class ParserNet {
	int m_nBatchSize;
	DType m_dLastValue;
	size_t m_nHiddenSize;

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
	vector<string> m_vecWords, m_vecPOSes, m_vecLabels;
	unordered_map<string, int> m_mapWordOrder, m_mapPOSOrder, m_mapLabelOrder;

	int m_nCorrectLabel;
	vector<int> m_vecBatchWords, m_vecBatchPOSes, m_vecBatchLabels;

	bool readOneAction(std::ifstream & ifs);
	void updateEmbeddingDiffs();

	void initBatch();
	void foreward();
	void backward();
	void update();
public:
	ParserNet(int embedding_num, const vector<vector<int>> & neuron_lens, const string & embedding_file, RandomGenerator<DType> * generator);
	~ParserNet();

	void train(const vector<string> & batchFiles, int max_iter, DType threshold);

	void writeModel(const std::string & output);
	void writeEmbeddings(const std::string & output);
};

// definitions

template<typename DType, template <typename> class Activation, template <typename> class PartialActivation, template <typename> class Loss, template <typename> class PartialLoss, template <typename Type, template <typename> class Neuron> class Updator>
ParserNet<DType, Activation, PartialActivation, Loss, PartialLoss, Updator>::ParserNet(int embedding_num, const vector<vector<int>> & neuron_lens, const string & embedding_file, RandomGenerator<DType> * generator) {
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
		m_mapWordOrder[word] = i;
		m_vecWords.push_back(word);
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
		m_mapPOSOrder[pos] = i;
		m_vecPOSes.push_back(pos);
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
		m_mapLabelOrder[label] = i;
		m_vecLabels.push_back(label);
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
	// free maps
	m_mapWordOrder.clear();
	m_mapPOSOrder.clear();
	m_mapLabelOrder.clear();
}

template<typename DType, template <typename> class Activation, template <typename> class PartialActivation, template <typename> class Loss, template <typename> class PartialLoss, template <typename Type, template <typename> class Neuron> class Updator>
bool ParserNet<DType, Activation, PartialActivation, Loss, PartialLoss, Updator>::readOneAction(std::ifstream & ifs) {
	int id;
	string line;
	stringstream ss;
	getline(ifs, line);
	if (line.empty()) return false;
	// increase batch size
	++m_nBatchSize;
	// read words
	ss << line;
	m_vecBatchWords.clear();
	while (ss >> id) {
		m_vecBatchWords.push_back(id);
	}
	ss.clear();
	getline(ifs, line);
	// read poses
	ss << line;
	m_vecBatchPOSes.clear();
	while (ss >> id) {
		m_vecBatchPOSes.push_back(id);
	}
	ss.clear();
	getline(ifs, line);
	// read labels
	ss << line;
	m_vecBatchLabels.clear();
	while (ss >> id) {
		m_vecBatchLabels.push_back(id);
	}
	// load embeddings
	m_vecInputNeurons[0]->loadEmbedding((const DType*)m_pWordMatrix, m_vecBatchWords);
	m_vecInputNeurons[1]->loadEmbedding((const DType*)m_pPOSMatrix, m_vecBatchPOSes);
	m_vecInputNeurons[2]->loadEmbedding((const DType*)m_pLabelMatrix, m_vecBatchLabels);
	ss.clear();
	getline(ifs, line);
	// read tags
	ss << line;
	ss >> m_nCorrectLabel;
	return true;
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
	for (size_t i = 1; i < m_nHiddenSize; ++i) {
		m_lyrHiddenLayers.foreward(m_vecsHiddenNeurons[i], m_vecsHiddenNeurons[i - 1]);
	}
	m_lyrLossLayer.foreward(m_vecsHiddenNeurons.back().back());
}

template<typename DType, template <typename> class Activation, template <typename> class PartialActivation, template <typename> class Loss, template <typename> class PartialLoss, template <typename Type, template <typename> class Neuron> class Updator>
void ParserNet<DType, Activation, PartialActivation, Loss, PartialLoss, Updator>::backward() {
	// backward
	m_lyrLossLayer.backward(m_vecsHiddenNeurons.back().back(), m_nCorrectLabel);
	for (size_t i = m_nHiddenSize - 1; i > 0; --i) {
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
void ParserNet<DType, Activation, PartialActivation, Loss, PartialLoss, Updator>::train(const vector<string> & batch_files, int max_iter, DType threshold) {
	// read one batch
	for (const auto & batchFile : batch_files) {
		// zero norm
		m_dLastValue = DType();
		for (int iter = 0; iter < max_iter; ++iter) {
			double timeFore = 0, timeBack = 0;
			initBatch();
			ifstream ifs(batchFile);
			DType value = DType();
			while (readOneAction(ifs)) {
				clock_t s = clock();
				foreward();
				timeFore += clock() - s;
				s = clock();
				backward();
				timeBack += clock() - s;
				value += m_vecsHiddenNeurons.back().back()->getActive()[m_nCorrectLabel];
			}
			update();
			if (iter % 1 == 0) {
				std::cout << "value = " << value << std::endl;
				std::cout << "foreward use time : " << timeFore / (double)CLOCKS_PER_SEC << std::endl;
				std::cout << "backward use time : " << timeBack / (double)CLOCKS_PER_SEC << std::endl;
			}
			if (isnan(value) || std::abs(m_dLastValue - value) < threshold) {
				break;
			}
			m_dLastValue = value;
		}
	}
	writeModel("E:\\models");
	writeEmbeddings("E:\\newEmbeddings");
}

template<typename DType, template <typename> class Activation, template <typename> class PartialActivation, template <typename> class Loss, template <typename> class PartialLoss, template <typename Type, template <typename> class Neuron> class Updator>
void ParserNet<DType, Activation, PartialActivation, Loss, PartialLoss, Updator>::writeModel(const std::string & output) {
	std::ofstream ofs(output);
	for (size_t i = 0, n = m_vecsHiddenNeurons.size(); i < n; ++i) {
		for (size_t j = 0, m = m_vecsHiddenNeurons[i].size(); j < m; ++j) {
			ofs << *m_vecsHiddenNeurons[i][j] << std::endl;
		}
	}
}

template<typename DType, template <typename> class Activation, template <typename> class PartialActivation, template <typename> class Loss, template <typename> class PartialLoss, template <typename Type, template <typename> class Neuron> class Updator>
void ParserNet<DType, Activation, PartialActivation, Loss, PartialLoss, Updator>::writeEmbeddings(const std::string & output) {
	std::ofstream ofs(output);
	ofs << m_nEmbeddingLen << std::endl;
	ofs << m_nWordNum << std::endl;
	for (int i = 0, offset = 0; i < m_nWordNum; ++i) {
		ofs << m_vecWords[i] << std::endl;
		for (int j = 0; j < m_nEmbeddingLen; ++j) {
			ofs << m_pWordMatrix[offset++] << ' ';
		}
		ofs << std::endl;
	}
	ofs << m_nPOSNum << std::endl;
	for (int i = 0, offset = 0; i < m_nPOSNum; ++i) {
		ofs << m_vecPOSes[i] << std::endl;
		for (int j = 0; j < m_nEmbeddingLen; ++j) {
			ofs << m_pPOSMatrix[offset++] << ' ';
		}
		ofs << std::endl;
	}
	ofs << m_nLabelNum << std::endl;
	for (int i = 0, offset = 0; i < m_nLabelNum; ++i) {
		ofs << m_vecLabels[i] << std::endl;
		for (int j = 0; j < m_nEmbeddingLen; ++j) {
			ofs << m_pLabelMatrix[offset++] << ' ';
		}
		ofs << std::endl;
	}
}

#endif