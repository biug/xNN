#ifndef DROP_LAYER_H_
#define DROP_LAYER_H_

#include <vector>
#include <random>

using std::vector;

template<typename DType, template <typename> class Neuron>
class DropLayer {
	double m_dRate;
	std::default_random_engine m_Engine;
	std::uniform_real_distribution<double> m_Distribution;
public:
	DropLayer(double rate) : m_dRate(rate), m_Engine((unsigned int)std::time(nullptr)), m_Distribution(0.0, 1.0) {}
	~DropLayer() = default;

	inline void drop(vector<Neuron<DType> *> neurons) {
		// neuron.active = Drop(down.output)
		double roll;
		for (Neuron<DType> * neuron : neurons) {
			for (int i = 0, n = neuron->getVecLen(); i < n; ++i) {
				roll = m_Distribution(m_Engine);
				if (roll < m_dRate) {
					neuron->getMutableActive()[i] = static_cast<DType>(0);
				}
			}
		}
	}
};

#endif