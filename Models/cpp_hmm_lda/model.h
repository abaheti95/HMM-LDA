#ifndef MODEL_H
#define MODEL_H

#include <cstdlib>
#include <cstdio>
#include <ctime>

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <map>
#include <random>

using namespace std;

class HiddenMarkovModelLatentDirichletAllocation {
public:
	// Data variables
	int vocab_size, num_topics, num_classes;
	double alpha, beta, gamma, delta;
	vector<vector<int> > documents, topic_assignments, class_assignments, num_words_in_doc_assigned_to_topic,
		num_same_words_assigned_to_topic, num_same_words_assigned_to_class, num_transitions;
	vector<int> num_words_assigned_to_topic, num_words_assigned_to_class;
	HiddenMarkovModelLatentDirichletAllocation(int vocab, int topcis, int classes, double n_alpha, double n_beta, double n_gamma, double n_delta);
	~HiddenMarkovModelLatentDirichletAllocation();

	void add_document(vector<int> document);
	void run_counts();
	void train(int iterations, int save_freq);
	void draw_class(int document_idx, int word_idx, int doc_size);
	void draw_topic(int document_idx, int word_idx, int doc_size);
	void save_assignments(int iteration);
private:
	mt19937 rng;			// random-number engine used (Mersenne-Twister in this case)
	uniform_int_distribution<int> topic_dist;		// guaranteed unbiased
	uniform_int_distribution<int> class_dist;
};

#endif