#ifndef MODEL_H
#define MODEL_H

#include <cstdlib>
#include <cstdio>
#include <ctime>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <map>
#include <random>

using namespace std;
class HMMLDA {
public:
	// Data variables
	int vocab_size, num_topics, num_classes;
	int start_word_id, end_word_id;
	int START_WORD_CLASS, END_WORD_CLASS;
	double alpha, beta, gamma, delta;
	string save_dir;
	vector<vector<int> > documents, topic_assignments, class_assignments, num_words_in_doc_assigned_to_topic,
		num_same_words_assigned_to_topic, num_same_words_assigned_to_class, num_transitions;
	vector<int> num_words_assigned_to_topic, num_words_assigned_to_class;
	HMMLDA(int vocab, int topcis, int classes, int start_word_id, int end_word_id, double n_alpha, double n_beta, double n_gamma, double n_delta, string save_directory);
	~HMMLDA();

	void add_document(vector<int> document);
	void load_class_assignments(string &class_file);
	void load_topic_assignments(string &topic_file);
	void run_counts();
	void add_counts_for_new_document(vector<int> &document, vector<int> &topic_assignment, vector<int> &class_assignment, int doc_size);
	void remove_counts_for_new_document(vector<int> &document, vector<int> &topic_assignment, vector<int> &class_assignment, int doc_size);
	void train(int iterations, int save_freq);
	void draw_class(int document_idx, int word_idx, int doc_size);
	void draw_topic(int document_idx, int word_idx, int doc_size);
	void relabel_document(vector<int> &document, vector<int> &topic_assignment, vector<int> &class_assignment, int doc_size, int iterations);
	void save_assignments(int iteration);
private:
	mt19937 rng;			// random-number engine used (Mersenne-Twister in this case)
	uniform_int_distribution<int> topic_dist;		// guaranteed unbiased
	uniform_int_distribution<int> class_dist;
};

// Other helper functions
template<typename Out>
void split(const std::string &s, char delim, Out result);
std::vector<std::string> split(const std::string &s, char delim);
int read_vocabulary(string &vocab_filename, vector<string> &vocab);

#endif