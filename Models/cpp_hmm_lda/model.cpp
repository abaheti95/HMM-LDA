#include "model.h"

using namespace std;

void print_vector(vector<int> &v) {
	int size = v.size();
	for(int i = 0; i < size; i++) {
		cout << v[i] << " ";
	}
	cout << endl;
}

// Real number generator between 0 and 1
random_device rd;
default_random_engine generator(rd());
uniform_real_distribution<double> distribution(0.0,1.0);
int categorical(vector<double> proportions) {
	// Sum all the elements and draw a random number for that
	double sum = 0;
	for(vector<double>::iterator it = proportions.begin(); it != proportions.end(); it++)
		sum += *it;
	double random_number = distribution(generator);
	double draw = sum * random_number;
	// Return the appropriate index based on the cumulative sum
	double cumsum = 0;
	int size = proportions.size();
	for(int idx = 0; idx < size; idx++) {
		cumsum += proportions[idx];
		if(draw < cumsum) {
			return idx;
		}
	}
	return size - 1;
}

HiddenMarkovModelLatentDirichletAllocation::HiddenMarkovModelLatentDirichletAllocation(int vocab, int topics, 
	int classes, double n_alpha, double n_beta, double n_gamma, double n_delta): vocab_size(vocab), 
	num_topics(topics), num_classes(classes), alpha(n_alpha), beta(n_beta), gamma(n_gamma), delta(n_delta), 
	rng(rd()), topic_dist(0,topics-1), class_dist(0, classes-1) {
	// All 2D vectors are empty at the moment
}

HiddenMarkovModelLatentDirichletAllocation::~HiddenMarkovModelLatentDirichletAllocation() {
	// Free up the 2d vectors
	int n_documents = documents.size();
	for(int document_idx = 0; document_idx < n_documents; document_idx++){
		documents[document_idx].clear();
		class_assignments[document_idx].clear();
		topic_assignments[document_idx].clear();
	}
}

void HiddenMarkovModelLatentDirichletAllocation::add_document(vector<int> document) {
	documents.push_back(document);
	// Sample random topics and classes for the words
	vector<int> topic_assignment(document.size(), 0);
	vector<int> class_assignment(document.size(), 0);
	int doc_size = document.size();
	for(int i = 0; i < doc_size; i++) {
		topic_assignment[i] = topic_dist(rng);
		class_assignment[i] = class_dist(rng);
	}
	// cout << "Printing topic assignments of a document" << endl;
	// print_vector(topic_assignment);
	topic_assignments.push_back(topic_assignment);
	class_assignments.push_back(class_assignment);
}

void HiddenMarkovModelLatentDirichletAllocation::run_counts() {
	int n_documents = documents.size();
	//num_words_in_doc_assigned_to_topic
	// dimension of this 2d vector will be n_documents * num_topics
	num_words_in_doc_assigned_to_topic.resize(n_documents, vector<int>(num_topics, 0));
	//num_same_words_assigned_to_topic
	// dimension of this 2d vector will be vocab_size * num_topics
	num_same_words_assigned_to_topic.resize(vocab_size, vector<int>(num_topics, 0));
	//num_words_assigned_to_topic
	// dimension of this 1d vector will be num_topics
	num_words_assigned_to_topic.resize(num_topics, 0);
	//num_same_words_assigned_to_class
	// dimension of this 2d vector will be vocab_size * num_classes
	num_same_words_assigned_to_class.resize(vocab_size, vector<int>(num_classes, 0));
	//num_words_assigned_to_class
	// dimension of this 1d vector will be num_classes
	num_words_assigned_to_class.resize(num_classes, 0);
	//num_transitions
	// dimension of this 2d vector will be num_classes * num_classes
	num_transitions.resize(num_classes, vector<int>(num_classes, 0));
	
	for(int document_idx = 0; document_idx < n_documents; document_idx++) {
		int doc_size = documents[document_idx].size();
		int previous_class = -1;
		for(int word_idx = 0; word_idx < doc_size; word_idx++) {
			int word = documents[document_idx][word_idx];
			if(class_assignments[document_idx][word_idx] == 0)		// Count the topic only when the class is 0
				num_words_in_doc_assigned_to_topic[document_idx][topic_assignments[document_idx][word_idx]] += 1;
				num_same_words_assigned_to_topic[word][topic_assignments[document_idx][word_idx]] += 1;
				num_words_assigned_to_topic[topic_assignments[document_idx][word_idx]]++;
			num_same_words_assigned_to_class[word][class_assignments[document_idx][word_idx]] += 1;
			num_words_assigned_to_class[class_assignments[document_idx][word_idx]]++;
			int current_class = class_assignments[document_idx][word_idx];
			if(previous_class != -1)
				num_transitions[previous_class][current_class]++;
			previous_class = current_class;
		}
	}
}
void HiddenMarkovModelLatentDirichletAllocation::train(int iterations, int save_freq) {
	save_assignments(0);
	// Run Gibbs sampling
	// iterations = number of full samplings of the corpus
	int n_documents = documents.size();
	for(int i = 1; i <= iterations; i++) {
		std::clock_t start = std::clock();
		for(int document_idx = 0; document_idx < n_documents; document_idx++) {
			int doc_size = documents[document_idx].size();
			for(int word_idx = 0; word_idx < doc_size; word_idx++) {
				draw_class(document_idx, word_idx, doc_size);
				draw_topic(document_idx, word_idx, doc_size);
			}
		}
		std::cout << "Time for iteration " << i <<": " << (std::clock() - start) / (double)(CLOCKS_PER_SEC) << " s" << std::endl;
		if((i%save_freq) == 0)
			save_assignments(i);
	}
}

void HiddenMarkovModelLatentDirichletAllocation::draw_class(int document_idx, int word_idx, int doc_size) {
	int old_class = class_assignments[document_idx][word_idx];
	int old_topic = topic_assignments[document_idx][word_idx];
	int word = documents[document_idx][word_idx];

	// Get neighboring classes
	int previous = -1;
	if(word_idx > 0)
		previous = class_assignments[document_idx][word_idx - 1];
	int future = -1;
	if(word_idx < doc_size-1)
		future = class_assignments[document_idx][word_idx + 1];
	
	// Build first term of numerator
	vector<double> term_1(num_classes, 0.0);
	if(previous != -1) {
		for(int i = 0; i < num_classes; i++)
			term_1[i] = num_transitions[previous][i] + gamma;
		term_1[old_class] -= 1.0;							// Exclude current word
	}

	// Build second term of numerator
	vector<double> term_2(num_classes, 0.0);
	if(future != -1) {
		for(int i = 0; i < num_classes; i++)
			term_2[i] = num_transitions[i][future] + gamma;
		term_2[old_class] -= 1.0; 							// Exclude current word
	}
	if(previous != -1 && future != -1 && (previous == future))
		term_2[previous] += 1.0;

	// Calculate numerator
	vector<double> numerator(num_classes, 0.0);
	for(int i = 0; i < num_classes; i++)
		numerator[i] = term_1[i] * term_2[i];

	// Build denominator
	vector<double> denominator(num_words_assigned_to_class.begin(), num_words_assigned_to_class.end());
	if(previous != -1)
		denominator[previous] += 1.0;
	for(int i = 0; i < num_classes; i++)
		denominator[i] += num_classes * gamma;
	
	// Calculate multiplier
	
	// Initialize numerator of multiplier with same word class/topic counts
	vector<double> multiplier_numerator(num_same_words_assigned_to_class[word].begin(), 
		num_same_words_assigned_to_class[word].end());
	multiplier_numerator[0] = num_same_words_assigned_to_topic[word][old_topic];
	// Exclude current word
	if(old_class != 0)
		multiplier_numerator[old_class] -= 1.0;
	multiplier_numerator[0] -= 1.0;
	// Smoothing
	for(int i = 0; i < num_classes; i++) {
		if(i == 0)
			multiplier_numerator[0] += beta;
		else
			multiplier_numerator[i] += delta;
	}

	// Initialize denominator of multiplier with global class/topic counts
	vector<double> multiplier_denominator(num_words_assigned_to_class.begin(), num_words_assigned_to_class.end());
	multiplier_denominator[0] = num_words_assigned_to_topic[old_topic];
	// Exclude current word
	if(old_class != 0)
		multiplier_denominator[old_class] -= 1.0;
	multiplier_denominator[0] -= 1.0;
	// Smoothing
	for(int i = 0; i < num_classes; i++) {
		if(i == 0)
			multiplier_denominator[0] += beta * vocab_size;
		else
			multiplier_denominator[i] += delta * vocab_size;
	}

	// Calculate probability proportions
	vector<double> proportions(num_classes, 0.0);
	for(int i = 0; i < num_classes; i++) {
		proportions[i] = (multiplier_numerator[i] / multiplier_denominator[i]) * numerator[i] / denominator[i];
	}

	// Draw class
	// logging.info('proportions = %s', proportions)
	int new_class = categorical(proportions);
	// logging.info('drew class %d', new_class)
	class_assignments[document_idx][word_idx] = new_class;

	// Correct counts
	if(previous != -1) {
		num_transitions[previous][old_class] -= 1;
		num_transitions[previous][new_class] += 1;
	}
	if(future != -1) {
		num_transitions[old_class][future] -= 1;
		num_transitions[new_class][future] += 1;
	}

	num_same_words_assigned_to_class[word][old_class] -= 1;
	num_same_words_assigned_to_class[word][new_class] += 1;

	num_words_assigned_to_class[old_class] -= 1;
	num_words_assigned_to_class[new_class] += 1;

	if(old_class == 0 && new_class != 0) {
		num_words_in_doc_assigned_to_topic[document_idx][old_topic] -= 1;
		num_same_words_assigned_to_topic[word][old_topic] -= 1;
		num_words_assigned_to_topic[old_topic] -= 1;
	} else if(old_class != 0 && new_class == 0) {
		num_words_in_doc_assigned_to_topic[document_idx][old_topic] += 1;
		num_same_words_assigned_to_topic[word][old_topic] += 1;
		num_words_assigned_to_topic[old_topic] += 1;
	}
}

void HiddenMarkovModelLatentDirichletAllocation::draw_topic(int document_idx, int word_idx, int doc_size) {
	int old_topic = topic_assignments[document_idx][word_idx];
	int old_class = class_assignments[document_idx][word_idx];
	int word = documents[document_idx][word_idx];

	// Initialize probability proportions with document topic counts
	vector<double> proportions(num_words_in_doc_assigned_to_topic[document_idx].begin(), 
		num_words_in_doc_assigned_to_topic[document_idx].end());

	// Exclude current word
	if(old_class == 0)
		proportions[old_topic] -= 1.0;

	// Smoothing
	for(int i = 0; i < num_topics; i++)
		proportions[i] += alpha;

	// If the current word is assigned to the semantic class
	if(old_class == 0) {
		// Initialize numerator with same word topic counts
		vector<double> numerator(num_same_words_assigned_to_topic[word].begin(),
			num_same_words_assigned_to_topic[word].end());
		
		// Initialize denominator with global topic counts
		vector<double> denominator(num_words_assigned_to_topic.begin(), num_words_assigned_to_topic.end());

		// Exclude current word
		numerator[old_topic] -= 1.0;
		denominator[old_topic] -= 1.0;

		// Smoothing
		for(int i = 0; i < num_topics; i++) {
			numerator[i] += beta;
			denominator[i] += vocab_size * beta;
			// Apply multiplier
			proportions[i] = proportions[i] * numerator[i] / denominator[i];
		}
	}
	
	// Draw topic
	// logging.info('proportions = %s', proportions)
	int new_topic = categorical(proportions);
	// logging.info('drew topic %d', new_topic)
	topic_assignments[document_idx][word_idx] = new_topic;

	// Correct counts
	if(old_class == 0) {
		num_words_in_doc_assigned_to_topic[document_idx][old_topic] -= 1 ;
		num_words_in_doc_assigned_to_topic[document_idx][new_topic] += 1;

		num_same_words_assigned_to_topic[word][old_topic] -= 1;
		num_same_words_assigned_to_topic[word][new_topic] += 1;

		num_words_assigned_to_topic[old_topic] -= 1;
		num_words_assigned_to_topic[new_topic] += 1;
	}
}

void HiddenMarkovModelLatentDirichletAllocation::save_assignments(int iteration) {
	cout << "Saving assignments for iteration " << iteration << endl;
	int n_documents = documents.size();
	// Create a new directory and save 2 files for topic assignments and class assignments
	string save_dir = "Results/" + to_string(iteration) + "_results";
	system(("mkdir -p " + save_dir).c_str());
	cout << "Directory :" << save_dir << " created" << endl;
	ofstream class_file((save_dir + "/class_assignments.txt").c_str());
	class_file << n_documents << endl;
	for(int document_idx = 0; document_idx < n_documents; document_idx++) {
		int doc_size = class_assignments[document_idx].size();
		for(int word_idx = 0; word_idx < doc_size; word_idx++)
			class_file << class_assignments[document_idx][word_idx] << " ";
		class_file << endl;
	}
	class_file.close();
	ofstream topic_file((save_dir + "/topic_assignments.txt").c_str());
	topic_file << n_documents << endl;
	for(int document_idx = 0; document_idx < n_documents; document_idx++) {
		int doc_size = topic_assignments[document_idx].size();
		for(int word_idx = 0; word_idx < doc_size; word_idx++)
			topic_file << topic_assignments[document_idx][word_idx] << " ";
		topic_file << endl;
	}
	topic_file.close();
}

// int main(int argc, char const *argv[])
// {
// 	HiddenMarkovModelLatentDirichletAllocation model(10, 100, 20, 0.1, 0.1, 0.1, 0.1);
// 	return 0;
// }

