#include "model.h"

using namespace std;

template<typename T>
void print_vector(vector<T> &v) {
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

HMMLDA::HMMLDA(int vocab, int topics, 
	int classes, int start_word, int end_word, double n_alpha, double n_beta, double n_gamma, double n_delta, string save_directory): 
	vocab_size(vocab), num_topics(topics), num_classes(classes), start_word_id(start_word), end_word_id(end_word), 
	alpha(n_alpha), beta(n_beta), gamma(n_gamma), delta(n_delta), save_dir(save_directory), rng(rd()), topic_dist(0,topics-1), 
	class_dist(0, classes-3) {
	// All 2D vectors are empty at the moment
	START_WORD_CLASS = num_classes - 2;
	END_WORD_CLASS = num_classes - 1;
}

HMMLDA::~HMMLDA() {
	// Free up the 2d vectors
	int n_documents = documents.size();
	for(int document_idx = 0; document_idx < n_documents; document_idx++){
		documents[document_idx].clear();
		class_assignments[document_idx].clear();
		topic_assignments[document_idx].clear();
	}
}

void HMMLDA::add_document(vector<int> document) {
	documents.push_back(document);
	// Sample random topics and classes for the words
	vector<int> topic_assignment(document.size(), 0);
	vector<int> class_assignment(document.size(), 0);
	int doc_size = document.size();
	for(int i = 0; i < doc_size; i++) {
		topic_assignment[i] = topic_dist(rng);
		if(document[i] == start_word_id)
			class_assignment[i] = START_WORD_CLASS;
		else if(document[i] ==  end_word_id)
			class_assignment[i] = END_WORD_CLASS;
		else
			class_assignment[i] = class_dist(rng);
	}
	// cout << "Printing topic assignments of a document" << endl;
	// print_vector<int>(topic_assignment);
	topic_assignments.push_back(topic_assignment);
	class_assignments.push_back(class_assignment);
}

void HMMLDA::load_class_assignments(string &class_file) {
	cout << "Loading class assingments from " << class_file << endl;string str;
	ifstream file(class_file, ios::in);
	// Check if file is open
	if(!file.is_open()) {
		cout << "Couldn't open file: " << class_file << endl;
		exit(1);
	}
	getline(file, str);			// remove first line
	int n_documents = documents.size();
	for(int i = 0; i < n_documents; i++) {
		getline(file, str);
		stringstream ss(str);
		int doc_size = class_assignments[i].size();
		for(int j = 0; j < doc_size; j++) {
			int num;
			ss >> num;
			class_assignments[i][j] = num;
		}
	}
}

void HMMLDA::load_topic_assignments(string &topic_file) {
	cout << "Loading topic assingments from " << topic_file << endl;string str;
	ifstream file(topic_file, ios::in);
	// Check if file is open
	if(!file.is_open()) {
		cout << "Couldn't open file: " << topic_file << endl;
		exit(1);
	}
	getline(file, str);			// remove first line
	int n_documents = documents.size();
	for(int i = 0; i < n_documents; i++) {
		getline(file, str);
		stringstream ss(str);
		int doc_size = topic_assignments[i].size();
		for(int j = 0; j < doc_size; j++) {
			int num;
			ss >> num;
			topic_assignments[i][j] = num;
		}
	}
}

void HMMLDA::run_counts() {
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
			if(class_assignments[document_idx][word_idx] == 0) {		// Count the topic only when the class is 0
				num_words_in_doc_assigned_to_topic[document_idx][topic_assignments[document_idx][word_idx]]++;
				num_same_words_assigned_to_topic[word][topic_assignments[document_idx][word_idx]]++;
				num_words_assigned_to_topic[topic_assignments[document_idx][word_idx]]++;
			}
			num_same_words_assigned_to_class[word][class_assignments[document_idx][word_idx]]++;
			num_words_assigned_to_class[class_assignments[document_idx][word_idx]]++;
			int current_class = class_assignments[document_idx][word_idx];
			if(previous_class != -1)
				num_transitions[previous_class][current_class]++;
			previous_class = current_class;
		}
	}
}


void HMMLDA::train(int iterations, int save_freq) {
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
		std::cout << "Time for iteration " << i <<": " << (std::clock() - start) / (double)(CLOCKS_PER_SEC) << " s" << endl;
		if((i%save_freq) == 0)
			save_assignments(i);
		// cout << endl << endl << endl << endl;
	}
}

void HMMLDA::draw_class(int document_idx, int word_idx, int doc_size) {
	int old_class = class_assignments[document_idx][word_idx];
	int old_topic = topic_assignments[document_idx][word_idx];
	int word = documents[document_idx][word_idx];
	if(word == start_word_id || word == end_word_id) 
		return;
	// Get neighboring classes
	int previous = -1;
	if(word_idx > 0)
		previous = class_assignments[document_idx][word_idx - 1];
	int future = -1;
	if(word_idx < doc_size-1)
		future = class_assignments[document_idx][word_idx + 1];

	// Remove current word from transition counts
	if(previous != -1)
		num_transitions[previous][old_class]--;
	if(future != -1)
		num_transitions[old_class][future]--;
	if(old_class == 0) {
		// Remove word from topic counts
		num_same_words_assigned_to_topic[word][old_topic]--;
		num_words_assigned_to_topic[old_topic]--;
		num_words_in_doc_assigned_to_topic[document_idx][old_topic]--;
	}
	// Remove the word from class counts
	num_same_words_assigned_to_class[word][old_class]--;
	num_words_assigned_to_class[old_class]--;

	// Build first term of numerator
	vector<double> term_1(num_classes, 0.0);
	if(previous != -1) {
		for(int i = 0; i < num_classes; i++)
			term_1[i] = num_transitions[previous][i];
	}
	//Smoothing
	for(int i = 0; i < num_classes; i++)
		term_1[i] += gamma;

	// Build second term of numerator
	vector<double> term_2(num_classes, 0.0);
	if(future != -1) {
		for(int i = 0; i < num_classes; i++)
			term_2[i] = num_transitions[i][future];
	}
	//Smoothing
	for(int i = 0; i < num_classes; i++)
		term_2[i] += gamma;
	// Adjusting for the itentity value in the numerator term 2
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
	// if(document_idx == 0) {
	// 	if(new_class == 0) {
	// 		cout << "Printing proportions for class sampling" << document_idx << " " << word_idx << endl;
	// 		print_vector<double>(proportions);
	// 		print_vector<double>(numerator);
	// 		print_vector<double>(multiplier_numerator);
	// 		cout << old_class << " New class chosen " << new_class << endl;
	// 		cout << alpha << " " << beta << " " << gamma << " " << delta << endl; 
	// 	}
	// }
	// logging.info('drew class %d', new_class)
	class_assignments[document_idx][word_idx] = new_class;

	// Restore counts counts
	if(previous != -1) {
		num_transitions[previous][new_class] += 1;
	}
	if(future != -1) {
		num_transitions[new_class][future] += 1;
	}
	if(new_class == 0) {
		// Add word to the topic counts
		num_words_in_doc_assigned_to_topic[document_idx][old_topic] += 1;
		num_same_words_assigned_to_topic[word][old_topic] += 1;
		num_words_assigned_to_topic[old_topic] += 1;
	}
	// Add word to the new class counts
	num_same_words_assigned_to_class[word][new_class] += 1;
	num_words_assigned_to_class[new_class] += 1;
}

void HMMLDA::draw_topic(int document_idx, int word_idx, int doc_size) {
	int old_topic = topic_assignments[document_idx][word_idx];
	int old_class = class_assignments[document_idx][word_idx];
	int word = documents[document_idx][word_idx];

	// Exclude current word from topic counts
	if(old_class == 0) {
		num_words_in_doc_assigned_to_topic[document_idx][old_topic]--;
		num_same_words_assigned_to_topic[word][old_topic]--;
		num_words_assigned_to_topic[old_topic]--;
	}


	// Initialize probability proportions with document topic counts
	vector<double> proportions(num_words_in_doc_assigned_to_topic[document_idx].begin(), 
		num_words_in_doc_assigned_to_topic[document_idx].end());

	// Smoothing
	for(int i = 0; i < num_topics; i++)
		proportions[i] += alpha;

	// If the current word is assigned to the semantic class
	// TODO: I have commented next line. Remove it later
	if(old_class == 0) {
		// Initialize numerator with same word topic counts
		vector<double> numerator(num_same_words_assigned_to_topic[word].begin(),
			num_same_words_assigned_to_topic[word].end());
		
		// Initialize denominator with global topic counts
		vector<double> denominator(num_words_assigned_to_topic.begin(), num_words_assigned_to_topic.end());

		for(int i = 0; i < num_topics; i++) {
			// Smoothing
			numerator[i] += beta;
			denominator[i] += vocab_size * beta;
			// Apply multiplier
			proportions[i] = proportions[i] * numerator[i] / denominator[i];
		}
	// TODO: I have commented next line. Remove it later
	}
	
	// Draw topic
	// logging.info('proportions = %s', proportions)
	int new_topic = categorical(proportions);
	// logging.info('drew topic %d', new_topic)
	topic_assignments[document_idx][word_idx] = new_topic;

	// Correct counts with new topic
	if(old_class == 0) {
		num_words_in_doc_assigned_to_topic[document_idx][new_topic] += 1;
		num_same_words_assigned_to_topic[word][new_topic] += 1;
		num_words_assigned_to_topic[new_topic] += 1;
	}
}

void HMMLDA::save_assignments(int iteration) {
	cout << "Saving assignments for iteration " << iteration << endl;
	int n_documents = documents.size();
	// Create a new directory and save 2 files for topic assignments and class assignments
	string save_dir_path = "Results/" + save_dir + "/" + to_string(iteration) + "_results";
	system(("mkdir -p " + save_dir_path).c_str());
	cout << "Directory :" << save_dir_path << " created" << endl;
	ofstream class_file((save_dir_path + "/class_assignments.txt").c_str());
	class_file << n_documents << " " << num_classes << endl;
	for(int document_idx = 0; document_idx < n_documents; document_idx++) {
		int doc_size = class_assignments[document_idx].size();
		for(int word_idx = 0; word_idx < doc_size; word_idx++)
			class_file << class_assignments[document_idx][word_idx] << " ";
		class_file << endl;
	}
	class_file.close();
	ofstream topic_file((save_dir_path + "/topic_assignments.txt").c_str());
	topic_file << n_documents << " " << num_topics << endl;
	for(int document_idx = 0; document_idx < n_documents; document_idx++) {
		int doc_size = topic_assignments[document_idx].size();
		for(int word_idx = 0; word_idx < doc_size; word_idx++)
			topic_file << topic_assignments[document_idx][word_idx] << " ";
		topic_file << endl;
	}
	topic_file.close();
}


/////////////////////////////////////////////////////////////
///////////// New Document Labeling Functions ///////////////
/////////////////////////////////////////////////////////////
/* The functions below are for the case when you encounter a new text document (not present in training data)
	and you want to get the topic and class labels of that document without retraining the entire model

	Here we simply initialize the topic and class assignments of the new document randomly, update the counts
	and re-run the gibbs sampling only for that document for couple of iterations. After that we remove the
	counts of the assignments of the new document so that our model returns to the initial state
	Ref: Section 3.3 http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.366.8625&rep=rep1&type=pdf

	NOTE: A lot of code below is similar to the code used in training but I didn't care to optimize the structure too much lest it becomes unreadable
*/
// For an unseen document we will add its topic and class assignment temporarily to the total counts
void HMMLDA::add_counts_for_new_document(vector<int> &document, vector<int> &topic_assignment, vector<int> &class_assignment, int doc_size) {
	int previous_class = -1;
	for(int word_idx = 0; word_idx < doc_size; word_idx++) {
		int word = document[word_idx];
		if(class_assignment[word_idx] == 0) {		// Count the topic only when the class is 0
			//TODO: Think about how to implement num_words_in_doc_assigned_to_topic for a new document
			num_words_in_doc_assigned_to_topic[document_idx][topic_assignment[word_idx]]++;
			num_same_words_assigned_to_topic[word][topic_assignment[word_idx]]++;
			num_words_assigned_to_topic[topic_assignment[word_idx]]++;
		}
		num_same_words_assigned_to_class[word][class_assignment[word_idx]]++;
		num_words_assigned_to_class[class_assignment[word_idx]]++;
		int current_class = class_assignment[word_idx];
		if(previous_class != -1)
			num_transitions[previous_class][current_class]++;
		previous_class = current_class;
	}
}

// For an unseen document we will remove its topic and class assignment temporarily to the total counts
void HMMLDA::remove_counts_for_new_document(vector<int> &document, vector<int> &topic_assignment, vector<int> &class_assignment, int doc_size) {
	int previous_class = -1;
	for(int word_idx = 0; word_idx < doc_size; word_idx++) {
		int word = document[word_idx];
		if(class_assignment[word_idx] == 0) {		// Count the topic only when the class is 0
			num_words_in_doc_assigned_to_topic[document_idx][topic_assignment[word_idx]]--;
			num_same_words_assigned_to_topic[word][topic_assignment[word_idx]]--;
			num_words_assigned_to_topic[topic_assignment[word_idx]]--;
		}
		num_same_words_assigned_to_class[word][class_assignment[word_idx]]--;
		num_words_assigned_to_class[class_assignment[word_idx]]--;
		int current_class = class_assignment[word_idx];
		if(previous_class != -1)
			num_transitions[previous_class][current_class]--;
		previous_class = current_class;
	}
}

void HMMLDA::relabel_document(vector<int> &document, vector<int> &topic_assignment, vector<int> &class_assignment, int doc_size, int iterations) {
	add_counts_for_new_document(document, topic_assignment, class_assignment, doc_size);
	// Resample the current document for some iterations
	for(int i = 1; i <= iterations; i++) {
		std::clock_t start = std::clock();
		for(int word_idx = 0; word_idx < doc_size; word_idx++) {
			draw_class(document_idx, word_idx, doc_size);
			draw_topic(document_idx, word_idx, doc_size);
		}
		std::cout << "Time for iteration " << i <<": " << (std::clock() - start) / (double)(CLOCKS_PER_SEC) << " s" << endl;
		if((i%save_freq) == 0)
			save_assignments(i);
		// cout << endl << endl << endl << endl;
	}
	remove_counts_for_new_document(document, topic_assignment, class_assignment, doc_size);
}






// https://stackoverflow.com/a/236803/4535284
template<typename Out>
void split(const std::string &s, char delim, Out result) {
	std::stringstream ss;
	ss.str(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		*(result++) = item;
	}
}

std::vector<std::string> split(const std::string &s, char delim) {
	std::vector<std::string> elems;
	split(s, delim, std::back_inserter(elems));
	return elems;
}


// Read the vocabulary from the file into an unordered_map
int read_vocabulary(string &vocab_filename, vector<string> &vocab) {
	ifstream file(vocab_filename, ios::in);
	string str;
	while(getline(file, str)) {
		vector<string> str_spl = split(str, ' ');
		vocab.push_back(str_spl[0]);
	}
	return vocab.size();
}

