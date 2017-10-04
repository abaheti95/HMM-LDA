msr_cpp_hmm_lda_trainer: cpp_hmm_lda_trainer.cpp model.o
	g++ -std=c++11 -g -Wall Models/cpp_hmm_lda/model.o cpp_hmm_lda_trainer.cpp -o cpp_hmm_lda_trainer.out
model.o:
	g++ -std=c++11 -g -Wall -c Models/cpp_hmm_lda/model.cpp -o Models/cpp_hmm_lda/model.o
clean:
	rm msr_cpp_hmm_lda_trainer.out