# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 22:12:51 2021

@author: Abhinav
"""

import numpy as np 
import re
import en_core_web_sm
#import contractions
from string import punctuation
import nltk
nltk.download('punkt')
nltk.download('stopwords')

import warnings
warnings.filterwarnings("ignore")

from spacy.lang.en.stop_words import STOP_WORDS
from gensim.summarization import summarize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

from abc import ABC
from abc import abstractmethod

class ExtractiveSummarizer(ABC):

    def __init__(self):
        super().__init__()
        pass
    
    def text_processing(self, text, lower=False) -> str:

      text = re.sub(r'.*\(CNN\) -- ',' ', text)
      text = re.sub(r'LRB',' ', text)
      text = re.sub(r'RRB',' ', text)
      text = re.sub(r'\n',' ', text)
      text = re.sub(r'>',' ', text)
      text = re.sub(r'<',' ', text)
      text = re.sub(r'[" "]+', " ", text)
      text = re.sub(r'-- ',' ', text)
      text = re.sub(r"([?!¿])", r" \1 ", text)
      text = re.sub(r'-',' ', text)
      text = text.replace('/',' ')
      text = re.sub(r'\s+', ' ', text)
      #text = contractions.fix(text)
      text = re.sub('[^A-Za-z0-9.,\']+', ' ', text)
      if lower:
        text = text.lower()
      text = text.strip()
      return text
    
    @abstractmethod
    def get_summary(self, text:str):
        raise NotImplementedError("Method 'generate_summary' must me implemented in derived classes.")

class TFIDFExtractiveSummarizer(ExtractiveSummarizer):
    
    def __init__(self):
        super().__init__()
        
    def __frequency_matrix(self, text) -> dict:
      nlp = en_core_web_sm.load()
      sentences = nlp(text).sents
      stopwords = list(STOP_WORDS)
      pos_tag = ['PROPN', 'ADJ', 'NOUN', 'VERB']
      frequency_matrix = {}
    
      for sentence in sentences:
          frequency_table = {}
          for word in sentence:
            if(word.text in stopwords or word.text in punctuation):
              continue
            if(word.pos_ in pos_tag):
              if word.lemma_ in frequency_table:
                  frequency_table[word.lemma_] += 1
              else:
                  frequency_table[word.lemma_] = 1
          if len(frequency_table) <=2:
            continue
          frequency_matrix[sentence.text] = frequency_table
    
      return frequency_matrix

    def __transfer_frequency_matrix(self,freq_matrix) -> dict:
        
        tf_matrix = {}
    
        for sentence, freq_table in freq_matrix.items():
            tf_table = {}
    
            words_count_in_sentence = len(freq_table)
            for word, count in freq_table.items():
                tf_table[word] = count / words_count_in_sentence
    
            tf_matrix[sentence] = tf_table
    
        return tf_matrix

    def __documents_per_word(self,freq_matrix) -> dict:
        word_per_doc_table = {}
    
        for sentence, freq_table in freq_matrix.items():
            for word, count in freq_table.items():
                if word in word_per_doc_table:
                    word_per_doc_table[word] += 1
                else:
                    word_per_doc_table[word] = 1
    
        return word_per_doc_table

    def __inv_doc_freq_matrix(self,freq_matrix, count_doc_per_words) -> dict:
        idf_matrix = {}
        total_documents = len(freq_matrix)
        for sent, freq_table in freq_matrix.items():
            idf_table = {}
    
            for word in freq_table.keys():
                idf_table[word] = np.log10(total_documents / float(count_doc_per_words[word]))
    
            idf_matrix[sent] = idf_table
    
        return idf_matrix

    def __tf_idf_matrix(self,tf_matrix, idf_matrix) -> dict:
        tf_idf_matrix = {}
    
        for (sent_tf, freq_table_tf), (sent_idf, freq_table_idf) in zip(tf_matrix.items(), idf_matrix.items()):
    
            tf_idf_table = {}
    
            for (word_tf, value_tf), (word_idf, value_idf) in zip(freq_table_tf.items(),
                                                        freq_table_idf.items()):  # here, keys are the same in both the table
                tf_idf_table[word_tf] = float(value_tf * value_idf)
    
            tf_idf_matrix[sent_tf] = tf_idf_table
    
        return tf_idf_matrix

    def __sentence_scores(self,tf_idf) -> dict:
        """
        score a sentence by its word's TF
        Basic algorithm: adding the TF frequency of every non-stop word in a sentence divided by total no of words in a sentence.
        :rtype: dict
        """
    
        scores = {}
    
        for sent, freq_table in tf_idf.items():
            total_score_per_sentence = 0
    
            words_count_in_sentence = len(freq_table)
            for word, score in freq_table.items():
                total_score_per_sentence += score
    
            scores[sent] = total_score_per_sentence / words_count_in_sentence
    
        return scores

    def __average_score(self, scores) -> float:
        """
        Find the average score from the sentence value dictionary
        :rtype: float
        """
        if isinstance(scores, dict):
          return np.average(list(scores.values()))
    
        return 0.0

    def __generate_summary(self, text, scores, threshold) -> str:
        sentence_count = 0
        summary = ''
        nlp = en_core_web_sm.load()
        sentences = list(nlp(text).sents)
        for sentence in sentences:
            #print(sentence)
            #print(sentence, scores[sentence.text.strip()])
            if sentence.text in scores and scores[sentence.text] >= 1.3 * threshold:
                #print(sentence, scores[sentence.text])
                summary += " " + sentence.text
                sentence_count += 1

        return summary.strip()
    def get_summary(self, rawtext):
        """
        :param text: text of long article
        :return: summarized generated summary
        """
        
        processed_text = self.text_processing(rawtext)
        # Create the Frequency matrix of the words in each sentence.
        freq_matrix = self.__frequency_matrix(processed_text)
        # Calculate TermFrequency and generate a matrix
        tf_matrix = self.__transfer_frequency_matrix(freq_matrix)
        # creating table for documents per words
        doc_count_per_words = self.__documents_per_word(freq_matrix)
        # Calculate IDF and generate a matrix
        idf_matrix = self.__inv_doc_freq_matrix(freq_matrix, doc_count_per_words)
        # Calculate tf_idf.py and generate a matrix
        tf_idf_matrix = self.__tf_idf_matrix(tf_matrix, idf_matrix)
        # Important Algorithm: score the sentences
        sentence_scores = self.__sentence_scores(tf_idf_matrix)
        # Find the threshold
        threshold = self.__average_score(sentence_scores)
        # Generate the summary
        summary = self.__generate_summary(processed_text, sentence_scores, threshold)
        
        return summary

class GensimExtractiveSummarizer(ExtractiveSummarizer):
    
    def __init__(self):
        super().__init__()
        
    def __gensim_summarizer(self,text):
        return summarize(text)
    
    def get_summary(self, rawtext):
        summary = self.__gensim_summarizer(rawtext)

        return summary
    
class SumyExtractiveSummarizer(ExtractiveSummarizer):
    
    def __init__(self):
        super().__init__()
    
    def __sumy_summarizer(self, text):
    	parser = PlaintextParser.from_string(text,Tokenizer("english"))
    	lex_summarizer = LexRankSummarizer()
    	summary = lex_summarizer(parser.document,3)
    	summary_list = [str(sentence) for sentence in summary]
    	summary = ' '.join(summary_list)
    	return summary
    
    def get_summary(self, rawtext):
        summary = self.__sumy_summarizer(rawtext)

        return summary

if __name__ == '__main__':
    text = "I live on Mars. It is a beautiful planet. It is known as Red Planet. Aliens are said to exist there"
    summarizer = TFIDFExtractiveSummarizer()
    summary = summarizer.get_summary(text)
    print(summary)
    
    summarizer = GensimExtractiveSummarizer()
    summary = summarizer.get_summary(text)
    print(summary)
    
    summarizer = SumyExtractiveSummarizer()
    summary = summarizer.get_summary(text)
    print(summary)
    
