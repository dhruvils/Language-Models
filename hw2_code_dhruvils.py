#!python

import os
from subprocess import check_output
import string
import operator
from math import log10, pow
from collections import Counter
import xml.etree.ElementTree as ET

## @file_list.txt will need absolute file path
## @corenlp_output_dir will need to be an absolute path as well, otherwise files will be saved in the wrong dir
def preprocess(file_list, corenlp_output_dir):
    os.system("(cd /home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09; java -cp stanford-corenlp-2012-07-09.jar:stanford-corenlp-2012-07-06-models.jar:xom.jar:" +\
                   "joda-time.jar -Xmx3g edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit," +\
                   "pos,lemma,ner -filelist " +file_list +" -outputDirectory " +corenlp_output_dir +")")

def get_all_files(dir):
    absoluteFileList = []
    for dirpath, dirs, files in os.walk(dir):
        absoluteFileList += [ dirpath + '/' + filename for filename in files]
    return absoluteFileList

def process_file(input_xml, output_file):
    tree = ET.parse(input_xml)
    output = "STOP"

    for sentence in tree.iter('sentence'):
        prev = ""
        for token in sentence.iter('token'):
            if token[0].text in string.punctuation:
                prev = token[0].text
                continue
            elif token[5].text == 'O':
                output += " " +token[0].text
                prev = token[0].text
            else:
                if prev != token[5].text:
                    output += " " +token[5].text
                prev = token[5].text            
        output += " STOP"

    f = open(output_file, 'w')
    f.write(output)
    f.close()

class BigramModel:
    def __init__(self, trainfiles):
        wordcount = dict()

        filestring = ""
        for filepath in trainfiles:
            f = open(filepath, 'r')
            filestring += f.read() +' '

        filestring_list = filestring.split()
        if '' in filestring_list: filestring_list.remove('')
        wordcount = Counter(filestring_list)
        
        index = 0
        for word in filestring_list:
            if wordcount[word] == 1:
                filestring_list[index] = "<UNK>"
            index += 1

        bigram_dict = dict()
        self.prec_occur_count = dict()
        index = 0
        
        for index_i in xrange(len(filestring_list) - 1):
            if filestring_list[index_i] != 'STOP' or filestring_list[index_i + 1] != 'STOP':
                if (filestring_list[index_i], filestring_list[index_i + 1]) not in bigram_dict:
                    bigram_dict[(filestring_list[index_i], filestring_list[index_i + 1])] = 1
                else:
                    bigram_dict[(filestring_list[index_i], filestring_list[index_i + 1])] += 1

                if filestring_list[index_i] in self.prec_occur_count:
                    self.prec_occur_count[filestring_list[index_i]] += 1
                else:
                    self.prec_occur_count[filestring_list[index_i]] = 1

        self.bigramModel = bigram_dict
        self.wordCount = Counter(filestring_list)
        
    def logprob(self, context, event):
        if context not in self.wordCount:
            context = '<UNK>'
        if event not in self.wordCount:
            event = '<UNK>'

        num = ((self.bigramModel[(context, event)]) if (context, event) in self.bigramModel else 0) + 1
        den = self.prec_occur_count[context] + len(self.wordCount)

        return log10(float(num) / den)

    def print_model(self, output_file):
        output_string = ""

        for context in self.wordCount.keys():
            output_string += context +":"
            for event in self.wordCount.keys():
                output_string += " " +event +" " +str(self.logprob(context, event))
            output_string += "\n"

        f = open(output_file, 'w')
        f.write(output_string)
        f.close()

    def getppl(self, testfile):
        f = open(testfile, 'r')
        input_string = f.read()
        f.close()

        testdata_list = input_string.split()
        if '' in testdata_list: testdata_list.remove('')

        num_words = len(testdata_list)
        total_prob = 0
        index = 1
        for index in xrange(num_words):
            total_prob += self.logprob(testdata_list[index - 1], testdata_list[index])

        log_prob = (total_prob / float(num_words))
        return (pow(10, -log_prob))

def get_srilm_ppl_for_file(lm_file, test_file):
    f = open(test_file, 'r')
    input_string = f.read()
    f.close()

    file_name = test_file.split('/')
    if not os.path.exists('./processed_test_set'):
        os.makedirs('./processed_test_set')
    processed_test_file = './processed_test_set/' +file_name[-1]

    output_string = string.replace(input_string, 'STOP', '\n')
    f = open(processed_test_file, 'w')
    f.write(output_string)
    f.close()

    output_list = check_output(['/home1/c/cis530/hw2/srilm/ngram','-lm',lm_file,'-ppl',processed_test_file]).split()
    return output_list[output_list.index('ppl=') + 1]

def create_concat_file(directory):
    file_list = get_all_files(directory)
    file_string = ""

    for file_name in file_list:
        f = open(file_name, 'r')
        file_string += f.read() +' '
        f.close()

    f = open('concat_test_data.txt', 'w')
    f.write(file_string)
    f.close()

def get_all_ppl(bigrammodel, directory):
    create_concat_file(directory)
    return bigrammodel.getppl('concat_test_data.txt')

def get_all_ppl_srilm(lm_file, directory):
    create_concat_file(directory)
    return get_srilm_ppl_for_file(lm_file, 'concat_test_data.txt')

def write_ppl_values():
    trainfilelist = get_all_files('/home1/c/cis530/hw2/data/processed_train_set')
    x = BigramModel(trainfilelist)                                                                                              
    test_dir = '/home1/c/cis530/hw2/data/processed_test_set'

    lm0 = get_all_ppl(x, test_dir)                                                                                                 
    lm1 = get_all_ppl_srilm('complete.unigram.srilm', test_dir)
    lm2 = get_all_ppl_srilm('complete.bigram.srilm', test_dir)
    lm3 = get_all_ppl_srilm('complete.trigram.srilm', test_dir)

    lm1 = float(lm1)
    lm2 = float(lm2)
    lm3 = float(lm3)
    lm_lst = [lm0, lm1, lm2, lm3]
    temp_lst = sorted([lm0, lm1, lm2, lm3])
    output_string = "LM ranking: "
    
    for lm in temp_lst:
        output_string += str(lm_lst.index(lm)) +' '

    f = open('results.txt', 'w')
    f.write(output_string)
    f.close()

def get_distinctive_measure(lm_file, mem_quote_file, nonmem_quote_file):
    return (get_srilm_ppl_for_file(lm_file, mem_quote_file), get_srilm_ppl_for_file(lm_file, nonmem_quote_file))

def distinctive_highppl_percentage(lm_file, directory):
    file_list = get_all_files(directory)
    f = open('quote_file_list.txt', 'w')
    for file_path in file_list:
        f.write(file_path +'\n')
    f.close()
    
    cwd = os.getcwd()

    if not os.path.exists('./corenlp_quotes_dir'): os.makedirs('./corenlp_quotes_dir')
    preprocess(cwd +'/quote_file_list.txt', cwd +'/corenlp_quotes_dir')

    if not os.path.exists('./processed_quotes_dir'): os.makedirs('./processed_quotes_dir')
    corenlp_file_list = get_all_files(cwd +'/corenlp_quotes_dir')
    for xml_file in corenlp_file_list:
        file_name = xml_file.split('/')[-1]
        process_file(xml_file, 'processed_quotes_dir/' +file_name[:-4])
    
    quote_file_list = get_all_files(cwd +'/processed_quotes_dir')
    notmem_file_dict = dict()
    mem_file_dict = dict()
    for f in quote_file_list:
        prefix = f.split('/')[-1].split('_')[0]
        if "not_mem" in f:
            notmem_file_dict[prefix] = f
        else:
            mem_file_dict[prefix] = f
    
    count = 0
    for prefix in notmem_file_dict.keys():
        ppl = get_distinctive_measure(lm_file, mem_file_dict[prefix], notmem_file_dict[prefix])
        if ppl[0] > ppl[1]: count += 1

    per = (float(count) / len(notmem_file_dict)) * 100
    
    f = open('results.txt', 'a')
    f.write('\nPercentage of memorable quotes from LM 3 with higher perplexity: ' +str(per) +'%')
    f.close()

def get_bestfit(sentence, wordlist, bigrammodel):
    cwd = os.getcwd()
    file_list = open(cwd +'/sentence_file_list.txt', 'w')

    if not os.path.exists('./sentences_dir'): os.makedirs('./sentences_dir')
    f = open(cwd +'/sentences_dir/temp_sentence_file.txt', 'w')
    f.write(sentence)
    f.close()
    file_list.write(cwd +'/sentences_dir/temp_sentence_file.txt')
    file_list.close()

    if not os.path.exists('./sentence_xml_dir'): os.makedirs('./sentence_xml_dir')
    preprocess(cwd +'/sentence_file_list.txt', cwd +'/sentence_xml_dir')

    if not os.path.exists('./processed_sentence_dir'): os.makedirs('./processed_sentence_dir')
    for xml_file in get_all_files(cwd +'/sentence_xml_dir'):
        xml_file_name = xml_file.split('/')[-1]
        process_file(xml_file, cwd +'/processed_sentence_dir/' +xml_file_name[:-4])
    
    f = open(cwd +'/processed_sentence_dir/temp_sentence_file.txt', 'r')
    processed_sent = f.read()
    f.close()
    ppl_dict = dict()
    
    if not os.path.exists('./temp_sentence_dir'): os.makedirs('./temp_sentence_dir')
    for word in wordlist:
        temp_sentence = string.replace(processed_sent, "<blank>", word)
        f = open(cwd +'/temp_sentence_dir/temp_sentence_file.txt', 'w')
        f.write(temp_sentence)
        f.close()
        
        ppl_dict[word] = bigrammodel.getppl(cwd +'/temp_sentence_dir/temp_sentence_file.txt')

    sorted_list = sorted(ppl_dict.items(), key = operator.itemgetter(1))
    return sorted_list[0][0]

def write_accuracy(bigrammodel):
    sentence_list = ["Stocks <blank> this morning.", "Stocks plunged this morning, despite a cut in interest <blank> by the Federal Reserve.", "Stocks plunged this morning, despite a cut in interest rates by the <blank> Reserve.", "Stocks plunged this morning, despite a cut in interest rates by the Federal Reserve, as Wall Street began <blank> for the first time.", "Stocks plunged this morning, despite a cut in interest rates by the Federal Reserve, as Wall Street began trading for the first time since last Tuesday's <blank> attacks."]
    options_list = [['plunged', 'walked', 'discovered', 'rise'], ['rates','patients','researchers', 'levels'], ['Federal','university','bank','Internet'], ['trading','wondering','recovering','hiring'], ['terrorist','heart','doctor','alien']]

    index = 0
    count = 0
    for index in xrange(len(sentence_list)):
        if options_list[index].index(get_bestfit(sentence_list[index], options_list[index], bigrammodel)) == 0:
            count += 1

    per = (float(count) / len(sentence_list)) * 100
    f = open('results.txt', 'a')
    f.write('\n' +str(per))
    f.close()
    
    return per

def fill_blank(sentence, bigrammodel):
    return get_bestfit(sentence, bigrammodel.wordCount.keys(), bigrammodel)


# main function:
def main():
    #1.1.1:
    #preprocess('/home1/c/cis530/hw3/corenlp/stanford-corenlp-2012-07-09/files.txt','/home1/d/dhruvils/homework2/corenlp_output_dir')

    #1.1.2:
    #input_xml = 'corenlp_output_dir/3164142.txt.xml'
    #output_file = 'processfile_output_dir/3164142.txt'
    #process_file(input_xml, output_file)

    #1.2.1:
    #trainfilelist = get_all_files('home1/c/cis530/hw2/data/processed_train_set')
    #x = BigramModel(trainfilelist)
    #x.print_model('output.txt')

    #x = BigramModel(['/home1/d/dhruvils/homework2/testdoc.txt'])
    #x.print_model('output.txt')
    
    #1.2.2:
    #trainfilelist = get_all_files('/home1/c/cis530/hw2/data/processed_train_set')

    #x = BigramModel(trainfilelist)
    #print x.getppl('/home1/c/cis530/hw2/data/processed_train_set/Finance_2005_06_20_1681717_p.txt')

    #1.3:
    #lm_file = 'complete.bigram.srilm'
    #test_file = '/home1/c/cis530/hw2/data/processed_test_set/Health_2005_05_16_1672977_p.txt'
    #print get_srilm_ppl_for_file(lm_file, test_file)

    #1.4:
    #trainfilelist = get_all_files('/home1/c/cis530/hw2/data/processed_train_set')
    #x = BigramModel(trainfilelist)
    #test_dir = '/home1/c/cis530/hw2/data/processed_test_set'
    #print get_all_ppl(x, test_dir)
    #print get_all_ppl_srilm('complete.unigram.srilm', test_dir)
    #print get_all_ppl_srilm('complete.bigram.srilm', test_dir)
    #print get_all_ppl_srilm('complete.trigram.srilm', test_dir)
    #write_ppl_values()

    #2.1:
    #lm_file = 'complete.trigram.srilm'
    #mem_quote_file = 'quotefile/mem.txt'
    #nonmem_quote_file = 'quotefile/nonmem.txt'
    #print get_distinctive_measure(lm_file, mem_quote_file, nonmem_quote_file)

    #2.2:
    #lm_file = 'complete.trigram.srilm'
    #directory = '/home1/c/cis530/hw2/data/quotes'
    #distinctive_highppl_percentage(lm_file, directory)

    #3.1:
    #trainfilelist = get_all_files('/home1/c/cis530/hw2/data/processed_train_set')
    #x = BigramModel(trainfilelist)
    #sentence = 'Stocks <blank> this morning.'
    wordlist = ['plunged', 'walked', 'discovered', 'rise']
    #sentence = "Stocks plunged this morning, despite a cut in interest rates by the Federal Reserve, as Wall Street began trading for the first time since last Tuesday's <blank> attacks."
    #wordlist = ['terrorist', 'heart', 'doctor', 'alien']
    
    #print get_bestfit(sentence, wordlist, x)

    #3.2:
    #print write_accuracy(x)

    #3.3:
    #trainfilelist = get_all_files('/home1/c/cis530/hw2/data/processed_train_set')
    #sentence = ['With great powers comes great <blank>', "Say hello to my little <blank>", "Hope is the quintessential human delusion, simultaneously the source of your greatest strength, and your greatest <blank>", 'You either die a hero or you live long enough to see yourself become the <blank>', "May the Force be with <blank>", "Every gun makes its own <blank>"]
    #x = BigramModel(trainfilelist)
    #for sent in sentence:
    #    print fill_blank(sent, x)

if __name__ == "__main__":
    main()
