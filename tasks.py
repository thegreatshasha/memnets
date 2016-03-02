import re
import numpy as np

# test on tasks

# if doesn't work, theorize and do unit tests


# try training on a single task, can it copy?
task1 = (['John went to the kitchen'], 'Where is John?', 'kitchen')

# can it learn the patter X went to the Y. Where is X? Y
task2 = (['Mary went to the garden'], 'Where is Mary?', 'garden')
task3 = (['Agnis went to the office'], 'Where is Agnis?', 'store')

max_length = 10

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

def build_vocab(task):
    facts = task[0]
    facts.append(task[1])
    vocab = {}
    count = 0
    for sent in facts:
        words = tokenize(sent)
        for word in words:
            if not word in vocab:
                vocab[word] = count = count + 1
    return vocab

def onehot_fact(fact, vocab, max_length):
    task_enc = np.zeros((len(vocab), max_length))
    for index, word in enumerate(fact):
        task_enc[vocab[word]-1, index] = 1
    return task_enc

def onehot_facts(facts, vocab, max_length):
    tasks_enc = np.zeros((len(facts), len(vocab), max_length))
    for index, fact in enumerate(facts):
        tasks_enc[index,:,:] = onehot_fact(fact, vocab, max_length)
    return tasks_enc

facts = [tokenize(fact) for fact in task1[0]]
vocab = build_vocab(task1)
inputt = onehot_facts(facts, vocab, max_length)
question = onehot_fact(tokenize(task1[1]), vocab, max_length)
n = inputt.shape[0]
answer = np.zeros((len(vocab), 1))
answer[vocab[task1[2]]-1] = 1
