import os, sys, re, copy, random

class StringDecorator():
    def __init__(self, chain):
        self.chain = chain
        self.fill = 20
        self.strlen = 8

    def decorate(self):
        if isinstance(self.chain, list): 
            return ' '.join([(s + ' ' * self.fill)[:self.strlen] for s in self.chain])
        else:
            return '\n'.join([str(k) + ' | ' + str(self.chain[k]) for k in self.chain])

class PrevState():
    def __init__(self, order, empty):
        self.order = order
        self.empty = empty
        self.prev_states = [empty] * order

    def shift(self, item):
        if self.order > 0:
            ck = PrevState(self.order, self.empty)
            ck.prev_states = copy.copy(self.prev_states)
            ck.prev_states[0] = item
            ck.prev_states.append(ck.prev_states.pop(0))
            return ck
        else:
            return self

    def __eq__(self, other):
        if isinstance(other, PrevState):
            return self.prev_states == other.prev_states
        else:
            return False

    def __ne__(self, other):
        return not(self.__eq__(other))

    def __hash__(self):
        return hash(tuple(self.prev_states))

    def __str__(self):
        return StringDecorator(self.prev_states).decorate()

class NextState():
    def __init__(self):
        self.next_states = []

    def add(self, item):
        self.next_states.append(item)

    def predict(self):
        return random.choice(self.next_states)

    def __str__(self):
        return StringDecorator(self.next_states).decorate()

class MarkovChain():
    def __init__(self, order, empty):
        self.empty = empty
        self.order = order
        self.chain = {}

    def add_state(self, prev, item):
        if prev not in self.chain:
            next_states = NextState()
        else:
            next_states = self.chain[prev]
        next_states.add(item)
        self.chain[prev] = next_states

    def fit(self, items):
        prev = PrevState(self.order, self.empty)
        for item in items:
            self.add_state(prev, item)
            prev = prev.shift(item)

        for i in range(self.order):
            self.add_state(prev, self.empty)
            prev = prev.shift(self.empty)

    def generate(self, n, start):
        result = []
        prev = PrevState(self.order, start) 
        for i in range(n):
            if prev in self.chain:
                next_states = self.chain[prev]
                _state = next_states.predict()  # random policy
                result.append(_state)
                prev = prev.shift(_state)
        return result

    def __str__(self):
        sd = StringDecorator(self.chain)
        return sd.decorate()

def train(texts):
    order = 2
    pred_horizon = 9
    empty_state = '-'

    mc = MarkovChain(order, empty_state)
    
    for text in texts:
        chain = text.split(' ')
        mc.fit(chain)

    print('previous states   | next states')
    print('-'*19)
    print(mc)
    print('-'*19)

    y_hat = mc.generate(pred_horizon, mc.empty)
    print(' '.join(y_hat))

def preproc(texts):
    comma    = '\x2c'      # ,
    dot      = '\x2e'      # .
    space    = ' +'
    match = r'[{}|{}]'.format(comma, dot)
    #match = r'[{}]'.format(comma)

    texts = [t.lower() for t in texts]
    texts = [re.sub(match, ' ', t) for t in texts]    
    texts = [re.sub(space, ' ', t) for t in texts]           
    return texts

def read_file(file):
    with open(file,'r') as f:
        text = f.read().replace('\n', '')
    return text

if __name__ == '__main__':
    text = 'The quick brown fox jumped over the lazy dog.'
    #text = read_file('input/kalevala.txt')
    
    text = preproc([text])[0]
    text2 = text.replace('fox','rabbit').replace('dog','cat')
    rows = [preproc([text])[0], preproc([text2])[0]]
    #rows = [text]

    train(rows)