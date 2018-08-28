import numpy as np
from RNN_numpy import RNN
from data_loader import data_loader

model = RNN(3507) #3507
loader = data_loader()
x,y= loader.data_load()

x_train, y_train = loader.data_load()

model.train(x_train,y_train, learning_rate=0.01, epoch = 1)


def generate_sentence(model):
    new_sentence = [loader.word_to_ix["<START_TOKEN>"]]
    while not new_sentence[-1] == loader.word_to_ix["<END_TOKEN>"]:
        next_word_probs = model.forward_propagation(new_sentence)[0]
        samples = np.random.multinomial(1, next_word_probs[-1])
        sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    new_sentence = loader.transform(np.array(new_sentence))

    return new_sentence

num_sentences = 10
senten_min_length = 5

print("generation")
for i in range(num_sentences):
    sent = []
    while len(sent) < senten_min_length:
        sent = generate_sentence(model)
    print(sent)