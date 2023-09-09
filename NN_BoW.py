import pandas as pd
import numpy as np
import nltk
import heapq
import random

my_heads = ['text', 'class_1', 'class_2']
df = pd.read_csv('data_1_with_classes.txt', delimiter='.', header=None, names=my_heads)
df['text'] = df['text'].str.replace(str([i for i in range(0, 10)]), ' ', regex=True)
df['text'] = df['text'].str.lower()
df['text'] = df['text'].str.strip()
df['text'] = df['text'].str.split(' ')
for i in df['text'].index:
    if '' in df['text'][i]:
        del df['text'][i][df['text'][i].index('')]
df['class_1'] = df['class_1'].astype('str')
for i in df['class_1']:
    if i == '1':
        df['class_1'] = df['class_1'].str.replace(i, '0')
for i in df['class_1']:
    if i == '2':
        df['class_1'] = df['class_1'].str.replace(i, '1')
df['class_1'] = df['class_1'].astype('int')

corpus = []
for i in df['text']:
    corpus.append(i)

aux_1 = 0
while True:
    corpus[aux_1] = ' '.join(corpus[aux_1])
    aux_1 += 1
    if aux_1 == len(corpus):
        break

wordfreq = {}
sent_vecs = []
sent_vec_length = 600 ### размер вектора предложения

for sentence in corpus:
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        if token not in wordfreq.keys():
            wordfreq[token] = 1
        else:
            wordfreq[token] += 1

most_freq = heapq.nlargest(sent_vec_length, wordfreq, key=wordfreq.get)

for sentence in corpus:
    sentence_tokens = nltk.word_tokenize(sentence)
    sv = []
    for token in most_freq:
        if token in sentence_tokens:
            sv.append(1)
        else:
            sv.append(0)
    sent_vecs.append(sv)

arr_data = np.asarray(sent_vecs)
arr_class = df['class_2'].values

dataset = []
aux_2 = 0
while aux_2 < len(arr_data):
    dataset.append(tuple((np.array([arr_data[aux_2]]), arr_class[aux_2])))
    aux_2 += 1

########################################################################################################################

### НЕЙРОНЫ
ins = sent_vec_length ### количество входов; значение этой переменной должно равняться значению переменной 'sent_vec_length'
outs = 4 ### количество нейронов выходного слоя; значение этой переменной должно совпадать с количеством классов
hiddens = 15 ### количество нейронов скрытого слоя

### ФУНКЦИИ АКТИВАЦИИ, НОРМАЛИЗАЦИИ, ПОТЕРЬ
def relu(t):
    return np.maximum(t, 0)
def relu_deriv(t):
    return (t >= 0).astype(float)
def softmax(t):
    out = np.exp(t)
    return out / np.sum(out)
def softmax_batch(t):
    out = np.exp(t)
    return out / np.sum(out, axis=1, keepdims=True)
def sparse_cross_entropy_batch(z, y):
    return -np.log(np.array([z[j, y[j]] for j in range(len(y))]))
def to_full_batch(y, num_classes):
    y_full = np.zeros((len(y), num_classes))
    for j, yj in enumerate(y):
        y_full[j, yj] = 1
    return y_full

### ВХОДЫ И ВЫХОДЫ
x = np.random.randn(1, ins)
y = np.random.randn(0, outs - 1)

### НАЧАЛЬНЫЕ ВЕСА
W1 = np.random.rand(ins, hiddens)
b1 = np.random.rand(1, hiddens)
W2 = np.random.rand(hiddens, outs)
b2 = np.random.rand(1, outs)

### ГИПЕРПАРАМЕТРЫ
learning_rate = 0.002 ### темп обучения
epochs = 200 ### количество эпох
batch_size = 25 ### размер пакетов

### ОБУЧЕНИЕ
loss_arr = []
for ep in range(epochs):

    random.shuffle(dataset)

    for i in range(len(dataset) // batch_size):

        batch_x, batch_y = zip(*dataset[i*batch_size : i*batch_size+batch_size])
        x = np.concatenate(batch_x, axis=0)
        y = np.array(batch_y)

        ### ПЯМОЕ РАСПРОСТРАНЕНИЕ

        t1 = x @ W1 + b1
        h1 = relu(t1)
        t2 = h1 @ W2 + b2
        z = softmax_batch(t2)
        E = np.sum(sparse_cross_entropy_batch(z, y))

        ### ОБРАТНОЕ РАСПРОСТРАНЕНИЕ

        y_full = to_full_batch(y, outs)
        dE_dt2 = z - y_full
        dE_dW2 = h1.T @ dE_dt2
        dE_db2 = np.sum(dE_dt2, axis=0, keepdims=True)
        dE_dh1 = dE_dt2 @ W2.T
        dE_dt1 = dE_dh1 * relu_deriv(t1)
        dE_dW1 = x.T @ dE_dt1
        dE_db1 = np.sum(dE_dt1, axis=0, keepdims=True)

        ### ОБНОВЛЕНИЕ ВЕСОВ

        W1 = W1 - learning_rate * dE_dW1
        b1 = b1 - learning_rate * dE_db1
        W2 = W2 - learning_rate * dE_dW2
        b2 = b2 - learning_rate * dE_db2

        # ПОТЕРИ

        loss_arr.append(E)

### ФУНКЦИЯ ПРЕДСКАЗАНИЯ
def predict(x):
    t1 = x @ W1 + b1
    h1 = relu(t1)
    t2 = h1 @ W2 + b2
    z = softmax(t2)
    return z

### ФУНКЦИЯ РАСЧЕТА ТОЧНОСТИ
def calc_accuracy():
    correct = 0
    for x, y in dataset:
        z = predict(x)
        y_pred = np.argmax(z)
        if y_pred == y:
            correct += 1
    acc = correct / len(dataset)
    return acc
print('Accuracy: ', calc_accuracy())

### ГРАФИК ОБУЧЕНИЯ
def learning_chart():
    import matplotlib.pyplot as plt
    plt.plot(loss_arr)
    plt.show()
learning_chart()

### ВВОД И ВЕКТОРИЗАЦИЯ ТЕСТОВОГО ПРЕДЛОЖЕНИЯ
def test_sentence_input_and_vectorization():
    print('Enter your sentence:')
    test_sentence = input(str())
    test_sentence = test_sentence[:-1]
    test_sentence = test_sentence.lower()
    test_sentence = list([test_sentence])
    for word in test_sentence:
        tokens = nltk.word_tokenize(word)
        for token in tokens:
            if token not in wordfreq.keys():
                wordfreq[token] = 1
            else:
                wordfreq[token] += 1
    for word in test_sentence:
        sentence_tokens = nltk.word_tokenize(word)
        sv = []
        for token in most_freq:
            if token in sentence_tokens:
                sv.append(1)
            else:
                sv.append(0)
        sent_vecs.append(sv)
    test_sentence_vector = np.array([sent_vecs[len(sent_vecs) - 1]])
    sent_vecs.pop()
    return test_sentence_vector
test_sentence_vector = test_sentence_input_and_vectorization()

### ПРЕДСКАЗАНИЕ КЛАССА ТЕСТОВОГО ПРЕДЛОЖЕНИЯ
prediction = predict(test_sentence_vector)
predicted_class = np.argmax(prediction)
class_names = ['Formal Subject', 'Personal subject', 'Object', 'None']
print('Predicted class: ', class_names[predicted_class])

### ФУНКЦИЯ ЗАПИСИ ВЕСОВ ПОСЛЕ ОБУЧЕНИЯ В CSV-ФАЙЛ
def writing_weights_to_csv():
    np.savetxt("learnt_NN_layer_1_weights.csv", W1, delimiter=",")
    np.savetxt("learnt_NN_layer_1_bias.csv", b1, delimiter=",")
    np.savetxt("learnt_NN_layer_2_weights.csv", W2, delimiter=",")
    np.savetxt("learnt_NN_layer_2_bias.csv", b2, delimiter=",")
writing_weights_to_csv()