import numpy as np
import NN
import csv
import scipy.io
from sklearn import preprocessing
import matplotlib.pyplot as plt


alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']


data = scipy.io.loadmat('hw6_data_dist/letters_data.mat')
test_X = data['test_x'] / 255
train_Y = data['train_y'] - 1
train_X = data['train_x'] / 255

N, D = train_X.shape

# shuffle
comb_XY = np.concatenate((train_X, train_Y), axis=1)
np.random.shuffle(comb_XY)
train_X = comb_XY[:, :D]
train_Y = comb_XY[:, D:].astype('int')
train_size = int(0.8 * N)

# split training and validation set
valid_X = train_X[train_size:, :]
valid_Y = train_Y[train_size:, :]
train_X = train_X[:train_size, :]
train_Y = train_Y[:train_size, :]

#
nn = NN.NeuralNet(784, 300, 26)
loss, train_acc = nn.train(train_X, train_Y, max_iter=50000, eps=3e-2, batch_size=100)
#

plt.figure(1)
plt.plot(loss)
plt.title('loss history, batch size: 100')
plt.xlabel('number of iterations (in 100)')
plt.ylabel('loss')
plt.savefig('loss_history.png')

plt.figure(2)
plt.plot(train_acc)
plt.title('training accuracy, batch size: 100')
plt.xlabel('number of iterations (in 100)')
plt.ylabel('accuracy')
plt.savefig('train_acc_history.png')


predict_Y = nn.predict(valid_X)
score = 0
count1 = 0
count2 = 0
correct = []
wrong = []
for i in range(valid_X.shape[0]):
    if int(predict_Y[i]) == int(valid_Y[i]):
        score += 1
        if count2 < 5:
            count2 += 1
            correct.append([i, predict_Y[i]])
    elif count1 < 5:
        count1 += 1
        wrong.append([i, predict_Y[i], valid_Y[i]])
print('validation accuracy: '+str(score / valid_X.shape[0]))




predict_Y = nn.predict(test_X) + 1
with open('res.csv', 'wt') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['Id', 'Category'])
    for i in range(predict_Y.shape[0]):
        writer.writerow([str(i + 1), str(predict_Y[i])])



for i in range(5):
    plt.imshow(valid_X[correct[i][0]].reshape((28,28)))
    plt.title('correct classification! true: '+alphabet[correct[i][1]])
    plt.savefig('correct_'+str(i)+'.png')

for i in range(5):
    plt.imshow(valid_X[wrong[i][0]].reshape((28,28)))
    plt.title('wrong classification! true: ' + alphabet[wrong[i][2]]+' predicted: '+alphabet[wrong[i][1]])
    plt.savefig('wrong_' + str(i) + '.png')






###
# validation accuracy: