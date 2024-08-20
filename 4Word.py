import math
import time
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt

all_letters = "abcdefghijklmnopqrstuvwxyz"
numbLetters = len(all_letters)
categories = ["Real", "Fake"]

def read_data(file):
    lines = open(file).read().strip().split('\n')
    return lines

def line_to_tensor(line):
    tensor = torch.zeros(4, 1, numbLetters) #4 letter words, 1 coloumn, 26 letters in alphabet
    for li, letter in enumerate(line):
        idx = all_letters.find(letter)
        tensor[li][0][idx] = 1
    return tensor

def getCategory(output):
    top_n, top_i = output.data.topk(1)
    category_i = top_i[0][0]
    return categories[category_i], category_i

def getTrainingPair():
    category = random.choice(categories)
    line = random.choice(testing_lines[category])
    category_tensor = torch.LongTensor([categories.index(category)])
    line_tensor = line_to_tensor(line).requires_grad_(True)
    return category, line, category_tensor, line_tensor

def train(category_tensor, line_tensor):
    rnn.zero_grad()
    hidden = rnn.init_hidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    optimizer.step()

    return output, loss.data.item()

def evaluate(line_tensor):
    hidden = rnn.init_hidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    return output

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(1)
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size, requires_grad=True)


training_lines = {}
testing_lines = {}

lines = read_data("fakeWords")
random.shuffle(lines)
training_lines["Fake"] = lines[:-50]
testing_lines["Fake"] = lines[-50:]

lines = read_data("realWords")
random.shuffle(lines)
training_lines["Real"] = lines[:-150]
testing_lines["Real"] = lines[-150:]

numbHidden = 128
rnn = RNN(numbLetters, numbHidden, 2)

input = line_to_tensor(training_lines["Real"][0])
input.requires_grad_(True)
hidden = rnn.init_hidden()

output, next_hidden = rnn(input[0], hidden)

criterion = nn.NLLLoss()

learning_rate = 0.025
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

n_epochs = 50000
print_every = 5000
plot_every = 1000

# Keep track of losses for plotting
current_loss = 0
all_losses = []


start = time.time()

for epoch in range(1, n_epochs + 1):
    # Get a random training input and target
    category, line, category_tensor, line_tensor = getTrainingPair()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss
    
    # Print epoch number, loss, name and guess
    if epoch % print_every == 0:
        guess, guess_i = getCategory(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (epoch, epoch / n_epochs * 100, time_since(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if epoch % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

plt.figure()
plt.xlabel("Time")
plt.ylabel("Loss")
plt.plot(all_losses)
plt.show()

correct = 0.0
for i in range(0, len(testing_lines["Fake"])):
    tensor = line_to_tensor(testing_lines["Fake"][i])
    output = evaluate(tensor)
    guess, guess_i = getCategory(output)
    if (guess == "Fake"):
        correct += 1
fakeRatio = correct/len(testing_lines["Fake"])

correct = 0.0
for i in range(0, len(testing_lines["Real"])):
    tensor = line_to_tensor(testing_lines["Real"][i])
    output = evaluate(tensor)
    guess, guess_i = getCategory(output)
    if (guess == "Real"):
        correct += 1
realRatio = correct/len(testing_lines["Real"])

print(f"In the test data the model predicted fake words correctly {fakeRatio*100}% of the time and predicted real words correctly {realRatio*100}% of the time")
