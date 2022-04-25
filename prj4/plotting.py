import pandas as pd
import matplotlib.pyplot as plt

simpleRNN1 = pd.read_csv('history_simpleRNN_100_winsize5_stride3.csv')
simpleRNN2 = pd.read_csv('history_simpleRNN_100_winsize10_stride6.csv')
simpleRNN3 = pd.read_csv('history_simpleRNN_200_winsize5_stride3.csv')
simpleRNN4 = pd.read_csv('history_simpleRNN_200_winsize10_stride6.csv')

lstm1 = pd.read_csv('history_lstm_100_winsize5_stride3.csv')
lstm2 = pd.read_csv('history_lstm_100_winsize10_stride6.csv')
lstm3 = pd.read_csv('history_lstm_200_winsize5_stride3.csv')
lstm4 = pd.read_csv('history_lstm_200_winsize10_stride6.csv')

plt.figure(figsize=(12, 8))
#plt.rcParams.update({'font.size': 15})
plt.subplot(1, 2, 1)
plt.plot(lstm1['val_loss'])
plt.plot(lstm2['val_loss'])
plt.plot(lstm3['val_loss'])
plt.plot(lstm4['val_loss'])
plt.title('LSTM - Validation Model Loss Comparison')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['hidden state=100, winsize=5, stride=3', 'hidden state=100, winsize=10, stride=6',
            'hidden state=200, winsize=5, stride=3', 'hidden state=200, winsize=10, stride=6'])

plt.subplot(1, 2, 2)
plt.plot(lstm1['val_accuracy'])
plt.plot(lstm2['val_accuracy'])
plt.plot(lstm3['val_accuracy'])
plt.plot(lstm4['val_accuracy'])
plt.title('LSTM - Validation Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['hidden state=100, winsize=5, stride=3', 'hidden state=100, winsize=10, stride=6',
            'hidden state=200, winsize=5, stride=3', 'hidden state=200, winsize=10, stride=6'])

plt.show()

plt.figure(figsize=(12, 8))
#plt.rcParams.update({'font.size': 15})
plt.subplot(1, 2, 1)
plt.plot(simpleRNN1['val_loss'])
plt.plot(simpleRNN2['val_loss'])
plt.plot(simpleRNN3['val_loss'])
plt.plot(simpleRNN4['val_loss'])
plt.title('simpleRNN - Validation Model Loss Comparison')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['hidden state=100, winsize=5, stride=3', 'hidden state=100, winsize=10, stride=6',
            'hidden state=200, winsize=5, stride=3', 'hidden state=200, winsize=10, stride=6'])

plt.subplot(1, 2, 2)
plt.plot(simpleRNN1['val_accuracy'])
plt.plot(simpleRNN2['val_accuracy'])
plt.plot(simpleRNN3['val_accuracy'])
plt.plot(simpleRNN4['val_accuracy'])
plt.title('simpleRNN - Validation Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['hidden state=100, winsize=5, stride=3', 'hidden state=100, winsize=10, stride=6',
            'hidden state=200, winsize=5, stride=3', 'hidden state=200, winsize=10, stride=6'])

plt.show()