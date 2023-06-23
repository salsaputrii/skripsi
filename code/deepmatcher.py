#1 import library
import pandas as pd
import deepmatcher as dm
import torch
torch.cuda.is_available()
import os
from sklearn.metrics import roc_auc_score

df = pd.read_csv("D:/salput/stopword/df_stopword_label.csv")
# The directory where the data splits will be saved.
split_path = os.path.join('D:\salput\stopword')

# Split labeled data into train, valid, and test csv files to disk, with the split ratio of 3:1:1.
dm.data.split(df, split_path, 'train.csv', 'valid.csv', 'test.csv',
              [3, 1, 1])

train = pd.read_csv('D:/salput/stopword/train.csv')
valid = pd.read_csv('D:/salput/stopword/valid.csv')
test = pd.read_csv('D:/salput/stopword/test.csv')


## Step 1. Process labeled data
train, validation, test = dm.data.process(
    path='D:/salput/stopword',
    train='train.csv',
    validation='valid.csv',
    test='test.csv',
    ignore_columns=('left_id', 'right_id', 'jaccard'),
    embeddings='fasttext.id.bin')


## Step 2. Define neural network model
model = dm.MatchingModel(attr_summarizer='hybrid')
# model = dm.MatchingModel(
#     attr_summarizer = dm.attr_summarizers.RNN(
#         word_aggregator=dm.word_aggregators.AttentionWithRNN(
#             rnn='lstm', rnn_pool_style='max')))


## Step 3. Train model
model.run_train(
    train,
    validation,
    epochs=10,
    batch_size=256,
    best_save_path='hybryd.pth',
    pos_neg_ratio=3)
    # ignore_columns=('left_id', 'right_id')

# evaluasi model menggunakan data test
test_eval = model.run_eval(test)
print(test_eval)

# Predict labeled
test_predictions = model.run_prediction(test, output_attributes=True)
print(test_predictions.head(50))

# -------------------------------

# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, roc_auc_score
#
# # Prediksi pada data tes
# test_df = pd.read_csv('D:/salput/stopword/test.csv')
# y_test_actual = test_df['jaccard']
# y_test_scores = model.run_prediction(test)
#
# # Menghitung nilai ROC AUC dan kurva ROC
# fpr, tpr, thresholds = roc_curve(y_test_actual, y_test_scores)
# roc_auc = roc_auc_score(y_test_actual, y_test_scores)
#
# # Plot kurva ROC
# plt.figure()
# plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], 'k--')  # Garis diagonal acak
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.show()

# -------------------------------

# Import modul
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt


# Buat data tes yang sebenarnya dan data tes yang diprediksi oleh model
test_df = pd.read_csv('D:/salput/stopword/test.csv')
y_test_actual = test_df['label']
y_test_scores = model.run_prediction(test)


# Hitung nilai precision, recall, dan thresholds dari data tes
precision, recall, thresholds = precision_recall_curve(y_test_actual, y_test_scores)


# Hitung nilai PR-AUC dari data tes
pr_auc = auc(recall, precision)


# Buat objek figure untuk menampilkan grafik
plt.figure()


# Buat grafik garis dengan sumbu x sebagai recall dan sumbu y sebagai precision
plt.plot(recall, precision, label='PR curve (area = %0.2f)' % pr_auc)


# Mengatur batas sumbu x dan y
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])


# Memberikan label untuk sumbu x dan y
plt.xlabel('Recall')
plt.ylabel('Precision')


# Memberikan judul untuk grafik
plt.title('Precision-Recall Curve')


# Menampilkan legenda grafik
plt.legend(loc="lower left")


# Menampilkan grafik yang telah dibuat
plt.show()
