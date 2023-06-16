import streamlit as st
import pandas as pd
import numpy as np

from sklearn.utils.validation import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import make_scorer, accuracy_score, precision_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_percentage_error
import pickle
from os.path import exists
from pickle import dump
from collections import OrderedDict

# st.sidebar.title("Selamat Datang!")
# st.sidebar.write(
#     "Di Website Prediksi Saham Perusahaan Bukalapak.")
page1, page2, page3, page4 = st.tabs(
    ["Data", "Preprocessing", "Modelling", "Prediksi"])

with page1:
    st.title("Dataset Saham Bukalapak")
    st.write(
        "Dataset Yang digunakan adalah **Saham Bukalapak** dari [Yahoo Finance](https://finance.yahoo.com/quote/BUKA.JK/history?p=BUKA.JK)")
    st.write("Deskripsi Data")
    st.write("""
    Dataset yang digunakan adalah dataset tentang Saham Bukalapak untuk memprediksi secara diagnostik apakah saham perusahaan Bukalapak akan naik atau turun,
    berdasarkan pengukuran diagnostik tertentu yang disertakan dalam kumpulan data. Data yang digunakan adalah data harian dari tanggal 16 Juni 2022 sampai 14 Juni 2023.
    Yang memiliki kolom close dengan nilai yang berubah-ubah setiap harinya.
    Dalam dataset ini terdapat 7 fitur yaitu : 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', dan 'Volume'.
    Dimana fitur 'Close' yang akan diprediksi.
    Karena 'Close' merupakan harga penutupan saham Bukalapak setiap harinya.
    """)
    st.write("Berikut adalah 5 data teratas dari dataset yang digunakan")
    df = pd.read_csv(
        'https://raw.githubusercontent.com/errjak/dataset/main/BUKA.JK.csv')
    st.dataframe(df.head())

with page2:
    st.title("Preprocessing")
    option = st.selectbox('Pilih Hasil PreProcessing yang diinginkan : ',
                          ('MinMaxScaler', 'Reduksi Dimensi(PCA)'))
    if option == 'MinMaxScaler':
        st.write(
            "Min Max Scaler digunakan untuk mengubah data numerik menjadi data yang memiliki range 0 sampai 1")
        st.write(
            "Berikut adalah 5 data teratas dari dataset yang sudah di Min Max Scaler")
        df = pd.read_csv(
            'https://raw.githubusercontent.com/errjak/dataset/main/minmaxscaler.csv')
        st.dataframe(df.head())
    elif option == 'Reduksi Dimensi(PCA)':
        st.write("PCA digunakan untuk mengurangi dimensi data")
        st.write(
            "Berikut adalah 5 data teratas dari dataset yang sudah di Reduksi Dimensi(PCA)")
        df = pd.read_csv(
            'https://raw.githubusercontent.com/errjak/dataset/main/pca.csv')
        st.dataframe(df.head())

with page3:
    st.title("Modelling")
    data = pd.read_csv(
        'https://raw.githubusercontent.com/errjak/dataset/main/BUKA.JK.csv')
    data = data[['Date', 'Close']]
    from array import array

    def split_sequence(sequence, n_steps):
        X, y = list(), list()
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the sequence
            if end_ix > len(sequence)-1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)

    df = data
    series = df['Close']
    n_steps = 3
    X, y = split_sequence(series, n_steps)

    for i in range(len(X)):
        print(X[i], y[i])

    scaler = MinMaxScaler()
    X_min = scaler.fit_transform(X)
    scalerFile = 'scaler.sav'
    pickle.dump(scaler, open(scalerFile, 'wb'))

    pca = PCA(n_components=1)
    X = pca.fit_transform(X_min)
    pcaFile = 'pca.sav'
    pickle.dump(pca, open(pcaFile, 'wb'))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, shuffle=False, test_size=0.2)

    with st.form(key='form3'):
        option = st.selectbox('Pilih Model yang diinginkan : ',
                              ('Naive Bayes', 'Decision Tree', 'MLP'))
        if option == 'Naive Bayes':

            st.write("Naive Bayes adalah sebuah metode klasifikasi menggunakan metode probabilitas dan statistik yg dikemukakan oleh ilmuwan Inggris Thomas Bayes.")

            model_NB = GaussianNB()
            model_NB.fit(X_train, y_train)
            filename_NB = 'model_NB.sav'
            pickle.dump(model_NB, open(filename_NB, 'wb'))
            y_NB = model_NB.predict(X_test)

            mape_NB = mean_absolute_percentage_error(y_test, y_NB)
            st.write("Hasil MAPE dari Naive Bayes : ", mape_NB)

        elif option == 'Decision Tree':

            st.write("Decision Tree adalah sebuah metode klasifikasi dengan membuat sebuah pohon keputusan dari data pembelajaran yang dimiliki.")

            model_DT = DecisionTreeClassifier()
            model_DT.fit(X_train, y_train)
            filename_DT = 'model_DT.sav'
            pickle.dump(model_DT, open(filename_DT, 'wb'))
            y_DT = model_DT.predict(X_test)

            mape_DT = mean_absolute_percentage_error(y_test, y_DT)
            st.write("Hasil MAPE dari Decision Tree : ", mape_DT)

        elif option == 'MLP':

            st.write("MLP adalah sebuah metode klasifikasi dengan menggunakan algoritma backpropagation untuk menghitung error dan mengupdate bobot.")

            model_MLP = MLPClassifier()
            model_MLP.fit(X_train, y_train)
            y_MLP = model_MLP.predict(X_test)
            filename_MLP = 'model_MLP.sav'
            pickle.dump(model_MLP, open(filename_MLP, 'wb'))

            mape_MLP = mean_absolute_percentage_error(y_test, y_MLP)
            st.write("Hasil MAPE dari MLP : ", mape_MLP)

        submit2 = st.form_submit_button(label='Submit')


with page4:
    st.title("Prediksi")
    st.write("Masukkan data yang ingin diprediksi")
    with st.form(key='form4'):
        option = st.selectbox('Pilih Model yang diinginkan : ',
                              ('Naive Bayes', 'Decision Tree', 'MLP'))
        input1 = st.number_input('Masukkan data ke-1 : ')
        input2 = st.number_input('Masukkan data ke-2 : ')
        input3 = st.number_input('Masukkan data ke-3 : ')

        input = np.array([input1, input2, input3])
        input = input.reshape(1, -1)
        input = scaler.transform(input)
        input = pca.transform(input)

        submit3 = st.form_submit_button(label='Submit')

        if submit3:
            if option == 'Naive Bayes':
                model_NB = pickle.load(open('model/model_NB.sav', 'rb'))
                prediksi = model_NB.predict(input)
                st.write("Prediksi harga saham : ", prediksi)

            elif option == 'Decision Tree':
                model_DT = pickle.load(open('model/model_DT.sav', 'rb'))
                prediksi = model_DT.predict(input)
                st.write("Prediksi harga saham : ", prediksi)

            elif option == 'MLP':
                model_MLP = pickle.load(open('model/model_MLP.sav', 'rb'))
                prediksi = model_MLP.predict(input)
                st.write("Prediksi harga saham : ", prediksi)

        # if option == 'Naive Bayes':
        #     st.write("Naive Bayes adalah sebuah metode klasifikasi menggunakan metode probabilitas dan statistik yg dikemukakan oleh ilmuwan Inggris Thomas Bayes.")
        #     model_NB = pickle.load(open('model_NB.sav', 'rb'))
        #     scaler = pickle.load(open('scaler.sav', 'rb'))
        #     pca = pickle.load(open('pca.sav', 'rb'))

        #     prediksi = model_NB.predict(input)

        # elif option == 'Decision Tree':
        #     st.write("Decision Tree adalah sebuah metode klasifikasi dengan membuat sebuah pohon keputusan dari data pembelajaran yang dimiliki.")
        #     model_DT = pickle.load(open('model_DT.sav', 'rb'))
        #     scaler = pickle.load(open('scaler.sav', 'rb'))
        #     pca = pickle.load(open('pca.sav', 'rb'))

        #     input1 = st.number_input('Masukkan data ke-1 : ')
        #     input2 = st.number_input('Masukkan data ke-2 : ')
        #     input3 = st.number_input('Masukkan data ke-3 : ')

        #     input = np.array([input1, input2, input3])
        #     input = input.reshape(1, -1)
        #     input = scaler.transform(input)
        #     input = pca.transform(input)

        #     prediksi = model_DT.predict(input)

        # elif option == 'MLP':
        #     st.write("MLP adalah sebuah metode klasifikasi dengan menggunakan algoritma backpropagation untuk menghitung error dan mengupdate bobot.")
        #     model_MLP = pickle.load(open('model_MLP.sav', 'rb'))
        #     scaler = pickle.load(open('scaler.sav', 'rb'))
        #     pca = pickle.load(open('pca.sav', 'rb'))

        #     input1 = st.number_input('Masukkan data ke-1 : ')
        #     input2 = st.number_input('Masukkan data ke-2 : ')
        #     input3 = st.number_input('Masukkan data ke-3 : ')

        #     input = np.array([input1, input2, input3])
        #     input = input.reshape(1, -1)
        #     input = scaler.transform(input)
        #     input = pca.transform(input)

        #     prediksi = model_MLP.predict(input)