import streamlit as st
import pandas as pd
from sequence_split import split_sequence
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from datetime import datetime

X_train, X_test, y_train, y_test = None, None, None, None
def main():
    st.sidebar.title("Menu")
    menu = ["Data", "Pre processing data", "Modelling", "Implementasi"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Data":
        st.title("Prediksi Data Time Series BTN")
        st.markdown("<h4>Data</h4>", unsafe_allow_html=True)
        data = pd.read_csv('https://raw.githubusercontent.com/wahyuarilsaputra/dataset/main/BBTN.JK.csv')
        
        st.write(data)
        if st.button("Penjelasan"):
            st.markdown("<br><hr><br>", unsafe_allow_html=True)
            st.markdown(
                "<h3 style='text-align:justify'>Dataset Finance Bank BTN</h3>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<h3 style='text-align:justify'>Halaman Menu Sistem</h3>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<h5 style='text-align:justify'>1. Data : Menampilkan Data</h5>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<h5 style='text-align:justify'>2. Pre Processing Data : Menampilkan data yang sudah di pre processing </h5>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<h5 style='text-align:justify'>3. Modelling : Menampilkan proses modeling data</h5>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<h5 style='text-align:justify'>4. Implementasi : Input data untuk prediksi data baru</h5>",
                unsafe_allow_html=True,
            )
        

    elif choice == "Pre processing data":
        st.title("Prediksi Data Time Series BTN")
        st.markdown("<h4>Pre processing data</h4>", unsafe_allow_html=True)
        selected_column = st.selectbox("Pilih Kolom", ('Open','High', 'Low', 'Close'))
        data = pd.read_csv('https://raw.githubusercontent.com/wahyuarilsaputra/dataset/main/BBTN.JK.csv')
        sequence = data[selected_column].astype(int).values
        n_steps = 3
        X, y = split_sequence(sequence, n_steps)

        st.markdown("<h6>Data Setelah Di sequence</h6>", unsafe_allow_html=True)
        newFitur = pd.DataFrame(X, columns=['t-'+str(i+1) for i in range(n_steps-1, -1, -1)])
        newTarget = pd.DataFrame(y, columns=['Data Prediksi'])
        newData = pd.concat([newFitur, newTarget], axis=1)
        st.write(pd.DataFrame(newData, columns=newData.columns))


        st.markdown("<h6>Pre Processing Data</h6>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("<h6>Data Setelah Dinormalisasi</h6>", unsafe_allow_html=True)
            scaler = MinMaxScaler()
            X_norm = scaler.fit_transform(newFitur)
            st.write(pd.DataFrame(X_norm, columns=newFitur.columns))
            # st.empty()  # Menggunakan st.empty() untuk mengatur tata letak di kolom pertama
        
        with col2:
            st.markdown("<h6>Data Setelah Dilakukan PCA</h6>", unsafe_allow_html=True)
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_norm)
            st.write(pd.DataFrame(X_pca, columns=['PCA Component 1', 'PCA Component 2']))
        
        X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=0)
        st.markdown("<h3>Modelling</h3>", unsafe_allow_html=True)
        selected_model = st.selectbox("Pilih Model", ('Decision Tree', 'KNN', 'MLP', 'Naive Bayes'))
        if selected_model == "Decision Tree":
            model = DecisionTreeClassifier()
        elif selected_model == "KNN":
            model = KNeighborsClassifier()
        elif selected_model == "MLP":
            model = MLPClassifier()
        elif selected_model == "Naive Bayes":
            model = GaussianNB()
        y_pred = model.fit(X_train, y_train)
        accuracy = mean_absolute_percentage_error(y_pred, y_test)
        # accuracy = model.score(X_test, y_test)
        st.write("Akurasi model:", accuracy)

    elif choice == "Implementasi":
        st.title("Prediksi Data Time Series BTN")
        st.markdown("Implementasi")
        # Tambahkan kode untuk halaman Implementasi di sini
        input_date = st.date_input("Date", value=datetime.today())
        formatted_date = input_date.strftime("%Y-%m-%d")
        input_row = None
        if st.button("Prediksi"):
             input_row = data[data['Date'] == input_date]

        if input_row.empty:
            st.write("Data tidak ditemukan")
        else:
            # Ambil nilai Open, High, Low, dan Close dari baris data yang ditemukan
            open_value = input_row['Open'].values[0]
            high_value = input_row['High'].values[0]
            low_value = input_row['Low'].values[0]
            close_value = input_row['Close'].values[0]

            # Bentuk array fitur sesuai dengan format yang diharapkan oleh model
            input_features = np.array([[open_value, high_value, low_value, close_value]])

            # Normalisasi fitur input
            input_features_norm = scaler.transform(input_features)

            # Transformasi fitur input menggunakan PCA
            input_features_pca = pca.transform(input_features_norm)

            # Lakukan prediksi menggunakan model
            prediction = model.predict(input_features_pca)

            # Tampilkan hasil prediksi
            st.write("Hasil Prediksi:")
            st.write("- Open: ", prediction[0, 0])
            st.write("- High: ", prediction[0, 1])
            st.write("- Low: ", prediction[0, 2])
            st.write("- Close: ", prediction[0, 3])

if __name__ == '__main__':
    main()
