import pandas as pd
import streamlit as st
import numpy as np
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
 

st.write(""" 
# Aplikasi Heart Disease Dataset 
By. Arshelia Romadhona (200411100053)
""")

st.write("----------------------------------------------------------------------------------")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Description", "Dataset", "Preprocessing", "Modeling", "Implementation"])

with tab1:
    st.subheader("""Pengertian""")
    st.write("""
    Dataset ini digunakan untuk menentukan apakah pasien menderita penyakit jantung atau tidak.
    """)

    st.markdown(
        """
        Dataset ini memiliki beberapa fitur yaitu :
        - Age : umur pasien.
        - Sex : Jenis kelamin pasien.
        - Cp  : tipe nyeri dada yang diderita pasien. Atribut ini memilik 4 nilai, yaitu : Tipe 1 : typical angina, Tipe 2 : atypical angina, Tipe 3 : non-anginal pain, Tipe 4 : asymptomatic
        - Trestbps :resting blood pressure yaitu tekanan darah pasien ketika dalam keadaan istirahat. Satuan yang dipakai adalah mm Hg.
        - Chol :Cholesterol yaitu kadar kolesterol dalam darah pasien, dengan satuan mg/dl.
        - Fbs :fasting blood sugar yaitu kadar gula darah pasien, atribut fbs ini hanya memiliki 2 nilai yaitu 1 jika kadar gula darah pasien lebih dari 120 mg/dl, dan 0 jika kadar gula darah pasien kurang dari sama dengan 120 mg/dl.
        - Restecg :resting electrocardiographic yaitu kondisi ECG pasien ketika dalam keadaan istirahat. Atribut ini memiliki 3 nilai yaitu nilai 0 untuk keadaan normal, nilai 1 untuk keadaan ST-T wave abnormality yaitu keadaan dimana gelombang inversions T dan atau ST meningkat maupun menurun lebih dari 0,5 mV, dan nilai 2 untuk keadaan dimana ventricular kiri mengalami hipertropi.
        - Thalach : rata-rata detak jantung pasien.
        - Exang : keadaan dimana pasien akan mengalami nyeri dada apabila berolah raga, 0 jika tidak nyeri, dan 1 jika menyebabkan nyeri.
        - Oldpeak : penurunan ST akibat olah raga.
        """
    )

    st.subheader("""Dataset""")
    st.write("""
    Dataset penyakit jantung ini diambil dari Kaggle 
    <a href="https://www.kaggle.com/datasets/yasserh/heart-disease-dataset">Dataset</a>""", unsafe_allow_html=True)

with tab2:
    st.subheader("""Heart Disease Dataset""")
    df = pd.read_csv('https://raw.githubusercontent.com/arshell19/datamining/main/heart.csv')
    st.dataframe(df) 

with tab3:
    st.subheader("""Rumus Normalisasi Data""")
    st.image('https://i.stack.imgur.com/EuitP.png', use_column_width=False, width=250)
    st.markdown("""
    Keterangan :
    - X = data yang akan dinormalisasi atau data asli
    - min = nilai minimum semua data asli
    - max = nilai maksimum semua data asli
    """)
    #Mendefinisikan Varible X dan Y
    X = df.drop(columns=['slope','target','ca','thal'])
    y = df['target'].values
    df_min = X.min()
    df_max = X.max()
    
    #NORMALISASI NILAI X
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    scaled_features = pd.DataFrame(scaled, columns=features_names)

    st.subheader('Hasil Normalisasi Data')
    st.write(scaled_features)

    st.subheader('Target Label')
    dumies = pd.get_dummies(df.target).columns.values.tolist()
    dumies = np.array(dumies)

    labels = pd.DataFrame({
        'Positive' : [dumies[1]],
        'Negative' : [dumies[0]]
    })

    st.write(labels)

with tab4:
    st.subheader("""Metode Yang Digunakan""")
    #Split Data 
    training, test = train_test_split(scaled_features,test_size=0.2, random_state=1)#Nilai X training dan Nilai X testing
    training_label, test_label = train_test_split(y, test_size=0.2, random_state=1)#Nilai Y training dan Nilai Y testing
    with st.form("modeling"):
        st.write("Pilih Metode yang digunakan : ")
        naive = st.checkbox('Gaussian Naive Bayes')
        k_nn = st.checkbox('K-NN')
        destree = st.checkbox('Decission Tree')
        submitted = st.form_submit_button("Submit")

        #Gaussian Naive Bayes
        gaussian = GaussianNB()
        gaussian = gaussian.fit(training, training_label)

        probas = gaussian.predict_proba(test)
        probas = probas[:,1]
        probas = probas.round()

        gaussian_akurasi = round(100 * accuracy_score(test_label,probas))

        #KNN
        K=10
        knn=KNeighborsClassifier(n_neighbors=K)
        knn.fit(training,training_label)
        knn_predict=knn.predict(test)

        knn_akurasi = round(100 * accuracy_score(test_label,knn_predict))

        #Decission Tree
        dt = DecisionTreeClassifier()
        dt.fit(training, training_label)
        # prediction
        dt_pred = dt.predict(test)
        #Accuracy
        dt_akurasi = round(100 * accuracy_score(test_label,dt_pred))

        if submitted :
            if naive :
                st.write('Naive Bayes accuracy score: {0:0.2f}'. format(gaussian_akurasi))
            if k_nn :
                st.write("K-NN accuracy score : {0:0.2f}" . format(knn_akurasi))
            if destree :
                st.write("Decision Tree accuracy score : {0:0.2f}" . format(dt_akurasi))
        
        grafik = st.form_submit_button("Grafik Akurasi")
        if grafik:
            data = pd.DataFrame({
                'Akurasi' : [gaussian_akurasi, knn_akurasi, dt_akurasi],
                'Model' : ['Gaussian Naive Bayes', 'K-NN', 'Decission Tree'],
            })

            chart = (
                alt.Chart(data)
                .mark_bar()
                .encode(
                    alt.X("Akurasi"),
                    alt.Y("Model"),
                    alt.Color("Akurasi"),
                    alt.Tooltip(["Akurasi", "Model"]),
                )
                .interactive()
            )
            st.altair_chart(chart,use_container_width=True)

with tab5:
        st.subheader("Form Implementasi")
        with st.form("my_form"):
            age = st.slider('Usia pasien', 29, 77)
            sex = st.slider('Jenis kelamin pasien', 0, 1)
            cp = st.slider('Tingkat CP pasien', 0, 3)
            bps = st.slider('Tingkat stres BPS pasien', 94, 200)
            chol = st.slider('Tingkat Kolestrol pasien', 126, 564)
            fbs = st.slider('Tingkat FBS pasien', 0, 1)
            restecg = st.slider('Tingkat EKG istirahat pasien', 0, 2)
            thalach = st.slider('Tingkat thalach pasien', 71, 202)
            exang = st.slider('Tingkat exang pasien', 0, 1)
            oldpeak = st.slider('Riwayat puncak lama pasien dicatat', 0.00, 6.20)
            model = st.selectbox('Model untuk prediksi',
                    ('Gaussian Naive Bayes', 'K-NN', 'Decision Tree'))

            prediksi = st.form_submit_button("Submit")
            if prediksi:
                inputs = np.array([
                    age,
                    sex,
                    cp,
                    bps,
                    chol,
                    fbs,
                    restecg,
                    thalach,
                    exang,
                    oldpeak
                ])
                
                df_min = X.min()
                df_max = X.max()
                input_norm = ((inputs - df_min) / (df_max - df_min))
                input_norm = np.array(input_norm).reshape(1, -1)

                if model == 'Gaussian Naive Bayes':
                    mod = gaussian
                if model == 'K-NN':
                    mod = knn 
                if model == 'Decision Tree':
                    mod = dt

                input_pred = mod.predict(input_norm)

                st.subheader('Hasil Prediksi')
                st.write('Menggunakan Pemodelan :', model)

                if input_pred == 1:
                    st.error('Positive')
                else:
                    st.success('Negative')