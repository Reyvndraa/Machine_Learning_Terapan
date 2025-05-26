# Laporan Proyek Machine Learning - Yuda Reyvandra Herman

## Domain Proyek

Stroke atau penyakit serebrovaskular merupakan gangguan fungsi otak yang terjadi secara tiba-tiba akibat gangguan aliran darah ke otak, baik karena sumbatan (iskemik) maupun pecahnya pembuluh darah (hemoragik). Stroke menjadi salah satu penyebab utama kematian dan kecacatan jangka panjang di seluruh dunia. Berdasarkan laporan World Health Organization (WHO), sekitar 15 juta orang mengalami stroke setiap tahun, dan lebih dari 5 juta orang meninggal dunia, sementara jutaan lainnya mengalami kecacatan permanen yang mengubah kualitas hidup mereka secara drastis (WHO, 2023).

Di Indonesia, stroke menjadi penyebab kematian tertinggi menurut data dari Riskesdas (Riset Kesehatan Dasar) tahun 2018. Prevalensi stroke di Indonesia mencapai 10,9 per 1.000 penduduk, dengan faktor risiko utama seperti hipertensi, diabetes mellitus, merokok, obesitas, serta pola makan yang tidak sehat (Kementerian Kesehatan RI, 2018). Peningkatan usia harapan hidup dan perubahan gaya hidup masyarakat modern turut menyebabkan tren peningkatan kasus stroke dari tahun ke tahun.

Deteksi dini terhadap risiko stroke sangat penting dalam upaya pencegahan dan pengurangan dampak yang ditimbulkan. Pendekatan tradisional yang hanya mengandalkan pemeriksaan klinis kadang kala tidak cukup untuk mengidentifikasi risiko secara cepat dan tepat. Oleh karena itu, penggunaan teknologi berbasis data seperti machine learning menjadi solusi potensial dalam mengembangkan sistem prediktif yang mampu menganalisis berbagai faktor risiko secara komprehensif dan akurat.

\*\*Referensi

- World Health Organization (WHO). (2023). Stroke: Key facts. Retrieved from: https://www.who.int/news-room/fact-sheets/detail/stroke
- Kementerian Kesehatan Republik Indonesia. (2018). Laporan Nasional Riskesdas 2018. Badan Penelitian dan Pengembangan Kesehatan. Retrieved from: https://www.litbang.kemkes.go.id/laporan-riset-kesehatan-dasar-riskesdas-2018/

## Business Understanding

Dalam dunia kesehatan, tantangan utama dalam penanganan stroke adalah deteksi dan pencegahan dini. Karena stroke bisa terjadi secara mendadak dan berdampak serius, identifikasi pasien berisiko tinggi secara cepat dan akurat sangat penting untuk menurunkan angka kematian dan kecacatan.

Masalah ini muncul dari kebutuhan tenaga medis untuk mengenali pasien berisiko tinggi berdasarkan data historis kesehatan. Pendekatan konvensional sering kurang efektif dalam menyaring risiko secara luas dan cepat.

Dengan memanfaatkan data seperti usia, tekanan darah, kadar glukosa, BMI, dan gaya hidup, dapat dibangun model prediktif berbasis data. Model ini diharapkan menjadi sistem peringatan dini yang mendukung pengambilan keputusan medis dan strategi pencegahan stroke.

### Problem Statements

- Bagaimana mengidentifikasi variabel-variabel yang paling berpengaruh terhadap risiko stroke pada pasien berdasarkan data historis kesehatan mereka?
- Bagaimana membangun model prediksi berbasis machine learning yang mampu mengklasifikasikan pasien berisiko tinggi dan rendah terhadap stroke?
- Apakah sistem prediksi stroke yang dikembangkan dapat diintegrasikan ke dalam sistem informasi rumah sakit atau layanan kesehatan untuk mendukung pengambilan keputusan klinis

### Goals

- Mengidentifikasi variabel paling berpengaruh terhadap risiko stroke.
  Dengan melakukan eksplorasi dan analisis data, proyek ini bertujuan menemukan faktor-faktor kunci seperti usia, hipertensi, kadar glukosa, dan status merokok yang paling memengaruhi kemungkinan seseorang terkena stroke.
- Membangun model prediksi stroke berbasis machine learning.
  Proyek ini bertujuan mengembangkan model klasifikasi yang akurat dan andal untuk memprediksi apakah seseorang berisiko mengalami stroke berdasarkan data kesehatan mereka.
- Menghasilkan sistem prediksi yang dapat diterapkan dalam praktik layanan kesehatan.
  Model yang dibangun diharapkan tidak hanya akurat, tetapi juga dapat diintegrasikan ke dalam sistem informasi rumah sakit atau aplikasi kesehatan untuk membantu pengambilan keputusan medis secara real-time.

Semua poin di atas harus diuraikan dengan jelas. Anda bebas menuliskan berapa pernyataan masalah dan juga goals yang diinginkan.

### Solution statements

- Menerapkan dan membandingkan beberapa algoritma machine learning, seperti Logistic Regression, Support Vector Machine dan Artificial Intelligence
- Menggunakan feature importance untuk mengetahui fitur yang paling penting
- Masing-masing model akan dievaluasi menggunakan metrik seperti: Accuracy, Precision, Recall, F1-score, ROC-AUC

## Data Understanding

Dataset yang digunakan berjudul “Healthcare Dataset Stroke Data”, terdiri dari 5110 data pasien dengan berbagai informasi terkait faktor risiko stroke. Fitur-fitur yang tersedia mencakup usia, jenis kelamin, riwayat hipertensi, penyakit jantung, kadar glukosa, BMI, status merokok, jenis pekerjaan, dan label target stroke yang menunjukkan apakah pasien pernah mengalami stroke.

| Keterangan   | Detail                                                                                                      |
| ------------ | ----------------------------------------------------------------------------------------------------------- |
| Jumlah Data  | 5.110 baris                                                                                                 |
| Jumlah Fitur | 11 kolom                                                                                                    |
| Target       | `Stroke` (Yes / No)                                                                                         |
| Format       | CSV                                                                                                         |
| Sumber       | [Kaggle - Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) |

### Fitur

| **Fitur**         | **Keterangan**                                                            |
| ----------------- | ------------------------------------------------------------------------- |
| id                | Unique identifier buat semua orang                                        |
| gender            | 0 atau 1 (male or female)                                                 |
| age               | Umur                                                                      |
| hypertension      | 0 atau 1 (pernah atau tidak)                                              |
| heart_disease     | 0 atau 1 (pernah atau tidak)                                              |
| ever_married      | 0 atau 1 (pernah atau tidak)                                              |
| work_type         | Tipe pekerjaan (Private, self-employed, govt-job, children, never worked) |
| Residence_type    | Urban atau Rural                                                          |
| avg_glucose_level | Kadar glukosa                                                             |
| bmi               | Berat Badan                                                               |
| smoking_status    | Kategori perokok (formerly smoked, never smoked, smokes, unknown)         |
| stroke            | 0 atau 1 (ya atau tidak)                                                  |

### Exploratory Data Analysis
Gambar barchart dibawah menunjukkan distribusi target 'Stroke'

![Distribusi Target Stroke](img/d1.png)

### Correlation Heatmap
Menampilkan korelasi antar fitur numerik dengan target 'Stroke'

![Correlation Heatmap](img/d2.png)

- Usia ('age') menunjukkan korelasi positif tertinggi dengan stroke (0.25) dibandingkan fitur lain, menunjukkan bahwa semakin tua usia seseorang, semakin tinggi kemungkinan untuk mengalami stroke.

## Data Preparation

### Data Cleaning
- Menghapus kategori 'other' pada kolom 'gender', karena hanya ada 2 gender saja di dunia ini, yaitu laki-laki dan perempuan
- Mengisi missing value pada kolom 'bmi' menggunakan median

### Data Preprocessing
- Mengubah fitur kategorikal menjadi fitur numerikal menggunakan LabelEncoder
- Melakukan feature scaling pada fitur numerikal dengan metode standarisasi (z-score)
  
### Data Splitting
- Melakukan pemisahan data fitur (X) dan label (y)
- Membagi data latih dan data uji menjadi 8:2 

### Data Balancing 
- Menerapkan SMOTE untuk mem-balance data yang ada, karena data sebelumnya sangat imbalance sehingga model nantinya akan cenderung memprediksi kelas mayoritas
- SMOTE hanya untuk data training


## Modeling
Pada tahap Modeling, digunakan tiga algoritma: Logistic Regression, SVM, dan ANN, dengan data yang telah diseimbangkan menggunakan SMOTE untuk mengatasi ketimpangan kelas antara kasus stroke dan non-stroke.

### Logistic Regression
Kelebihan:
- Sederhana dan mudah diinterpretasi: Cocok untuk memahami hubungan antar variabel.
- Cepat dilatih: Komputasinya ringan, cocok untuk dataset kecil-menengah.
- Bekerja baik jika hubungan antar fitur dan target bersifat linier.
- Output probabilistik: Menghasilkan probabilitas prediksi, berguna untuk klasifikasi berbasis ambang batas (thresholding).

Kekurangan:
- Tidak cocok untuk hubungan non-linier (kecuali dimodifikasi dengan polynomial features).
- Sensitif terhadap outlier dan multikolinearitas.
- Kurang akurat dibanding model kompleks jika data sangat kompleks.

### Support Vector Machine 
Kelebihan:
- Akurasi tinggi terutama pada data yang kompleks dan berdimensi tinggi.
- Efektif pada data non-linier dengan penggunaan kernel (misalnya RBF, polynomial).
- Robust terhadap overfitting terutama pada dataset dengan fitur banyak dan jumlah data terbatas.

Kekurangan:
-Lambat pada dataset besar (scalability buruk).
- Pemilihan kernel dan tuning parameter seperti C dan gamma cukup rumit.
- Sulit diinterpretasi, tidak cocok untuk aplikasi yang memerlukan transparansi model.

### Artificial Neural Network
Kelebihan:
- Sangat fleksibel dan mampu menangkap hubungan non-linier yang kompleks.
- Mampu belajar dari data besar dengan banyak fitur.
- Bisa menghasilkan prediksi yang sangat akurat jika dilatih dengan benar.
  
Kekurangan:
- Butuh waktu dan sumber daya komputasi besar.
- Tuning hyperparameter (jumlah neuron, layer, learning rate, dll.) bisa rumit.
- Kurang interpretatif (dikenal sebagai "black-box model").


## Evaluation
| **Metrik**    | **Deskripsi**                                                                                                                                                                                                            |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Accuracy**  | Rasio prediksi yang benar dari seluruh prediksi. Pada kasus ini, Logistic Regression memberikan akurasi tertinggi (77.98%), artinya model ini paling banyak menghasilkan prediksi yang benar secara keseluruhan.         |
| **Precision** | Dari seluruh prediksi stroke, berapa yang benar-benar stroke. SVM dan ANN punya precision rendah untuk kelas stroke, menandakan banyak false positive. Logistic Regression sedikit lebih baik, tapi tetap belum optimal. |
| **Recall**    | Dari seluruh kasus stroke asli, berapa banyak yang berhasil dideteksi. Logistic Regression punya **recall tinggi (70%)**, artinya cukup bagus dalam mendeteksi kasus stroke walau dengan risiko false positive.          |
| **F1-Score**  | Rata-rata harmonis antara precision dan recall. Karena model Logistic Regression punya balance antara keduanya, F1-nya (24%) jadi yang paling mending dibanding dua model lain.                                          |

### Hasil evaluasi model 
| Model                   | Kelas | Accuracy | Precision | Recall | F1-Score |
| ----------------------- | ----- | -------- | --------- | ------ | -------- |
| **Logistic Regression** | 0     | 0.7798   | 0.98      | 0.78   | 0.87     |
|                         | 1     |          | 0.14      | 0.70   | 0.24     |
| **SVM**                 | 0     | 0.7407   | 0.95      | 0.77   | 0.85     |
|                         | 1     |          | 0.04      | 0.18   | 0.06     |
| **ANN**                 | 0     | 0.1145   | 0.99      | 0.07   | 0.13     |
|                         | 1     |          | 0.05      | 0.98   | 0.10     |



### Confussion Matrix
**Logistic Regression**
![Confussion Matrix](img/d4.png)

**Support Vector Machine**
![Confussion Matrix](img/d5.png)

**Artificial Neural Network**
![Confussion Matrix](img/d6.png)


### ROC Comparison
![ROC Comparison](img/d7.png)


### Feature Importance 
Gambar ini menampilkan fitur apa yang paling penting untuk model ini menggunakan **Logistic Regression**

![Feature Importance](img/d3.png)

- Berdasarkan hasil pada gambar, fitur yang paling berpengaruh terhadap kemungkinan stroke adalah usia (age) dengan koefisien paling besar positif, menandakan bahwa semakin tua seseorang, semakin tinggi risikonya


**---Ini adalah bagian akhir laporan---**

_Catatan:_

- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
