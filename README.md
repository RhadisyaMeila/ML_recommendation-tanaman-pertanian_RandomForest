# Laporan Proyek Machine Learning: Sistem Rekomendasi Tanaman Pertanian
Penulis: Rhadisya Meila Parmanti & Nazwa Akmalia Padla
# Project Overview
Pertanian merupakan sektor penting yang berperan dalam memenuhi kebutuhan pangan masyarakat. Salah satu tantangan utama dalam bidang pertanian adalah menentukan jenis tanaman yang paling sesuai dengan kondisi lahan dan lingkungan. Kesalahan dalam pemilihan tanaman dapat menyebabkan produktivitas rendah, penggunaan pupuk yang tidak optimal, serta kerugian ekonomi bagi petani.

Dengan memanfaatkan Machine Learning, proses pemilihan tanaman dapat dilakukan secara otomatis berdasarkan karakteristik tanah dan kondisi lingkungan. Pada proyek ini dibangun sistem rekomendasi tanaman menggunakan algoritma Random Forest Classifier untuk memprediksi jenis tanaman yang paling sesuai berdasarkan kandungan unsur hara tanah (N, P, K), suhu, kelembaban, pH, dan curah hujan.

💡 Manfaat Proyek:
1. Membantu petani menentukan tanaman yang sesuai dengan kondisi lahan.
2. Mengurangi risiko kesalahan dalam pemilihan tanaman.
3. Mendukung pengambilan keputusan berbasis data pada sektor pertanian.

Format Referensi: [SISTEM REKOMENDASI TANAMAN](https://ojs.trigunadharma.ac.id/index.php/jsi/article/view/12348)

# Business Understanding
Problem Statements
1. Bagaimana membangun sistem yang mampu merekomendasikan jenis tanaman berdasarkan kondisi tanah dan lingkungan?
2. Seberapa baik algoritma Random Forest dalam melakukan klasifikasi jenis tanaman berdasarkan fitur lingkungan?

# Goals
1. Mengembangkan model Machine Learning yang mampu memprediksi jenis tanaman dengan akurasi tinggi.
2. Mengevaluasi performa Random Forest dalam menyelesaikan masalah klasifikasi multi-kelas pada data pertanian.

# Data Understanding
Dataset yang digunakan adalah Crop Recommendation Dataset yang berisi 2.200 sampel data.
1. Fitur: N, P, K, temperature, humidity, ph, rainfall.
2. Target: label (terdiri dari 22 jenis tanaman).
```
# Memuat dataset
try:
    df = pd.read_csv('Crop_recommendation.csv')
    print(df.head())
except FileNotFoundError:
    print("Dataset tidak ditemukan.")
  ```
4. Hasil Analisis: Dataset dalam kondisi sangat baik, tidak memiliki missing values, dan setiap jenis tanaman memiliki distribusi data yang seimbang (100 sampel per kelas). Analisis korelasi menunjukkan tidak terdapat multikolinearitas yang signifikan antar fitur.
```
print("\n--- Informasi Struktur Data & Missing Values ---")
print(df.info())
print("\nJumlah Missing Values per Kolom:")
print(df.isnull().sum())
```
fitur paling berpengaruh terhadap rekomendasi
<img width="984" height="583" alt="Feature Importance (fitur palin berpengaruh terhadap rekomendasi)" src="https://github.com/user-attachments/assets/96527a01-3d47-4d0c-8640-d013d43e9294" />

Distribusi jumlah sampel per jenis tanaman
<img width="1484" height="584" alt="distribusi jumlah sampel per jenis tanaman" src="https://github.com/user-attachments/assets/543c0d63-604d-4496-b252-2d5d70b3847c" />
Confusion matrix rekomendai tanaman

<img width="473" height="402" alt="image" src="https://github.com/user-attachments/assets/a2a79247-d87a-426e-8091-a684f8b8a19f" />


# Data Preparation
1. Pemisahan Data: Dataset dibagi menjadi Training (80%) dan Testing (20%) menggunakan stratified split untuk menjaga proporsi kelas tetap konsisten.
2. Feature Scaling: Seluruh fitur numerik distandarisasi menggunakan *StandardScaler* untuk memastikan performa model maksimal.

```
# Split Dataset 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardisasi fitur
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```


# Modeling and Results
penelitian ini menggunakan Random Forest Classifier dengan optimasi hyperparameter.
```
# Tuning parameter dengan GridSearchCV
param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [10, 20, None]}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
grid_search.fit(X_train_scaled, y_train)
```
1. Hyperparameter Tuning: Hasil optimasi menggunakan GridSearchCV (3-Fold CV) memberikan konfigurasi terbaik: n_estimators=150, max_depth=10, min_samples_split=2.
2. Hasil Akhir: Model mencapai akurasi 99,32%.

# Evaluation
1. Akurasi: 99,32% pada data pengujian.
2. Klasifikasi: Nilai precision, recall, dan f1-score rata-rata sebesar 0.99, menunjukkan performa yang sangat stabil.
3. Analisis: Fitur lingkungan (kelembapan dan curah hujan) memiliki pengaruh signifikan dalam penentuan rekomendasi tanaman.
<img width="1467" height="1183" alt="confusion matrix - sistem rekomendasi tanaman" src="https://github.com/user-attachments/assets/00d1f78e-ad8e-4820-84b0-61e42e0588c0" />


# Kesimpulan & Deployment
# Kesimpulan
Sistem rekomendasi tanaman ini berhasil dikembangkan dengan akurasi tinggi (99,32%). Penggunaan Random Forest terbukti sangat efektif untuk mengklasifikasikan 22 jenis tanaman berdasarkan variabel lingkungan. Pengerjaan proyek ini menjadi pengalaman berharga bagi kami dalam memahami alur kerja data science secara menyeluruh, mulai dari penyiapan data hingga deployment. Kami berharap proyek ini dapat menjadi langkah awal bagi kami untuk terus menciptakan solusi teknologi yang bermanfaat bagi masyarakat luas.
<img width="959" height="468" alt="ML_rec-tanaman-pertanian_RandomForest_hasil" src="https://github.com/user-attachments/assets/4e4ccbc9-c627-4579-b393-5bed09d19df8" />


#Deployment
1. Model Export: Model dan objek scaler disimpan dalam format .pkl (model_rf.pkl & scaler.pkl).
2. Implementasi: Sistem telah di-deploy pada platform Hugging Face Spaces, memungkinkan pengguna untuk mendapatkan rekomendasi secara real-time.
