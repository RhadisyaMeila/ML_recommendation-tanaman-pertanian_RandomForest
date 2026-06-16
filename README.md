# Laporan Proyek Machine Learning - Rhadisya Meila Parmanti & Nazwa Akmalia Padla
# Project Overview
Sistem Rekomendasi Tanaman Pertanian Menggunakan Random Forest
Pertanian merupakan sektor penting yang berperan dalam memenuhi kebutuhan pangan masyarakat. Salah satu tantangan utama dalam bidang pertanian adalah menentukan jenis tanaman yang paling sesuai dengan kondisi lahan dan lingkungan. Kesalahan dalam pemilihan tanaman dapat menyebabkan produktivitas rendah, penggunaan pupuk yang tidak optimal, serta kerugian ekonomi bagi petani.

Dengan memanfaatkan Machine Learning, proses pemilihan tanaman dapat dilakukan secara otomatis berdasarkan karakteristik tanah dan kondisi lingkungan. Pada proyek ini dibangun sistem rekomendasi tanaman menggunakan algoritma Random Forest Classifier untuk memprediksi jenis tanaman yang paling sesuai berdasarkan kandungan unsur hara tanah (N, P, K), suhu, kelembaban, pH, dan curah hujan.

💡 Manfaat Proyek:
1. Membantu petani menentukan tanaman yang sesuai dengan kondisi lahan.
2. Mengurangi risiko kesalahan dalam pemilihan tanaman.
3. Mendukung pengambilan keputusan berbasis data pada sektor pertanian.

# Business Understanding
- Problem Statements
1. Bagaimana membangun sistem yang mampu merekomendasikan jenis tanaman berdasarkan kondisi tanah dan lingkungan?
2. Seberapa baik algoritma Random Forest dalam melakukan klasifikasi jenis tanaman berdasarkan fitur lingkungan?
- Goals
1. Mengembangkan model Machine Learning yang mampu memprediksi jenis tanaman dengan akurasi tinggi.
2. Mengevaluasi performa Random Forest dalam menyelesaikan masalah klasifikasi multi-kelas pada data pertanian.

- Solution Approach
1. Menggunakan algoritma Random Forest Classifier sebagai model utama.
2. Melakukan Hyperparameter Tuning menggunakan GridSearchCV untuk memperoleh kombinasi parameter terbaik.
3. Mengevaluasi model menggunakan Accuracy Score, Classification Report, dan Confusion Matrix.

FORMAT REFERENSI: https://ojs.trigunadharma.ac.id/index.php/jsi/article/view/12348

# Data Understanding
Dataset yang digunakan adalah Crop Recommendation Dataset yang berisi 2.200 sampel data.
1. Fitur: N, P, K, temperature, humidity, ph, rainfall.
2. Target: label (terdiri dari 22 jenis tanaman).
3. Hasil Analisis: Dataset dalam kondisi sangat baik, tidak memiliki missing values, dan setiap jenis tanaman memiliki distribusi data yang seimbang (100 sampel per kelas). Analisis korelasi menunjukkan tidak terdapat multikolinearitas yang signifikan antar fitur.




