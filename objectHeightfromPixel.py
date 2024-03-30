import cv2

# Baca gambar
image = cv2.imread('objek.jpg')

# Tentukan titik referensi pada objek (misalnya, bagian atas atau bawah objek)
titik_atas = (100, 50)  # Contoh titik atas objek dalam piksel (x, y)

# Tentukan titik lain di objek (misalnya, bagian bawah objek)
titik_bawah = (100, 150)  # Contoh titik bawah objek dalam piksel (x, y)

# Hitung tinggi objek dalam piksel
tinggi_objek_piksel = abs(titik_bawah[1] - titik_atas[1])

print("Tinggi objek dalam piksel:", tinggi_objek_piksel)
