import numpy as np
import cv2
import glob

# Ambil beberapa gambar pola kalibrasi
image_paths = glob.glob("DETECTION_RANDOM_WITH_FRAME_LENGTH/gambar/*.jpg")

# Persiapan pola kalibrasi
pattern_size = (8, 6)  # Jumlah titik pada pola kalibrasi (kolom, baris)

# List untuk menyimpan sudut-sudut dari pola kalibrasi
object_points = []  # Koordinat objek dalam ruang 3D
image_points = []   # Koordinat sudut pola kalibrasi dalam gambar

# Inisialisasi koordinat objek
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

# Loop untuk setiap gambar pola kalibrasi
for image_path in image_paths:
    # Baca gambar
    image = cv2.imread(image_path)
    
    # Konversi ke citra keabuan
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Deteksi sudut pola kalibrasi
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    # Jika sudut-sudut ditemukan, tambahkan ke list
    if ret:
        object_points.append(objp)
        image_points.append(corners)
        
        # Kalibrasi kamera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp], [corners], gray.shape[::-1], None, None)

        # Focal length kamera
        focal_length_x = mtx[0, 0]
        focal_length_y = mtx[1, 1]

        print("Focal Length (x):", focal_length_x)
        print("Focal Length (y):", focal_length_y)
