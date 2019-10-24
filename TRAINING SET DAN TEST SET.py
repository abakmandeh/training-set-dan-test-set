# Mengimpor library yang diperlukan # TRAINING SET DAN TEST SET
import numpy as np
import pandas as pd
 
# Import data ke python
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values  #menentukan X sebagai variable dependen
y = dataset.iloc[:, 3].values    #menentukan y sebagai variable independen
 
# Memproses data yang hilang (missing)
from sklearn.impute import SimpleImputer #mengisi data yg hilang(nan) dgn rata2(mean)
imputer = SimpleImputer(missing_values= np.nan, strategy = 'mean') #memilih usia dan gaji, sehingga kita memilih X index 1 dan 2
imputer = imputer.fit(X[:, 1:3]) #implementasi kebasris yg hilang
X[:, 1:3] = imputer.transform(X[:, 1:3])
 
# Encoding(konversi jd angka) data kategori : variable dependen(negara) dan variabel independen(Beli)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()                     # Bisa dihilangkan, baca pembahasan di bawahnya
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])     # Bisa dihilangkan, baca pembahasan di bawahnya
transformer = ColumnTransformer(
        [('Negara', OneHotEncoder(), [0])],
        remainder='passthrough')
X = np.array(transformer.fit_transform(X), dtype=np.float)
 
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Membagi menjadi training set dan test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#X_train sebagai data X yang ingin kita latih (train), 
#X_test sebagai data X yang ingin kita tes, 
#y_train adalah variabel dependen yang dilatih, dan 
#y_test merupakan variabel dependen yang diuji