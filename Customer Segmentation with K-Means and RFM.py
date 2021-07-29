#######################################  Customer Segmentation with K-Means  ########################################

# Gerekli Import İşlemleri

import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.max_rows', None)

# Veri Seti
df_ = pd.read_excel("datasets/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()

# Veri setinin betimsel istatistikleri
df.describe().T

# Eksik gözlem
df.isnull().sum()

# eksik değerler
df.dropna(inplace=True)

#  İptal edilen işlemleri veri setinden çıkarma
df = df[~df["Invoice"].str.contains("C", na=False)]

# TotalPrice Hesaplanması
df["TotalPrice"] = df["Quantity"] * df["Price"]
df.head()

# Recency Hesabı için today date Belirlenmesi
today_date = dt.datetime(2011, 12, 11)

# RFM Metriklerinin Oluşturulması
rfm = df.groupby('Customer ID').agg({'InvoiceDate': lambda InvoiceDate: (today_date - InvoiceDate.max()).days,
                                     'Invoice': lambda Invoice: Invoice.nunique(),
                                     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

# Oluşturulan Metriklerin İsimlendirilmesi
rfm.columns = ['recency', 'frequency', 'monetary']

# Monetary Değerleri 0 'dan büyük olan Gözlemlerin Seçilmesi
rfm = rfm[rfm["monetary"] > 0]

## K-MEANS İLE KARŞILAŞTIRILMA YAPILABİLMESİ İÇİN ÖNCELİKLE RFM SEGMENTLERİNİ OLUŞTURACAĞIZ
# RFM Metriklerinin Skorlara Dönüştürülmesi
rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

# RFM Skorlarının İsimlendirilmesi için Recency ve Frequency Metriklerinin Seçilmesi
rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str))

# İsimlendirilme için Oluşturulan Regex Kodları
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

# RF Skorlarının regex yardımıyla isimlendirilmesi
rfm['segment'] = (rfm['recency_score'].astype(str) +rfm['frequency_score'].astype(str)).replace(seg_map, regex=True)
rfm.head()

### K-MEANS HESABI İŞLEMLERİ

# K-Means Değerleri için Recency,Frequency ve Monetary Değerlerini Seçerek İşlem Yapacağız.
df = rfm.loc[:,"recency":"monetary"]

# Uzaklık temelli bir yöntem olduğundan dolayı ölçüm problemlerinin önüne geçmek için veri setini 0-1 aralığına dönüştürdük.
sc = MinMaxScaler((0,1))
df = sc.fit_transform(df)
df[0:5]

# K-Means model nesnesini oluşturup fit ediyoruz. Ön tanımlı değerleri gözlemliyoruz.
kmeans = KMeans()
k_fit = kmeans.fit(df)

k_fit.get_params()
# CLuster sayısı
k_fit.n_clusters
# Cluster Merkezleri
k_fit.cluster_centers_
# Oluşturulan clusterlar
k_fit.labels_
# Toplam hata değeri(SSE)
k_fit.inertia_

# Kümelerin Görselleştirilmesi

k_means = KMeans(n_clusters=10).fit(df)
kumeler = k_means.labels_
type(df)
df = pd.DataFrame(df)


plt.scatter(df.iloc[:, 0],
            df.iloc[:, 1],
            c=kumeler,
            s=50,
            cmap="viridis")
plt.show()

# Merkezlerin İşaretlenmesi
merkezler = k_means.cluster_centers_

plt.scatter(df.iloc[:, 0],
            df.iloc[:, 1],
            c=kumeler,
            s=50,
            cmap="viridis")

plt.scatter(merkezler[:, 0],
            merkezler[:, 1],
            c="red",
            s=200,
            alpha=0.8)
plt.show()

# Optimum küme sayısının bulunması

# Brute Force Yöntemiyle Küme Sayısı Bulma

ssd = []
K = range(1, 30)

# Oluşturulacak K adet kümenin hatalarının toplamı

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df)
    ssd.append(kmeans.inertia_)

# Kümelerin hata kareler toplamı
ssd

# Elbow Yöntemiyle Optimum Küme SAyısı Belirlenmesi
plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık Uzaklık Artık Toplamları")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
elbow.show()

elbow.elbow_value_

# Biz RFM Segmentleriyle aynı sayıda olması için 10 küme oluşturmak istiyoruz.
kmeans = KMeans(n_clusters=10).fit(df)

# Elde ettiğimiz Labelları Segmentler Olarak Atama Yapıyoruz.
segmentler = kmeans.labels_

# Dataframe'e çeviriyoruz.
df = pd.DataFrame(df)

# Müşterilerin Yer aldıkları Segmentleri Gözlemliyoruz.
pd.DataFrame({"Müşteriler": df.index, "Segmentler": segmentler})
rfm.head()

# Oluşturulan Segmentleri cluster_no adında bir değişken olarak veri setimize ekliyoruz.
rfm["cluster_no"] = segmentler

# Segmentler içerisindeki 0 segmentinden kurtuluyoruz. Hepsi 1-10 arasında gözlemlenecek
rfm["cluster_no"] = rfm["cluster_no"] + 1

rfm.head(20)

# Oluşturulan clusterların dağılımını gözlemek için groupby a alıyoruz.
rfm.groupby("cluster_no").agg({"cluster_no": "count"})

# Oluşturulan Cluster ve RFM Segmentlerinin Karşılaştırılması
rfm.groupby(["cluster_no","segment"]).agg(["mean","count"])



















