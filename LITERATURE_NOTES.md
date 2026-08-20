# Literature Notes and Claim Boundaries

Bu dosya raporda hangi iddianın hangi çalışmaya dayandırılabileceğini ve hangi
iddianın bu PoC tarafından **desteklenmediğini** özetler.

## Klinik çerçeve

### Sprecher et al. (1997)

- Beş seviyeli locomotion scoring yaklaşımında duruş ve gait birlikte
  değerlendirilir.
- Skor 1'de sırt ayakta ve yürürken düz; skor 2'de ayakta düz olup yürürken
  arched olabilir; skor 3'te arched posture ayakta ve yürürken görülür.
- Bu çalışma, `arched-back posture` hedefinin veterinerlik açısından anlamlı
  motivasyonunu sağlar; bizim görüntü etiketimizi klinik locomotion score'a
  dönüştürmez.

Kaynak: https://pubmed.ncbi.nlm.nih.gov/16728067/  
DOI: https://doi.org/10.1016/S0093-691X(97)00098-8

### Hoffman et al. (2014)

- Back-arched posture lameness ile ilişkilidir.
- Yazarların ana sonucu, bu postürlerin tek başına yüksek sensitivity veya
  specificity gösteren tanı testleri olmadığıdır.
- Çalışmadaki klinik sensitivity/specificity, bizim `görüntü → insan posture
  etiketi` modelimizin metrikleriyle doğrudan benchmark edilemez.

Kaynak: https://pubmed.ncbi.nlm.nih.gov/24508427/  
DOI: https://doi.org/10.3168/jds.2013-7528

## Görüntüden sırt ölçümü

### Poursaberi et al. (2010)

- Yan görünüş görüntülerinden back posture/shape analizi yapan temel 2D
  çalışmalardandır.
- Back curvature, hoof-contact anlarıyla birlikte ele alınmıştır.
- Bizim fixed-trim siluet feature'larımız bu çalışmanın birebir replikasyonu
  değildir; `literature-inspired experimental baseline` diye yazılmalıdır.

Kaynak: https://doi.org/10.1016/j.compag.2010.07.004

### Viazzi et al. (2013)

- Bireyselleştirilmiş back-posture ölçümü ve lameness sınıflandırmasını inceler.
- Bu çalışma 3D top-view çalışması değildir; 2014 karşılaştırmasıyla
  karıştırılmamalıdır.

Kaynak: https://pubmed.ncbi.nlm.nih.gov/23164234/  
DOI: https://doi.org/10.3168/jds.2012-5806

### Viazzi et al. (2014)

- Side-view 2D sistem ile top-view 3D sistemi karşılaştırır.
- 3D top-view, arka plan ve gölge kaynaklı 2D segmentasyon sorunlarına alternatif
  olarak ele alınmıştır.

Kaynak: https://doi.org/10.1016/j.compag.2013.11.005

### Jiang et al. (2022)

- Deep-learning destekli back localization/curvature feature yaklaşımıdır.
- `keypoint pose model` diye sınıflandırılmamalıdır; bizim contour/curvature
  hattının modern yakınlarından biridir.

Kaynak: https://doi.org/10.1016/j.compag.2022.106729

## Pose ve çoklu locomotion trait

### Russello et al. (2024)

- T-LEAP ile dokuz keypoint trajectory'si çıkarır.
- Back posture, head bob, tracking distance, stride length, stance duration ve
  swing duration gibi birden fazla locomotion trait kullanır.
- Tek trait'ten çoklu trait'e geçişin yararı, `kamburluk ≠ tam topallık modeli`
  sınırını destekler.

Kaynak: https://arxiv.org/abs/2401.05202

### Barney et al. (2023)

- Multi-cattle pose estimation ve lameness hattında back, head ve limb
  keypoint'lerinin birlikte modellenmesine örnektir.

Kaynak: https://doi.org/10.1038/s41598-023-31297-1

## Dataset/transfer kaynakları ve rolleri

Bu kaynaklar birbirinin yerine kullanılamaz. Her birinin PoC içindeki olası rolü
ve sağlayamayacağı supervision aşağıda ayrı yazılmıştır.

| Kaynak | Bu projedeki olası rol | Sınır / ground-truth durumu |
|---|---|---|
| AP-10K | Genel animal-pose pretraining veya transfer başlangıç noktası | Özel dorsal nokta ya da `arched` posture etiketi yoktur. Hedef posture dataseti değildir. |
| Livestock Keypoint Detection | Cattle pose transferi ve genel anatomik keypoint başlangıç noktası | Özel `arched` etiketi yoktur; proje dorsal protokolünün ground truth'u değildir. |
| Animal-Pose | Genel cow bbox/keypoint transferi | Lisans ve yeniden kullanım açıklığı kaynak bazında doğrulanmadan kullanılmamalıdır; özel `arched` etiketi sağlamaz. |
| CattleLameness | İnternet kaynaklı 50 video / 42 cow içeren, lameness araştırmasına yönelik aday analiz kaynağı | Title/description temelli etiketler **weak label** niteliğindedir; frame-level arched-back posture ground truth değildir. Repo'da LICENSE dosyası yok; `check-before-use`. |
| Cattle Side/Back Views (Mendeley h2s22wr5py) | 72 sığırın yan+arka görünüşü; explicit side-view kaynağı, geometri/keypoint denemesi için elverişli | CC BY 4.0. Posture etiketi yok — `arched/normal/uncertain/invalid` projede üretilir. Vücut ölçümü (withers height vb.) amaçlı toplanmıştır, lameness/posture çalışması değildir. **Dataset iki ayrı klasöre bölünmüş** (`folder_id` 55b368b1=yan, c54ab6a2=arka, dosya numarası ineği eşler); arka görünüş dorsal kavis göstermediği için `scripts/08_build_external_sources.py` yalnız yan klasörü kullanır ve her ineği ayrı `source_id`/`video_id` yapar — aksi halde 72 inek split'te tek gruba düşer. |
| Cattle Images for Lameness (Mendeley f4j83j77ng) | 277 Wagyu/Angus ham kamera karesi (Folder 1); yardımcı normal/pozitif aday havuzu | CC BY 4.0. İnek kırpılmamış, tam kare — side-view verimi `02_prepare.py` detektörüne bağlı. Posture etiketi yok. |
| CowScreeningDB | Bacak sensörlerinden gelen CSV/IMU zaman serileriyle sensör tabanlı lameness araştırması | RGB video benchmark değildir ve RGB arched-back posture modelini eğitmek için kullanılmamalıdır. |
| Kullanıcı/şirket/farm-owned lateral görüntüler | İnsan tarafından tanımlanan hedef posture datasetinin ana kaynağı | Provenance, izin/lisans, hayvan kimliği ve grup bilgileri kaydedilmeli; `arched/normal` etiketleri kör posture annotation ile üretilmelidir. |
| YouTube / Roboflow | Yalnızca aday görüntü/video havuzu | Her kaynak için kullanım koşulu ve lisans ayrı ayrı doğrulanıp `approved` olmadan indirilemez veya dataset'e alınamaz. Platformda bulunması kullanım izni anlamına gelmez. |

Bağlantılar:

- AP-10K: https://github.com/AlexTheBad/AP-10K
- Livestock Keypoint Detection:
  https://github.com/yww0411/Livestock-keypoint-detection
- Animal-Pose: https://github.com/noahcao/animal-pose-dataset
- CowScreeningDB:
  https://github.com/Shahid-Ismail/CowScreeningDB-A-public-database-for-lameness-detection
- CattleLameness: https://github.com/fahimsohan/CattleLameness
- Cattle Side/Back Views: https://doi.org/10.17632/h2s22wr5py.3
- Cattle Images for Lameness: https://doi.org/10.17632/f4j83j77ng.1

### İndirilmedi / erişilemedi

- **SideCow-VSS** (https://www.mdpi.com/2306-7381/12/11/1104): Bu projenin hedef
  kurulumuna (sağım yolu yan duvarı, 3m yükseklik) en yakın kaynak, ama veri API
  ile indirilemiyor — MDPI sayfası bot erişimine kapalı (Akamai "Access Denied"),
  data availability bölümünde yazarlardan talep gerekiyor. Manuel iletişim
  gerektiren tek kaynak budur.
- **CBVD-5** (Kaggle, ~10.8 GB) ve **Cows2021** (data.bris.ac.uk, ~18.9 GB):
  API ile indirilebilir durumdalar (Kaggle credentials mevcut, doğrudan zip
  linki çalışıyor) ama planda ikincil öncelikli (kamera dayanıklılığı / ikinci
  aşama) oldukları ve ilk 300–500 görüntü hedefi diğer dört kaynakla zaten
  karşılanabildiği için bilinçli olarak ertelendi. Disk alanı hazır olduğunda
  tek komutla indirilebilir.
- **Livestock Keypoint Detection cattle alt kümesi** indirildi (888 görüntü,
  `data/external/Livestock-keypoint-detection/data_process/cattle/{A,B,C}`)
  ama `data/sources.csv`'ye bilerek eklenmedi: 18 noktalı keypoint şeması bu
  projenin 5 noktalı dorsal protokolüyle uyuşmuyor ve görüntüler karışık
  açılardan, önceden kırpılmış geliyor. Rolü yalnız pose-pretraining'dir.

Sonuç olarak genel pose datasetleri yalnız transfer/pretraining rolündedir;
sensör verisi, weak label ve posture ground truth da ayrı kavramlardır. Hedef
`arched/normal` posture etiketi ile projeye özel dorsal annotation, izinli lateral
görüntüler üzerinde proje tarafından üretilir.

## Raporda güvenle kullanılabilecek cümleler

- “Arched-back posture, lameness ile ilişkili görsel göstergelerden biridir.”
- “Model, insan tarafından oluşturulmuş arched/normal posture etiketlerini
  ayırmak üzere değerlendirilmiştir.”
- “Train/validation/test ayrımı video/kaynak grubu bazında yapılmıştır.”
- “Anatomik geometri beş dorsal keypoint ile normalize edilmiştir.”
- “Pretrained segmentation yalnız veri hazırlama/pre-annotation amacıyla
  kullanılmış, otomatik ground truth kabul edilmemiştir.”

## Kullanılmaması gereken cümleler

- “Kambur inek topaldır.”
- “Model topallık/hastalık teşhisi koyar.”
- “Hoffman 0.63/0.64'ten daha iyi sonuç aldık.”
- “YouTube üzerindeki score bağımsız klinik ground truth'tur.”
- “İki uçtan yüzde 20 kırpmak torakolomber bölgeyi garanti eder.”
- “Poursaberi yöntemini birebir replike ettik.”
- “Bu sonuçlar sahada genellenebilirliği kanıtlar.”
