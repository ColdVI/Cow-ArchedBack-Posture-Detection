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

## Dataset/transfer kaynakları

- AP-10K: 10,015 görüntü ve 54 tür; genel animal pose pretraining/transfer için
  kullanılabilir, özel dense cow spine annotation sağlamaz.
  https://github.com/AlexTheBad/AP-10K
- Animal-Pose: cow dahil beş ana hayvan sınıfında bbox ve keypoint annotation.
  https://github.com/noahcao/animal-pose-dataset
- Livestock Keypoint Detection: cattle/horse/sheep ve 18 anatomik keypoint.
  https://github.com/yww0411/Livestock-keypoint-detection
- CowScreeningDB: lameness araştırması için yayınlanmış cattle video benchmark
  kaynağı; kullanım koşulları indirmeden önce ayrıca doğrulanmalıdır.
  https://github.com/Shahid-Ismail/CowScreeningDB-A-public-database-for-lameness-detection

Bu genel pose datasetleri özel `arched` etiketi veya beş dense dorsal nokta
sunmayabilir. Bu nedenle posture label ve dorsal keypoint alt kümesi proje
tarafından üretilir.

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

