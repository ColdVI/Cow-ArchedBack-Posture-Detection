# 2–3 Günlük Uygulama Planı

## Sabit proje tanımı

**Araştırma sorusu:** Yan görünüşlü bir inek crop'ında insan tarafından
tanımlanmış `arched-back posture` otomatik olarak ayırt edilebilir mi?

**Ana çıktı:** `P(arched)` ve seçilen eşikte `arched / normal` kararı.

**Kapsam dışı:** Klinik topallık teşhisi, hastalık teşhisi, locomotion score
tahmini ve saha genellenebilirliği iddiası.

## Yarım gün granülaritesinde plan

| Zaman | İş | Kabul kapısı | Çıktı |
|---|---|---|---|
| 0–1 saat | Ortam ve kaynak manifesti | Örnek komutlar çalışıyor; kaynakların lisans/izin alanı dolu | `sources.csv` |
| 1. gün sabah | İndirme, 1 fps frame çıkarma, YOLO cow crop, pHash dedup | En az 200 geçerli crop; reject nedenleri kayıtlı | `manifest.csv`, crop/mask |
| 1. gün öğleden sonra | Group split; test/val bağımsız etiketleme; train etiketleme | Grup sızıntısı sıfır; `uncertain/invalid` ayrı | Etiketli manifest |
| 2. gün sabah | 150–250 örneğe 5 dorsal keypoint; Model A/B/C | En az iki model eğitilebiliyor; threshold yalnız val'de seçiliyor | model bundle'ları |
| 2. gün öğleden sonra | Test, hata analizi, overlay ve metrikler | Test'e model seçimi için bakılmamış; rapor tüm limitasyonları yazıyor | `REPORT.md`, figürler |
| 3. gün / tampon | Veri eksiği, bozuk kaynak, demo ve sunum | Yeni bir video grubunda smoke demo | teslim paketi |

## Faz detayları

### F0 — Ortam ve veri sözleşmesi

1. Python 3.11/3.12 virtual environment oluştur.
2. `requirements.txt` kur.
3. Her kaynak için `source_id`, `url/local_path`, `license`, `kind`,
   `source_score` alanlarını doldur.
4. `source_score` yalnız auxiliary metadata'dır; görsel posture ground truth
   olarak kullanılmaz.

**Go:** En az 8 bağımsız video/kaynak grubu bulunmuş olmalı.

### F1 — Otomatik veri hazırlama

1. Videoları varsayılan 1 fps örnekle.
2. COCO-pretrained segmentation modelindeki sınıf adını `cow` olarak dinamik
   çöz; sabit sınıf indeksine güvenme.
3. Tek inekli kareleri crop et; çoklu inek, kadraj kenarı, küçük bbox ve düşük
   güven durumlarını reject nedeni olarak kaydet.
4. pHash/dHash benzerliğiyle ardışık duplicate kareleri ele.
5. Aspect ratio, blur, bbox/mask alanı ve `view_hint` üret; aspect ratio'yu
   varsayılan olarak kesin side-view filtresi sayma.
6. Otomatik topline/sagitta çıkarılıyorsa özelliği `auto_*` adıyla işaretle.

**Go:** En az 200 geçerli crop ve en az 8 grup.

### F2 — Split ve etiketleme

1. Split'i etiket kararından önce grup bazında sabitle.
2. Test ve validation örneklerini rastgele sırada, model tahmini görmeden
   etiketle.
3. Train örneklerinde `auto_sagitta` yalnız kuyruk önceliği olabilir; otomatik
   etikete çevrilmez.
4. Etiket kuralı:
   - `arched`: cidago–sacrum hattına göre torakolomber bölgede belirgin dorsal
     kavis.
   - `normal`: düz/normale yakın dorsal hat.
   - `uncertain`: yeterli görüntü var fakat karar güvenilir değil.
   - `invalid`: yan görünüş değil, ciddi oklüzyon, crop eksik, dönme/yatma vb.
5. Beş dorsal keypoint'i mümkünse iki geçişte etiketle: önce posture class,
   sonra keypoint.

**Go:** Her iki sınıf train ve test'te mevcut; grup sızıntısı sıfır.

### F3 — Modelleme

1. **Model A:** keypoint geometri → StandardScaler → Logistic Regression.
2. **Model B:** frozen ResNet18 → 512-D embedding → StandardScaler → Logistic
   Regression.
3. **Model C:** geometri + embedding → StandardScaler → Logistic Regression.
4. `C` hiperparametresini validation PR-AUC ile seç.
5. Karar threshold'unu validation F1 ile seç.
6. Test'i yalnız final değerlendirmede kullan.

**Go:** En az bir model test tahmini üretmiş olmalı. Başarı eşiği önceden
uydurulmaz; sonuç güven aralığı ve hata örnekleriyle raporlanır.

### F4 — Rapor

Raporlanacaklar:

- PR-AUC ve ROC-AUC
- arched recall/sensitivity
- specificity
- precision, F1, balanced accuracy
- confusion matrix
- geometri feature dağılımları
- yanlış pozitif/yanlış negatif örnekleri
- `head_down`, `oblique`, `occluded` gibi slice'larda hata
- veri kaynağı, etiketçi ve kamera limitasyonları

Kullanılacak cümle:

> Model, insan tarafından oluşturulan yan-görünüş arched-back posture
> etiketlerini ayırmak üzere değerlendirilmiştir; klinik topallık veya hastalık
> teşhisi üretmez.

## Sonraki aşama

PoC sinyal gösterirse ikinci faz:

1. İkinci etiketçi/veteriner ile agreement ölçümü.
2. Cow ID veya video bazlı daha büyük dış test.
3. Baş ve hoof keypoint'leri.
4. `back posture + head bob + tracking distance + stride` temporal modeli.
5. Gerçek klinik locomotion score ile bağımsız validasyon.

