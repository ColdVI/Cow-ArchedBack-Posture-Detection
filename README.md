# Cow Arched-Back Posture Longitudinal Triage

Sabit yan kameradan geçen her ineği kendi geçmişiyle karşılaştıran,
`arched-back posture` değişimini erken inceleme kuyruğuna taşıyan ürün hattıdır.
Eski tek-kare sınıflandırıcı geriye dönük deneyler için korunur; v1'in ana sinyali
segmentasyon maskesinden deterministik geometri ve geçiş-seviyesi agregasyondur.

Bu proje **topallık veya hastalık teşhisi koymaz**. Çıktısı, bir ineğin kendi
normalinden sapan duruş ölçümü için inceleme önerisidir. Kod hattı uygulanmıştır;
saha performansı ancak P2 varyans kapısı ve retrospektif/prospektif validasyonla
gösterilebilir.

## V1 akışı

```text
30 fps ham kayıt (değiştirilmez)
    → 1 fps türetilmiş frame/mask manifesti
    → passage bazında robust geometri özeti
    → varyans GO/NO-GO kapısı
    → rolling medyan + MAD baseline, EWMA/CUSUM
    → günlük kapasiteden türetilen inceleme eşiği
    → tedavi kayıtlarıyla retrospektif validasyon
```

Ham videolar `data/raw/` gibi türetilmiş `data/prepared/` dizininden ayrı bir
yerde saklanmalıdır. `02_prepare.py` kaynak dosyayı silmez veya dedupe etmez;
1 fps örnekleme ve pHash dedupe yalnız hazırlanan dataset'e uygulanır.

## Bilimsel sınırlar

- Sırt kamburluğu topallıkla ilişkilidir fakat tek başına duyarlı ve özgül bir
  klinik test değildir.
- Veri bölümü frame bazında değil varsayılan olarak `cow_id` bazında yapılır.
- `cow_id` eksik veya boşsa split ve eğitim sessizce `video_id`'ye düşmez.
- `normal`, `arched`, `uncertain`, `invalid` etiketleri kullanılır; son iki sınıf
  ana eğitime alınmaz.
- Segmentasyon maskesinin iki ucunu sabit yüzdeyle kırparak çıkarılan geometri
  yalnızca **deneysel yardımcı özellik** kabul edilir.
- Eski supervised karşılaştırma için anatomik geometri yolu beş dorsal keypoint'tir:
  `withers → thoracic → thoracolumbar → lumbar → sacrum`.
- V1 longitudinal ölçümde sabit `%20` kırpmalı `auto_sagitta` tercih edilen
  tekrarlanabilir sinyaldir; anatomik doğruluk iddiası taşımaz.
- Sistem değişimi tespit eder, durumu değil. Baseline'ı zaten yüksek kronik bir
  inek hiç tetiklenmeyebilir.

## Kurulum

Python 3.11–3.13 gerekir. `pyproject.toml` 3.14'ü dışarıda bırakır çünkü
Torch ve Ultralytics'in 3.14 wheel'leri henüz yok. Homebrew'lu macOS'ta
sistem `python3` 3.14 olabilir, o yüzden venv'i açıkça 3.13 ile kur:

```bash
/opt/homebrew/bin/python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Ultralytics ve Torch ağırlıkları ilk kullanımda indirilebilir. İnternetsiz
çalışacaksan checkpoint'leri önceden indirip komutlarda yerel yollarını ver.

## Tek görüntüyle başla

İlk önerilen giriş noktası `notebooks/00_single_image_walkthrough.ipynb`'dir.
Bir görüntüyü şuraya koy:

```text
data/inbox/one_cow.jpg
```

Notebook varsayılan olarak `yolo11n-seg.pt` ile yalnız bellekte önizleme yapar;
`sources.csv`, manifest, split veya hazırlanmış dataset istemez. Birden fazla cow
bulursa ekrandaki detection indekslerinden birini sen seçersin. Elinde hazır crop
varsa `MODEL_NAME = "none"` kullanabilirsin. Maske yoksa raw/bbox/crop panelleri
yine gösterilir, yalnız contour geometrisi atlanır.

Preview modu eksik checkpoint'i otomatik indirmez: `yolo11n-seg.pt` dosyasını
önceden proje köküne koy, `MODEL_NAME` için başka bir yerel path ver veya hazır
crop ile `"none"` kullan. Böylece varsayılan koşum model ağırlığı da yazmaz.

Hiçbir çıktı varsayılan koşumda yazılmaz. Yazma ancak iki bayrak birlikte
değiştirilirse mümkündür:

```python
PREVIEW_ONLY = False
SAVE_OUTPUTS = True
```

Notebook ham görüntüden selected crop ve maskeye, kırpılmamış topline'dan açıkça
`experimental silhouette baseline` diye işaretlenen `%20 trim` ölçümüne, son
olarak tıklanan withers/sacrum anchor'ları arasındaki dense contour ve signed
sagitta'ya kadar sekiz inspection paneli gösterir. Sagitta bir feature'dır;
hiçbir aşamada ground-truth posture etiketi değildir.

## Çalışma biçimi: notebook-first

Kontrol notebook'lardadır. `scripts/` altındaki CLI'lar **backend**'dir: aynı
fonksiyonları toplu işlem için çağırırlar, notebook'ta gördüğün kod yolunun
birebir aynısını koştururlar (`cowarch/prepare.py:process_frame`).

Çok örnekli inspection ve batch notebook'ları sonucu göstermeden yazmaz; bu
notebook'lar üç bayrakla açılır (`00_single_image_walkthrough` tek örnek olduğu
için `MAX_SAMPLES` kullanmaz):

```python
PREVIEW_ONLY = True     # sadece bak, hiçbir şey yazma
MAX_SAMPLES = 20        # önizlemede kaç örnek
SAVE_OUTPUTS = False    # yazmak için bu da True olmalı
```

Görselleri kontrol ettikten sonra `PREVIEW_ONLY = False` ve `SAVE_OUTPUTS = True`
yapıp bütün veri seti üzerinde koşturursun. `03_labeling` bunun bilinçli
istisnasıdır: bir insan posture veya geometry kararını kaydettiği anda manifesti
atomik olarak yazar; önizleme/batch commit notebook'u değildir.

```bash
source .venv/bin/activate
jupyter lab notebooks/
```

| Notebook | Ekranda ne görürsün | Neye karar verirsin |
|---|---|---|
| `00_single_image_walkthrough` | Tek görüntüde raw, indeksli bbox, crop, mask, topline ve anchored geometri | Detector/mask/geometri bu görüntüde anlamlı mı |
| `00_dataset_browser` | Kaynak tablosu, lisans, thumbnail grid | Hangi kaynak kullanılacak |
| `01_detection_segmentation_inspector` | Ham kare, bbox, crop, maske overlay, reject sebepleri | Eşikler doğru mu, maske sırtı kapsıyor mu |
| `02_back_geometry_inspector` | Topline, chord, sagitta, keypoint geometrisi | Kamburluk neye göre ölçülüyor |
| `03_labeling` | Pass A'da yalnız crop+sınıf; Pass B'de ayrı dorsal annotation | Önce posture, sonra bağımsız geometri |
| `04_training_evaluation` | Frame ve group/passage metrikleri, cluster bootstrap, hata örnekleri | Model bağımsız gruplarda ne yapıyor |

### Altı panel

`02_back_geometry_inspector` tek örnek için şunu gösterir:

```
1. ham görüntü      2. YOLO bbox         3. cow crop
4. maske overlay    5. topline + chord   6. withers–sacrum + keypoint + sagitta
```

Sagitta, withers–sacrum doğrusuna en büyük dik uzaklığın doğru uzunluğuna
bölümüdür — farklı mesafe ve vücut ölçülerinde karşılaştırılabilir olsun diye.

**Kritik ayrım:** sagitta kamburluğu *açıklayan bir feature*'dır, otomatik etiket
değildir. `sagitta > 0.07 → arched` diyerek veri seti kurup sonra sagitta ile model
eğitmek, modelin kendi eşiğine fit olması demektir. Etiketler `03_labeling`'de
insandan gelir.

### İki geçişli etiketleme ve active learning

`03_labeling` iki ayrı annotation geçişi kullanır:

- **Pass A — posture:** yalnız görüntü ile `arched / normal / uncertain / invalid`
  kararı. Keypoint, sagitta, probability ve `source_score` gösterilmez.
- **Pass B — geometry:** yalnız Pass A'sı tamamlanmış `arched/normal` satırlarda
  beş dorsal keypoint veya withers/sacrum anchor'ları. Posture label yeniden
  yazılmaz; geometri skoru ancak noktalar kaydedildikten sonra görülebilir.

Test ve validation posture sırası kör ve random kalır. Active/priority sıralama
yalnız train'de kullanılabilir. Manifest `posture_reviewed_by`,
`geometry_reviewed_by` ve `annotation_pass` alanlarını taşır. Eski
`reviewed_by` okunmaya devam eder ve kayıpsız biçimde yeni reviewer alanlarına
yorumlanır; eski kolon silinmez.

Active learning döngüsü:

```
5 arched + 5 normal  →  frozen ResNet18 embedding  →  ilk sınıflandırıcı
                     →  modelin en kararsız olduğu 20 kare  →  sen etiketle  →  tekrar
```

Her sınıfta sekiz train etiketi oluşana kadar cosine nearest-centroid, bu eşiğin
ardından logistic regression kullanılır. Seed model yalnız
accepted+labeled+`split=="train"` satırlara fit
edilir; queue yalnız accepted+unlabeled train satırlarından seçilir. Probability
tüm satırlar için hesaplanabilir fakat validation/test fitting veya seçimde
kullanılmaz. `labelable_splits` varsayılanı yalnız `("train",)`'dir; validation
veya test'i active-learning pool'una açıkça katma girişimi hata üretir.

Roboflow veya Label Studio bu ölçekte gerekmez. İkinci bir etiketçi/veteriner
sürece katılırsa, dataset 1.000+ görüntüye çıkarsa veya annotation geçmişi/reviewer
sistemi gerekirse Label Studio anlamlı hale gelir.

## CLI backend

Notebook'lar bu scriptleri çağırır; ayrıca doğrudan da çalıştırılabilirler.

```bash
python scripts/01_collect.py  --sources data/sources.csv --output-dir data/raw \
                              --output-sources data/sources_resolved.csv
python scripts/02_prepare.py  --sources data/sources_selected.csv \
                              --output-dir data/prepared --manifest data/manifest.csv \
                              --target-fps 1 --model yolo11n-seg.pt
python scripts/03_split.py    --manifest data/manifest.csv
python scripts/04_label.py    --manifest data/manifest.csv --split test --order random \
                              --reviewer posture-reviewer-01
python scripts/05_train.py    --manifest data/manifest.csv --output-dir outputs/run01 \
                              --models geometry embedding fusion
python scripts/06_report.py   --run-dir outputs/run01 --manifest data/manifest.csv
python scripts/07_predict.py  --model outputs/run01/embedding.joblib \
                              --manifest data/new_manifest.csv --output outputs/new_predictions.csv
```

Longitudinal backend:

```bash
python scripts/10_aggregate_passages.py \
  --manifest data/manifest.csv --output data/passages.csv \
  --min-quality 0.5 --min-valid-frames 3

python scripts/11_variance_study.py \
  --passages data/passages.csv --arched-cow-ids data/known_arched_cows.csv \
  --camera-id camera_01 --output-dir outputs/variance_study

# Yalnız variance study kararı GO ise çalışır. sigma floor ve günlük inceleme
# kapasitesi zorunlu girdidir; sabit posture/z eşiği kullanılmaz.
python scripts/12_detect_changes.py \
  --passages data/passages.csv \
  --variance-summary outputs/variance_study/summary.json \
  --calvings data/calvings.csv --sigma-floor 0.002 --daily-capacity 8 \
  --output-dir outputs/changes

python scripts/12_validate_retrospective.py \
  --daily-signals outputs/changes/daily_signals.csv \
  --treatments data/treatments.csv --calvings data/calvings.csv \
  --output-dir outputs/retrospective
```

`treatments.csv` en az `date,cow_id` taşır; `lesion_type`, `foot` ve özellikle
`trigger=routine|observed` önerilir. Eksik `trigger` aynı gün müdahale edilen
inek sayısından yaklaşık üretilir ve raporda inferred olarak işaretlenir.
`calvings.csv` şeması `cow_id,date`'tir. Peripartum penceresinde alarm bastırılır
ve doğum sonrası kişisel baseline yeniden kurulur.

Detektör fine-tuning v1'deki tek supervised iştir. Metadata dosyası
`image,is_ir,behind_rails` kolonlarını taşır; preflight en az 300 etiketli kareyi,
IR/gece ve korkuluk arkası örneklerini zorunlu tutar:

```bash
python scripts/13_train_detector.py \
  --data data/detector/data.yaml \
  --dataset-metadata data/detector/metadata.csv \
  --output-dir outputs/detector_v1
```

`04_label.py` yalnız image-only **Pass A posture** için cv2 tabanlı terminal
alternatifidir. Combined `--keypoints` ve geometriye göre `auto-sagitta` sırası
açık hata ile reddedilir. **Pass B geometry** için notebook 03 içindeki
`DorsalGeometryLabeler` kullanılır.

### Veri kaynağı ve lisans kapısı

`sources.example.csv` yeni provenance şemasını gösterir: kaynak/lisans alanlarına
ek olarak farm, cow, video, passage ve camera kimlikleri tutulur. Geçerli
`license_status` değerleri `approved`, `restricted`, `unresolved`'dır. Yalnız
`approved` satırlar `scripts/01_collect.py` tarafından resolve veya download
edilir; diğer durumlar herhangi bir indirme başlamadan açık hata verir.

Eski `license` kolonu okunabilir, fakat boş, `check-before-use`, `unknown` ve
`unresolved` değerleri asla onay sayılmaz. Collector yalnız elle verilmiş, kaynak
bazında onaylanmış URL'leri işler; otomatik YouTube araması veya query-based
indirme yapmaz. `data/` ve `outputs/` gitignore altındadır; örnek inek görüntüsü
repoya eklenmez.

**Az grupta dejenere split riski:** Split etiketlemeden önce yapıldığı için
gruplara göre bölme, bir split'e tek sınıf düşürebilir; `05_train.py` bu durumda
sessizce devam etmez, hata verir. Doğru çözüm seed'i sonuç beğenilene kadar
yeniden çevirmek **değildir** — bu test setine dolaylı bakmaktır. Doğru çözüm
kaynak grubu sayısını artırmak veya her grubun birden çok inek/duruş içermesini
sağlamaktır. Bu senaryo `00_smoke_test.py` ile birebir üretilebilir.

## Modeller

- **A — geometry:** beş dorsal keypoint'ten açıklanabilir geometri + Logistic Regression
- **B — embedding:** dondurulmuş ImageNet ResNet18 (512-D) + Logistic Regression
- **C — fusion:** ikisinin birleşimi

Keypoint geometrisi upward arch ile downward sag'i signed özelliklerle ayırır;
işaret yatay flip'te değişmez ve normalize değerler resize'a invarianttır.
Withers/sacrum anchor'ları varsa `anchored_topline_features` yalnız bu anatomik
aralıktaki dense mask contour'unu ölçer. Sabit `%20 trim` `auto_*` özellikleri
geriye uyumluluk için korunur fakat deneysel baseline'dır.

`C` hiperparametresi validation PR-AUC ile, karar eşiği validation F1 ile seçilir.
Test yalnız final değerlendirmede kullanılır.

Sadece deneysel otomatik maske geometrisiyle çalışmak açık onay ister:

```bash
python scripts/05_train.py --manifest data/manifest.csv --output-dir outputs/auto_geometry \
                           --models geometry --allow-auto-geometry
```

## Minimum veri hedefi

Longitudinal GO/NO-GO çalışması için hedef yaklaşık 50 inek, inek başına 2–3
hafta ve günde birden çok geçiştir. Bilinen kambur ineklerin yalnız `cow_id`
listesi gerekir; yeni posture etiketi veya model eğitimi gerekmez. Varyans
çalışması kamera ve `pipeline_version` bazında ayrı koşulur.

Eski tek-kare supervised karşılaştırma için önceki pratik hedefler:

- 200–500 geçerli yan görünüş crop
- Mümkünse en az 50–100 `arched` örneği
- En az 8–10 bağımsız video/kaynak grubu
- Test için en az 3–4 bağımsız grup
- Geometri modeli için 150–250 örnekte beş dorsal keypoint

Bu sayılar klinik performans garantisi değil, kısa PoC'nin çalışabilmesi için
pratik hedeflerdir.

## Testler

### Grup seviyesinde değerlendirme

Frame'ler bağımsız örnekler değildir. Rapor frame metriklerini korurken ayrıca
seçilebilir `video_id`, `cow_id` veya `passage_id` üzerinden mean probability ile
group-level metrik üretir. Aynı grupta çelişkili ground truth varsa sessizce
çoğunluk oyu kullanmaz, hata verir. Confidence interval frame değil grup resample
eden cluster bootstrap ile hesaplanır:

```bash
python scripts/06_report.py --run-dir outputs/run01 --manifest data/manifest.csv \
                            --group-column passage_id
```

### Sentetik uçtan uca smoke test

Gerçek veri, YOLO ağırlığı veya Torch olmadan `prepare → split → train → report`
zincirinin tamamını sentetik siluetlerle çalıştırır. Etiketlemeye oturmadan önce
hattın ayakta olduğunu doğrulamak içindir:

```bash
python scripts/00_smoke_test.py
```

Ürettiği `outputs/smoke/` metrikleri **kanıt değildir**; problem sentetik olarak
ayrılabilir olduğu için PR-AUC 1.000 çıkar. Test yalnızca tesisatı doğrular.

### Notebook koşum kontrolü

Notebook'lar kendi kendilerini rapor edemez: üç gün önce yazılmış bir hücre,
bir fonksiyon adı değiştiği için sessizce kırılabilir. Bu script hepsini derler
ve etkileşim gerektirmeyen ikisini smoke verisi üzerinde gerçekten çalıştırır.

```bash
python scripts/00_smoke_test.py        # önce sentetik veri seti
python scripts/00_check_notebooks.py   # sonra notebook'lar
python scripts/00_check_notebooks.py --compile-only   # hızlı sürüm
```

`03_labeling` tasarımı gereği insan tıklaması beklediği için yalnız derlenir;
`00` ve `01` gerçek kaynak gerektirdiğinden aynı şekilde.

### Birim testleri

Geometri, grup sızıntısı, kare hazırlama ve active-learning guard'ları ağır CV
bağımlılıkları olmadan çalışır:

```bash
pip install -r requirements-test.txt
python -m unittest discover -s tests -v
python -m compileall cowarch scripts
python scripts/00_check_notebooks.py --compile-only
```

`.github/workflows/tests.yml` bu hafif koşumları Python 3.12 üzerinde çalıştırır;
Torch, Ultralytics, model ağırlığı, internet dataseti veya GPU gerekmez.

## Protokol dokümanları

- `docs/ANNOTATION_GUIDE.md`: posture ve geometry annotation kuralları
- `docs/DATASET_CARD_TEMPLATE.md`: provenance, lisans, split, bağımsızlık ve bias şablonu
- `docs/FUTURE_SYSTEM.md`: yalnız future phase; mevcut PoC'de doğrulanmamış temporal/identity/alert taslağı
- `LITERATURE_NOTES.md`: literatür iddia sınırları ve dataset rollerinin ayrımı

## Referanslar

- Hoffman et al. (2014), back arch ve lameness ilişkisi:
  https://pubmed.ncbi.nlm.nih.gov/24508427/
- Poursaberi et al. (2010), 2D back-posture shape analysis:
  https://doi.org/10.1016/j.compag.2010.07.004
- Viazzi et al. (2013), bireysel back-posture sınıflandırması:
  https://doi.org/10.3168/jds.2012-5806
- Russello et al. (2024), pose ve çoklu locomotion trait:
  https://arxiv.org/abs/2401.05202
- Ultralytics instance segmentation dokümantasyonu:
  https://docs.ultralytics.com/tasks/segment/
- TorchVision ResNet18 ağırlıkları:
  https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
