# Cow Arched-Back Posture PoC

Yan görünüşlü inek görüntülerinde `arched-back posture` tespiti için 2–3 günlük,
literatürle uyumlu ve yeniden üretilebilir bir proof-of-concept hattıdır.

Bu proje **topallık veya hastalık teşhisi koymaz**. Çıktısı yalnızca insanın
verdiği görsel duruş etiketine göre `arched-back probability` değeridir.

## Bilimsel sınırlar

- Sırt kamburluğu topallıkla ilişkilidir fakat tek başına duyarlı ve özgül bir
  klinik test değildir.
- Veri bölümü frame bazında değil `source/video/cow` grubu bazında yapılır.
- `normal`, `arched`, `uncertain`, `invalid` etiketleri kullanılır; son iki sınıf
  ana eğitime alınmaz.
- Segmentasyon maskesinin iki ucunu sabit yüzdeyle kırparak çıkarılan geometri
  yalnızca **deneysel yardımcı özellik** kabul edilir.
- Anatomik geometri için önerilen ana yol beş dorsal keypoint'tir:
  `withers → thoracic → thoracolumbar → lumbar → sacrum`.
- Otomasyon insan kararının yerine geçmez; kare çıkarma, cow crop, duplicate
  eleme, kalite metadatası ve etiketleme sırasını hızlandırır.

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

## Çalışma biçimi: notebook-first

Kontrol notebook'lardadır. `scripts/` altındaki CLI'lar **backend**'dir: aynı
fonksiyonları toplu işlem için çağırırlar, notebook'ta gördüğün kod yolunun
birebir aynısını koştururlar (`cowarch/prepare.py:process_frame`).

Kural: hiçbir aşama önce sonucu göstermeden yazmaz. Her notebook üç bayrakla açılır.

```python
PREVIEW_ONLY = True     # sadece bak, hiçbir şey yazma
MAX_SAMPLES = 20        # önizlemede kaç örnek
SAVE_OUTPUTS = False    # yazmak için bu da True olmalı
```

Görselleri kontrol ettikten sonra `PREVIEW_ONLY = False` ve `SAVE_OUTPUTS = True`
yapıp bütün veri seti üzerinde koşturursun.

```bash
source .venv/bin/activate
jupyter lab notebooks/
```

| Notebook | Ekranda ne görürsün | Neye karar verirsin |
|---|---|---|
| `00_dataset_browser` | Kaynak tablosu, lisans, thumbnail grid | Hangi kaynak kullanılacak |
| `01_detection_segmentation_inspector` | Ham kare, bbox, crop, maske overlay, reject sebepleri | Eşikler doğru mu, maske sırtı kapsıyor mu |
| `02_back_geometry_inspector` | Topline, chord, sagitta, keypoint geometrisi | Kamburluk neye göre ölçülüyor |
| `03_labeling` | Crop + sınıf butonları + 5 dorsal nokta | arched / normal / uncertain / invalid |
| `04_training_evaluation` | Split dengesi, PR/ROC, hata örnekleri, occlusion | Model gerçekten sırta mı bakıyor |

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

### Etiketleme ve active learning

`03_labeling` içindeki `PostureLabeler` şu kuralları kodda zorlar:

- Split etiketlemeden önce kilitlenmiş olmalı.
- Test ve validation **kör** etiketlenir: `order="priority"` train dışında hata verir.
- Model sırayı değiştirebilir, etiketi **yalnız insan** yazar.
- Keypoint modunda beş nokta girilmeden `arched`/`normal` kabul edilmez.
- Manifest her karardan sonra atomik yazılır, `reviewed_by` dolar.

Active learning döngüsü:

```
5 arched + 5 normal  →  frozen ResNet18 embedding  →  ilk sınıflandırıcı
                     →  modelin en kararsız olduğu 20 kare  →  sen etiketle  →  tekrar
```

Sekiz etiketin altında cosine nearest-centroid, üstünde logistic regression
kullanılır. `cowarch/active.py:build_pool_mask` test satırlarının havuza girmesi
durumunda `TestSetLeakError` fırlatır — test kareleri active learning'e hiç
girmez, ayrıca ve rastgele sırada etiketlenir.

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
python scripts/03_split.py    --manifest data/manifest.csv --group-column video_id
python scripts/04_label.py    --manifest data/manifest.csv --split test --order random
python scripts/05_train.py    --manifest data/manifest.csv --output-dir outputs/run01 \
                              --models geometry embedding fusion
python scripts/06_report.py   --run-dir outputs/run01 --manifest data/manifest.csv
python scripts/07_predict.py  --model outputs/run01/embedding.joblib \
                              --manifest data/new_manifest.csv --output outputs/new_predictions.csv
```

`04_label.py`, notebook arayüzünün cv2 tabanlı terminal karşılığıdır; aynı dört
sınıfı ve beş keypoint'i toplar.

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

`C` hiperparametresi validation PR-AUC ile, karar eşiği validation F1 ile seçilir.
Test yalnız final değerlendirmede kullanılır.

Sadece deneysel otomatik maske geometrisiyle çalışmak açık onay ister:

```bash
python scripts/05_train.py --manifest data/manifest.csv --output-dir outputs/auto_geometry \
                           --models geometry --allow-auto-geometry
```

## Minimum veri hedefi

- 200–500 geçerli yan görünüş crop
- Mümkünse en az 50–100 `arched` örneği
- En az 8–10 bağımsız video/kaynak grubu
- Test için en az 3–4 bağımsız grup
- Geometri modeli için 150–250 örnekte beş dorsal keypoint

Bu sayılar klinik performans garantisi değil, kısa PoC'nin çalışabilmesi için
pratik hedeflerdir.

## Testler

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
python -m unittest discover -s tests -v
python -m compileall cowarch scripts
```

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
