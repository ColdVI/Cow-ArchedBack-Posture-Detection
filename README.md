# Cow Arched-Back Posture Absolute-Score Triage

Sabit yan kameradan geçen ineğin withers–sacrum arasındaki sırt kavisini ölçer,
geçiş boyunca robust biçimde birleştirir ve iki gözlemcinin kalibrasyonundan
türetilmiş mutlak posture bandına çevirir.

Bu proje **topallık veya hastalık teşhisi koymaz**. Arched-back posture topallıkla
ilişkili olabilir; başka durumlar da kambur duruşa yol açabilir ve bazı topal
inekler kamburlaşmaz. Çıktı yalnız inceleme/triyaj skorudur.

## V1 akışı

```text
30 fps ham kayıt (ayrı, değişmeden saklanır)
    → plumb-line undistortion
    → optik merkez bandı
    → cow detection + segmentation
    → withers, sacrum, head keypoint
    → head-down frame filtresi
    → 101 noktalı anchored dorsal profile
    → passage medyanı + IQR
    → gözlemci kalibrasyonundan mutlak posture skoru
    → treatments.csv ile retrospektif korelasyon
```

`cow_id`, `passage_id`, `timestamp_utc` ve ham kareler v1 mutlak skor için de
tutulur. Böylece ertelenen inek-özel baseline/CUSUM aşaması ileride yeni veri
toplamadan eklenebilir. `cowarch/baseline.py` ve `scripts/12_detect_changes.py`
bu gelecek aşama için korunur fakat v1 ana akışında çalıştırılmaz.

## Ölçüm sözleşmesi

- Üretim keypoint sırası: `withers, sacrum, head`.
- Withers ve sacrum yoğun sırt profilinin çapalarıdır.
- Head ölçüme girmez; `head_drop_max_norm` üstündeki kareyi eler.
- Ana frame metriği `anchored_sagitta_signed_norm`'dur.
- `auto_sagitta` sabit `%20` kırpmalı deneysel karşılaştırmadır; ana skor değildir.
- Düşük kaliteli frame/geçiş silinmez. `measurement_reject_reason`,
  `reject_breakdown`, `passage_quality` ve `score_eligible` ile izlenir.
- Hiçbir sagitta→skor sınırı hardcode edilmez. Bantlar iki gözlemcinin verisinden
  `scripts/15_calibrate_scores.py` ile türetilir.

Tarihsel beş-keypoint supervised deneyleri silinmemiştir ancak
`--allow-legacy-keypoints` açık onayı olmadan train, predict veya report yoluna
giremez.

## Kurulum

Ana ortam Python 3.11–3.13 kullanır:

```bash
/opt/homebrew/bin/python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

DeepLabCut T1 deneyi ayrı Python 3.10–3.12 ortamında çalışır; ayrıntı ve mevcut
karar durumu [docs/KEYPOINT_MODEL_DECISION.md](docs/KEYPOINT_MODEL_DECISION.md)
içindedir. Üretim YOLO-pose adapter'ı COCO human-pose checkpoint'ini kabul etmez;
checkpoint tam üç keypoint taşımalı ve sırası `withers,sacrum,head` olmalıdır.

## Kamera ve kaynak şeması

`sources.csv`, `camera_role=measurement|passage_detection` taşır. Yalnız
`measurement` kamerası geometri üretir. Örnek şema `sources.example.csv` içindedir.
Ölçüm kamerası yan görünüş, rectilinear yaklaşık 45–55° yatay FOV, sabit beyaz
ışık/pozlama/WB/odak, mat arka plan ve rijit montaj kullanmalıdır. Tepeden 2D RGB
sagittal kavisi göremez.

Mevcut balık gözü kamera için düz olduğu bilinen en az üç kiriş üzerinde, her
birinde en az dört nokta içeren JSON hazırlanır:

```json
{
  "image_width": 1920,
  "image_height": 1080,
  "lines": [
    [[120, 80], [125, 300], [132, 600], [145, 980]],
    [[900, 60], [905, 300], [910, 650], [918, 1020]],
    [[80, 160], [500, 170], [1100, 185], [1820, 210]]
  ]
}
```

```bash
python scripts/14_calibrate_camera.py \
  --lines data/camera_01_plumb_lines.json \
  --output configs/camera_01_calibration.json
```

Kalibrasyon, düz çizgi RMS hatasını düşürmezse açık hata verir. Bu plumb-line
düzeltmesi metrik kamera kalibrasyonu iddiası taşımaz; yalnız radyal bias'ı
azaltır.

## Veri hazırlama ve geçiş agregasyonu

Ham 30 fps kaynaklar `data/raw/` altında kalır. İnceleme/etiketleme dataseti
`--target-fps 1` ve dedupe ile ayrı üretilir. Passage ölçümü ise aynı ham kaynağı
`--measurement-stream --target-fps 30` ile, dedupe etmeden işler:

```bash
python scripts/02_prepare.py \
  --sources data/sources_selected.csv \
  --output-dir data/prepared_after \
  --manifest data/frames_after.csv \
  --measurement-stream --target-fps 30 \
  --model outputs/detector_v1/best.pt \
  --anchor-model outputs/keypoints_v1/best.pt \
  --camera-calibration configs/camera_01_calibration.json \
  --center-band-fraction 0.50 \
  --head-drop-max-norm 0.30

python scripts/10_aggregate_passages.py \
  --manifest data/frames_after.csv \
  --output data/passages_after.csv \
  --min-quality 0.5 --min-valid-frames 3
```

Kamera hafifletmesi öncesi karşılaştırma aynı passage'lar üzerinde ayrı bir
hazırlama/agregasyon koşumudur: kalibrasyon verilmez ve
`--center-band-fraction 1.0` kullanılır. `scripts/11_variance_study.py` passage
setleri farklıysa, anchored + fixed-trim kolonlarından biri yoksa veya `after`
tablosu hem undistortion hem daraltılmış merkez bandını kanıtlamıyorsa durur:

```bash
python scripts/11_variance_study.py \
  --passages-before data/passages_before.csv \
  --passages-after data/passages_after.csv \
  --arched-cow-ids data/known_arched_cows.csv \
  --camera-id camera_01 \
  --output-dir outputs/variance_study
```

`report.md` dört hücreyi birlikte verir: before/after × anchored/fixed-trim.
Her hücrede eski `Δ/σ_within_cow` okuması ve v3 kapısı
`Δ/σ_between_healthy` bulunur; ayrıca before/after `σ_between_healthy`
karşılaştırılır.

## Skor kalibrasyonu (T9)

Yaklaşık 100–150 passage iki kişi tarafından bağımsız, ordinal olarak
skorlanır. Uzun-form şema:

```csv
passage_id,observer_id,score
passage_0001,observer_a,1
passage_0001,observer_b,1
```

```bash
python scripts/15_calibrate_scores.py \
  --passages data/passages_after.csv \
  --observations data/score_observations.csv \
  --output-dir outputs/score_calibration \
  --report docs/SCORE_CALIBRATION.md
```

Çıktı quadratic weighted kappa, sagitta–consensus Spearman korelasyonu ve
isotonic regression'dan türetilen monoton bant sınırlarını içerir. İki
gözlemcinin aynı passage setini tamamlamaması veya sınırların veriden
tanımlanamaması açık hatadır. `scored_passages.csv`, türetilmiş
`posture_score` kolonunu taşır.

## Retrospektif validasyon (T6)

`treatments.csv` en az `date,cow_id` taşır; `trigger=routine|observed` en değerli
alandır. Eksik trigger aynı gün işlem gören inek sayısından yaklaşık üretilir ve
audit çıktısında işaretlenir. `calvings.csv` için `cow_id,date` zorunludur.

```bash
python scripts/12_validate_retrospective.py \
  --passages outputs/score_calibration/scored_passages.csv \
  --treatments data/treatments.csv \
  --calvings data/calvings.csv \
  --score-column posture_score \
  --output-dir outputs/retrospective
```

Rapor mutlak sagitta ile belirlenen ufuk içindeki `observed` tedavi kaydı
arasındaki Pearson/Spearman korelasyonunu verir. Routine işlemler pozitif olay
sayılmaz; peripartum passage'lar bastırılır. T5 ertelendiği için detection
latency ve inek başına aylık yanlış alarm v1 raporunda yer almaz.

## Raporlama ve PPV

Eski supervised karşılaştırma raporu korunur. Test-set precision'ını doğrudan
sürü performansı gibi sunmaz; beklenen PPV'yi verilen sürü prevalansında zorunlu
olarak yeniden hesaplar:

```text
PPV = prevalence × sensitivity /
      (prevalence × sensitivity + (1 − prevalence) × (1 − specificity))
```

```bash
python scripts/06_report.py \
  --run-dir outputs/run01 --manifest data/manifest.csv \
  --herd-prevalence 0.02
```

IR/gece ve parçalanmış maske slice'ları raporda ayrıca kalır.

## Etiketleme ve detektör

- [docs/ANNOTATION_GUIDE.md](docs/ANNOTATION_GUIDE.md): kör Pass A ve üç-nokta
  Pass B protokolü; ayrıca deneysel 19-nokta tam-postür turu.
- `label_studio/configs/pass_b_geometry.xml`: withers/sacrum/head arayüzü.
- `scripts/16_label_pose.py`: görünürlük bilgili, mouse-first tam-postür
  etiketleyicisi. Bu katman üretim Pass B şemasının yerini almaz.
- `scripts/19_ingest_pose_media.py`: yerel MP4/görüntü klasörünü tarar, her
  karedeki inekleri ayrı kırpar ve tam-postür manifestini üretir.
- `scripts/13_train_detector.py`: IR/gece ve korkuluk arkası örnekleri zorunlu
  detektör fine-tuning preflight'ı; varsayılan `YOLO11m-seg` ve held-out
  withers–sacrum topline MAE değerlendirmesi.
- `cowarch/anchors.py`: framework-neutral `predict_anchors()` sözleşmesi ve
  üç-keypoint YOLO adapter'ı.

```bash
python scripts/13_train_detector.py \
  --data data/detector/data.yaml \
  --dataset-metadata data/detector/metadata.csv \
  --topline-eval data/detector/topline_eval.csv \
  --output-dir outputs/detector_v1
```

Yerel saha medyasından hızlı bir tam-postür etiketleme turu hazırlamak için:

```bash
python scripts/19_ingest_pose_media.py \
  --input-dir "/Users/anil/Downloads/inek data ve metadoloji" \
  --output-dir data/pose_round_01 \
  --target-fps 2

python scripts/16_label_pose.py \
  --manifest data/pose_round_01/manifest.csv \
  --reviewer anil
```

Etiketleyicide sol tık görünür, sağ tık korkuluk/başka bacak arkasında olup
konumu güvenle çıkarılabilen nokta, `0` ise tahmin edilmemesi gereken kayıp
noktadır. `Enter` kaydeder; `Z` geri alır. Kaynaklar yerinde bırakılır, yalnız
crop/mask/manifest `--output-dir` altında oluşur. Mevcut geniş açı, korkuluk ve
tepeden görüntüler `passage_detection` verisidir; measurement-kamera geometrisi
olarak işaretlenmez. Görüntü dosyalarında düşük-kroma IR tahmini `is_ir` ve
`ir_inference` alanlarına yazılır; video IR durumu otomatik tahmin edilmez.

`topline_eval.csv`, `image,ground_truth_mask,withers_x,sacrum_x,is_ir,behind_rails`
kolonlarını taşır. Rapor ana seçim metriği olarak piksel topline MAE'yi ve gövde
uzunluğuna oranını; ayrıca tüm/IR/korkuluk arkası detection rate dilimlerini
yazar. Hedef ortalama oran `<0.005`'tir.

Üç-nokta keypoint checkpoint'i ayrıca kendi verisinde değerlendirilir:

```bash
python scripts/17_evaluate_keypoints.py \
  --predictions outputs/keypoints_v1/pckh_input.csv \
  --output-dir outputs/keypoints_v1/evaluation
```

Uzun-form giriş her `sample_id` için üç keypoint ve
`pred_x,pred_y,true_x,true_y,head_scale_px` taşır. Withers ve sacrum
`PCKh@0.2 >= 0.85` olmadıkça anchor kapısı geçmez.

## Testler

```bash
pip install -r requirements-test.txt
MPLCONFIGDIR=/tmp/cow_arch_matplotlib python -m unittest discover -s tests -v
python -m compileall cowarch scripts
python scripts/00_check_notebooks.py --compile-only
```

Sentetik eski supervised tesisat kontrolü bilinçli olarak
`--allow-legacy-keypoints` kullanır:

```bash
python scripts/00_smoke_test.py
```

Sentetik metrikler saha kanıtı değildir. T0 donanım, T1 görsel model kararı, T4
saha varyansı ve T9 gerçek gözlemci katsayıları veri/donanım gelmeden tamamlanmış
sayılmaz.
