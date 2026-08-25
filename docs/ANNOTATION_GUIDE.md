# Arched-back Posture Annotation Guide

Bu kılavuz, tek kare lateral inek görüntülerinde görsel posture etiketi ve dorsal
geometri annotation'ı üretmek içindir. Etiket bir lameness veya hastalık tanısı
değildir. Kampanya başlamadan önce örnek bir kalibrasyon seti üzerinde birlikte
çalışılmalı; protokol kampanya ortasında değiştirilmemelidir.

## Pass A — posture labeling

Pass A'da yalnız cow crop gösterilir. Keypoint, anchor, sagitta, kontur, model
probability, source score, dosya/kaynak kimliği veya önceki geometri görünmez.
Karar aşağıdaki dört etiketten tam olarak biridir:

| Etiket | Kullanım ölçütü |
|---|---|
| `arched` | Withers–sacrum arasındaki torakolomber dorsal hatta açık, yukarı yönlü konvekslik görülür. Görünüm bu kararı verecek kadar lateraldir. |
| `normal` | Aynı dorsal bölge düz veya düze yakındır ve güvenilir biçimde değerlendirilebilir. `arched` bulgusu olmaması tek başına yeterli değildir; görüntü kullanılabilir olmalıdır. |
| `uncertain` | Görüntü genel olarak kullanılabilir, fakat `arched` ile `normal` arasında güvenilir karar verilemez. Sınır vaka, orta dereceli açı, geçici hareket veya head-down etkisi buna neden olabilir. |
| `invalid` | Hedef dorsal bölge değerlendirilemez: ciddi oblique/rear/front görünüm, ağır occlusion, kesilmiş gövde, dönüş anı, yatma veya bozuk görüntü/crop. |

`Uncertain` ve `invalid` eğitimde `normal` sınıfına çevrilmez. İkisi eğitimden
çıkarılır, fakat sayıları ve nedenleri veri kalite sonucu olarak raporlanır.

### Standing ve walking ayrımı

- Her kare için mümkünse `standing`, `walking` veya `unknown` locomotion state
  ayrı metadata olarak kaydedilir; bu alan posture etiketinin yerine geçmez.
- Ayakta düz, yürürken arched görünen aynı cow için kareler kendi gözlenen
  durumlarına göre etiketlenir. Bir kareden diğer durum hakkında çıkarım yapılmaz.
- Standing ve walking kareleri raporda ayrı dağılımlarla gösterilir. Aynı video,
  passage veya cow içindeki ardışık kareler bağımsız hayvan örnekleri sayılmaz.
- Tek karede hareket fazı veya görüntü bulanıklığı kararı güvenilmez kılıyorsa
  `uncertain`; dorsal bölge okunamıyorsa `invalid` kullanılır.

### Lateral ve oblique görünüm

**Lateral görünüm:** Cow'un flank bölgesi görünür, withers–sacrum hattında ciddi
foreshortening yoktur ve torakolomber dorsal kontur baştan sona değerlendirilebilir.
Hayvanın sağa veya sola bakması etiketi değiştirmez.

**Oblique görünüm:** Gövde ekseni görüntü düzleminden belirgin biçimde uzaklaşır;
ön veya arka yüzey normalden fazla görünür ve dorsal uzunluk perspektifle kısalır.
Hafif oblique bir karede dorsal şekil hâlâ güvenle okunabiliyorsa karar verilebilir;
açı sınıf kararını etkiliyorsa `uncertain`, bölgeyi geçersiz kılıyorsa `invalid`
seçilir. Aspect ratio yalnız inceleme ipucudur, tek başına view ground truth değildir.

### Head-down confounder

Grazing, yem yeme veya su içme sırasında başın aşağıda olması dorsal hattı mekanik
olarak değiştirebilir. Head-down görünüm **arched kanıtı değildir** ve ağrı ya da
lameness olarak yorumlanmaz.

- Torakolomber hat yine açıkça değerlendirilebiliyorsa yalnız gözlenen sırt
  posture'ına göre etiket verilir ve head-down confounder metadata/not olarak
  kaydedilir.
- Head position `arched` ile `normal` kararını güvenilmez kılıyorsa `uncertain`
  kullanılır.
- Baş/boyun veya crop hedef dorsal bölgeyi kapatıyor ya da kesiyorsa `invalid`
  kullanılır.

### Occlusion ve crop kuralları

`Arched` veya `normal` yalnız şu koşulların tamamında verilir:

- withers–sacrum arasındaki dorsal bölge görünürdür;
- gövde tek cow'a aittir ve başka bir cow ile karışmaz;
- crop, torakolomber şekli değerlendirecek bağlamı korur;
- motion blur, düşük çözünürlük, gölge veya segmentasyon hatası kararı bozmaz.

Kuyruk, distal bacak veya başın küçük bir kısmının crop dışında olması, hedef
dorsal bölge ve yönelim açık ise otomatik olarak `invalid` değildir. Withers,
sacrum ya da aradaki hattın occlusion/crop nedeniyle güvenle yerleştirilememesi
`invalid` nedenidir. Kısmi ve hafif occlusion yalnız sınıf güvenini düşürüyorsa
`uncertain` kullanılabilir. Crop içinde birden fazla cow varsa ve hedef bireyin
dorsal hattı karışıyorsa örnek `invalid` olur.

Validation ve test Pass A örnekleri kör ve random sırada etiketlenir. Active
ordering yalnız train split'inde kullanılabilir. Her karar
`posture_reviewed_by` alanında reviewer kimliğiyle izlenir.

## Pass B — geometry annotation

Pass B yalnız Pass A tamamlanmış, kabul edilmiş ve etiketi `arched` veya `normal`
olan örnekleri alır. Mümkün olduğunda mevcut posture etiketi annotator'a
gösterilmez; arayüzde posture butonu bulunmaz ve `label` hiçbir zaman yeniden
yazılmaz. Geometry reviewer ayrı `geometry_reviewed_by` alanına kaydedilir.

Üretim protokolünde tam üç nokta tıklanır: `withers → sacrum → head`.

- `withers` ve `sacrum`, maske üzerindeki anchor-bounded dense topline ölçümünün
  çapalarıdır.
- `head` sagitta hesabına girmez; baş-aşağı karelerini ölçümden çıkarmak için
  kullanılır.
- Tarihsel beş dorsal keypoint yalnız açık `allow_legacy_keypoints` onayıyla eski
  supervised deneyleri yeniden üretmek içindir; yeni kampanyada kullanılmaz.

Noktalar crop koordinatındadır. Anatomik noktalar görünmüyorsa tahmin edilmez;
örnek atlanır ve gerekçe kaydedilir. Withers ve sacrum aynı nokta olamaz. Sagitta
ve diğer geometri skorları noktalar yerleştirilirken canlı gösterilmez; ancak
annotation kaydedildikten sonra veya ayrı inspection panelinde hesaplanabilir.

## Neden iki ayrı geçiş var?

Canlı geometri, model skoru veya mevcut posture etiketi annotator'ı beklenen
sonuca doğru yönlendirebilir. Pass A insanın görsel posture kararını bu
bilgilerden kör biçimde toplar. Pass B ise aynı posture kararını değiştirmeden
geometriyi bağımsız bir ölçüm olarak kaydeder. Bu ayrım, geometriyi ground-truth
etiket gibi döngüsel kullanmayı ve reviewer alanlarının birbirine karışmasını
önler.

## Deneysel Pass C — 19-nokta tam postür

Ayak fazı ile sırt postürünü aynı crop üzerinde araştırmak için ayrı
`cow_pose_19_v1` şeması kullanılabilir. Bu şema Pass B'deki üretim sözleşmesini
değiştirmez: v1 ölçümü hâlâ yalnız `withers, sacrum, head` anchor'ları ve yoğun
segmentasyon profiliyle yapılır. Pass C verisi `pose_keypoints_json` alanında
saklanır; `keypoints_json` alanına yazılmaz.

Sabit tıklama sırası şöyledir:

| No | Nokta | Anatomik yer |
|---:|---|---|
| 1 | `poll` | Boynuzlar arasının hemen gerisi; oksipital/poll bölgesinin dorsal merkezi |
| 2 | `neck_mid` | Poll ile withers arasındaki dorsal boyun hattının anatomik orta noktası |
| 3 | `withers` | Kürek kemikleri üzerindeki cidago; gövde dorsal hattının ön çapası |
| 4–6 | `dorsal_25/50/75` | Withers–sacrum hattı boyunca yaklaşık %25, %50 ve %75 konumları; tüy siluetinin üst sınırı |
| 7 | `sacrum` | Kuyruk kökünün hemen önü; pelvis üzerindeki dorsal arka çapa |
| 8–10 | `near_front_*` | Kameraya yakın ön bacağın karpus, fetlock ve tırnak zemine temas merkezi |
| 11–13 | `far_front_*` | Kameradan uzak ön bacağın aynı üç noktası |
| 14–16 | `near_hind_*` | Kameraya yakın arka bacağın hock, fetlock ve tırnak zemine temas merkezi |
| 17–19 | `far_hind_*` | Kameradan uzak arka bacağın aynı üç noktası |

`near` görüntüdeki sağ/sol yönü değil, kameraya yakınlık anlamına gelir; hayvan
hangi yöne bakarsa baksın değişmez. Dorsal ara noktalar düz görüntü koordinatı
interpolasyonu değildir: gerçek tüy/sırt silueti üzerinde işaretlenir.

Görünürlük kuralı COCO/YOLO ile uyumludur:

- Sol tık (`v=2`): anatomik nokta doğrudan görülüyor.
- Sağ tık (`v=1`): nokta korkuluk veya diğer bacağın arkasında, fakat konumu
  komşu anatomiden güvenle çıkarılabiliyor.
- `0`/`M` (`v=0`): nokta crop dışında, ağır oklüzyonda veya güvenle tahmin
  edilemiyor. Koordinat otomatik olarak `(0,0)` kaydedilir.

Her noktayı her karede doldurmak hedef değildir. Özellikle korkuluk arkasında
uydurma ayak noktası, eksik etiketten daha zararlıdır. Aynı videonun ardışık
kareleri train/validation/test arasında bölünmez; passage bazında gruplanır.

Etiketleme turu:

```bash
python scripts/19_ingest_pose_media.py \
  --input-dir "/Users/anil/Downloads/inek data ve metadoloji" \
  --output-dir data/pose_round_01 \
  --target-fps 2

python scripts/16_label_pose.py \
  --manifest data/pose_round_01/manifest.csv \
  --reviewer anil
```

Etiketleme penceresi crop'ı kullanılabilir ekran alanına otomatik büyütür ve
üstteki kontrol bandı tıklama kabul etmez. `Z` veya `U` son noktayı geri alır,
`R` kareyi sıfırlar,
`Enter` tamamlanmış 19 pozisyonu kaydedip sonraki crop'a geçer. `N` o crop'ı
`pose_status=skipped` olarak kaydeder, `B` önceki crop'a döner ve `Q` mevcut
ilerlemeyi kaydedip çıkar. Atlanan veya etiketlenmiş örnekleri yeniden açmak için
`--include-reviewed` kullanılır.

Manifest ayrıca `pose_candidate` ve `pose_prefilter_reason` alanlarını taşır.
Bu yalnız ilk kuyruğu hızlandıran otomatik bir ön filtredir; anatomi etiketi
değildir. Kenardan kesilmiş, çok küçük, düşük güvenli veya belirgin non-lateral
kutular varsayılan kuyrukta gösterilmez. Bunların tümünü ayrıca incelemek için
`--include-noncandidates` kullanılabilir.

## Session kalite kontrolü

- Reviewer kimliği, protokol sürümü ve session tarihi kaydedilir.
- Yaklaşık 100–150 passage iki kişi tarafından bağımsız ordinal skorlanır;
  quadratic weighted kappa raporlanır. Anlaşmazlıklar orijinal kararların üzerine
  yazılmaz.
- Etiket değişikliği gerekiyorsa kim, ne zaman ve neden değiştirdiği audit notuna
  eklenir.
- Sınıf dağılımı, `uncertain`/`invalid` oranı ve nedenleri reviewer ve
  standing/walking durumuna göre kontrol edilir.
