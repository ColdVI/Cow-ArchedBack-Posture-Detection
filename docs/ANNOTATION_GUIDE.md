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

## Session kalite kontrolü

- Reviewer kimliği, protokol sürümü ve session tarihi kaydedilir.
- Yaklaşık 100–150 passage iki kişi tarafından bağımsız ordinal skorlanır;
  quadratic weighted kappa raporlanır. Anlaşmazlıklar orijinal kararların üzerine
  yazılmaz.
- Etiket değişikliği gerekiyorsa kim, ne zaman ve neden değiştirdiği audit notuna
  eklenir.
- Sınıf dağılımı, `uncertain`/`invalid` oranı ve nedenleri reviewer ve
  standing/walking durumuna göre kontrol edilir.
