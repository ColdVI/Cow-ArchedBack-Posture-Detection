# Dataset Card Template

Bu şablon, her dataset sürümünde gerçek değerlerle doldurulmalıdır. Köşeli
parantezli alanlar placeholder'dır; bilinmeyen bilgiler boş bırakılmak yerine
`unknown` olarak ve nedeni ile kaydedilir.

## 1. Kimlik ve sürüm

| Alan | Değer |
|---|---|
| Dataset adı | [name] |
| Sürüm / release tarihi | [version, YYYY-MM-DD] |
| Dataset sahibi / sorumlusu | [organization, contact] |
| Annotation protocol sürümü | [version veya commit] |
| Manifest checksum / commit | [value] |
| Kapsam | [lateral single-frame arched-back posture] |

## 2. Amaç ve kapsam dışı kullanım

- **Amaçlanan kullanım:** [insan tarafından verilen `arched/normal` görsel
  posture etiketinin araştırma amaçlı sınıflandırılması]
- **Kapsam dışı:** Lameness/hastalık tanısı, tedavi kararı, kimlik tespiti veya
  klinik gözetim yerine kullanım.
- **Population ve ortam:** [farmalar, hayvan grubu, kamera ortamı, tarih aralığı]

## 3. Kaynak, kimlik ve provenance

Her kaynak ve örnek için aşağıdaki alanların doluluk oranını raporlayın:

| Alan | Açıklama / değer |
|---|---|
| `sample_id` | Dataset sürümü içinde benzersiz örnek kimliği |
| `source_id` | Kaynak kaydı kimliği |
| `farm_id` | Farm kimliği; bilinmiyorsa `unknown` |
| `cow_id` | Kalıcı cow kimliği; bilinmiyorsa `unknown` |
| `video_id` | Kaynak video kimliği |
| `passage_id` | Aynı geçiş/sekans içindeki kareleri bağlayan kimlik |
| `camera_id` | Kamera veya kurulum kimliği |
| `frame_idx` / timestamp | Video içindeki konum |
| `kind` | `video`, `image`, `image_dir` vb. |
| `url` / `local_path` | Kaynağın izlenebilir konumu; yayımlanan kartta hassas path redakte edilebilir |
| acquisition | [tarih, toplama yöntemi, fps/frame sampling, camera/view] |
| transformations | [detection, crop, resize, deduplication, kalite filtreleri] |

Provenance özeti: [veriyi kimin, nerede, hangi izinle topladığı; üçüncü taraf
kaynaklarda orijinal yayıncı ve erişim tarihi]. Üretilmiş crop/maskelerin hangi
orijinal kayıttan geldiği manifest üzerinden geriye izlenebilmelidir.

## 4. Lisans ve kullanım izni

Her `source_id` için aşağıdaki alanları raporlayın:

| Alan | Değer |
|---|---|
| `license_status` | `approved`, `restricted` veya `unresolved` |
| `license_name` | [lisans/izin adı] |
| `license_url` | [koşulların doğrudan bağlantısı] |
| owner / permission evidence | [sahip ve yazılı izin kaydı] |
| allowed uses | [training, evaluation, internal, redistribution vb.] |
| restrictions / expiry | [kısıtlar ve son tarih] |

Yalnız `approved` kaynaklar işlenebilir. `restricted` ve `unresolved` kayıtlar
dataset sayımlarına dahil edilmez. Bir platformda herkese açık görünmek, eğitim
veya yeniden dağıtım izni anlamına gelmez.

## 5. Annotation

- Label set: `arched`, `normal`, `uncertain`, `invalid`.
- Posture protocol: [ANNOTATION_GUIDE sürümü ve sapmalar].
- Posture labeler'lar: [anonim reviewer ID, eğitim/uzmanlık, session tarihleri].
- Geometry protocol: [five keypoints / two anchors].
- Geometry labeler'lar: [anonim reviewer ID ve session tarihleri].
- Manifest alanları: `posture_reviewed_by`, `geometry_reviewed_by`,
  `annotation_pass`; legacy `reviewed_by` kullanıldıysa migration açıklaması.
- Blinding: [model/source/geometri bilgileri Pass A'da nasıl gizlendi].
- Adjudication: [double-label örnek sayısı, anlaşmazlık ve çözüm yöntemi].
- Missing/skip politikası: [nedenler ve sayılar].

## 6. Sınıf ve grup dağılımı

Frame/crop sayıları ile bağımsız biyolojik/kayıt gruplarını ayrı raporlayın:

| Split | `arched` frames | `normal` frames | `uncertain` | `invalid` | farms | cows | videos | passages |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | [n] | [n] | [n] | [n] | [n] | [n] | [n] | [n] |
| validation | [n] | [n] | [n] | [n] | [n] | [n] | [n] | [n] |
| test | [n] | [n] | [n] | [n] | [n] | [n] | [n] | [n] |

Ayrıca standing/walking/unknown, lateral/oblique, source, farm, camera, reviewer
ve temel kalite/occlusion kategorilerine göre dağılımları ekleyin. Hariç tutulan
örnekleri ve reject nedenlerini de raporlayın.

## 7. Split strategy ve leakage kontrolleri

- Split group column: [`video_id`, `cow_id` veya `passage_id`; seçim gerekçesi].
- Split üretim tarihi, seed ve oranlar: [values].
- Aynı group değerinin birden fazla split'te olmadığını doğrulayan kontrol:
  [komut/sonuç].
- Aynı cow farklı video/passage kimlikleriyle görünüyorsa uygulanan üst seviye
  gruplama: [policy].
- Near-duplicate ve ardışık-frame leakage kontrolü: [method/result].
- Active learning'in yalnız train split'inde çalıştığı kontrol: [result].
- Validation/test'in split ve label kararlarının ne zaman kilitlendiği: [date].

### Bağımsızlık uyarısı

**Aynı video, cow veya passage içindeki frame'ler bağımsız örnek değildir.**
Dolayısıyla toplam frame/crop sayısı bağımsız hayvan ya da bağımsız olay sayısı
olarak sunulamaz. Split, değerlendirme ve confidence interval hesabı seçilen
grup düzeyini korumalıdır; frame-level metrikler tek başına genellenebilirlik
kanıtı değildir. Kart, hem frame sayısını hem benzersiz farm/cow/video/passage
sayılarını açıkça vermelidir.

## 8. Değerlendirme birimi

- Frame-level metrikler: [metrics, threshold, result].
- Group/passage aggregation: [group column, varsayılan mean probability].
- Group-level metrikler: [metrics, result].
- Confidence interval: [group-resampled cluster bootstrap ayarları ve sonuç].
- Aynı grup içindeki çelişkili ground-truth label kontrolü: [result].

## 9. Known biases ve limitations

Her madde için etkilenen grubu, olası yönü ve azaltma/izleme planını yazın:

- farm/source ile sınıf etiketi arasındaki korelasyon;
- cow, ırk, yaş/laktasyon veya vücut kondisyonu temsil dengesizliği;
- standing/walking ve gait fazı dengesizliği;
- head-down, turning ve diğer posture confounder'ları;
- lateral/oblique açı, camera, çözünürlük, ışık, gölge ve arka plan;
- occlusion, crop/detection/segmentation seçilim yanlılığı;
- internetteki title/description weak label'larının sınırı;
- labeler drift'i, reviewer dağılımı ve anlaşmazlık;
- bilinmeyen `cow_id` nedeniyle gizli tekrarlar ve etkili örnek sayısı;
- yalnız belirli farm/source'larda test edilmiş olmanın dış geçerlilik sınırı.

## 10. Veri yönetişimi

- Saklama süresi ve erişim kontrolü: [policy].
- Kişisel/ticari hassas bilgi ve redaction: [policy].
- Düzeltme, silme ve yeni sürüm süreci: [policy].
- Bilinen sorunlar ve değişiklik günlüğü: [link/section].
