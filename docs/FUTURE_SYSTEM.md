# Future System — Mevcut PoC'de Doğrulanmadı

> **Durum: future phase / mevcut PoC'de doğrulanmadı.** Bu belge bir araştırma
> yönü taslağıdır; mevcut repoda uygulanmış veya sahada valide edilmiş özellikler
> olarak okunmamalıdır. Repo artık deterministik passage agregasyonu, kişisel
> baseline ve offline triyaj sinyali kodunu içerir; bunlar P2 kapısı ve saha
> validasyonu geçmeden klinik kabiliyet sayılmaz. Sistem lameness/hastalık tanısı
> üretmez.

## Future phase — Temporal BiLSTM/TCN (doğrulanmadı)

Ardışık frame embedding'leri veya posture/geometri özellikleri passage bazında
dizilere dönüştürülerek BiLSTM ve TCN adayları karşılaştırılabilir. Bu aşamada:

- sekans sınırları `video_id`/`passage_id` ile korunmalı;
- padding, sampling rate, eksik frame ve değişken passage uzunluğu açıkça
  tanımlanmalı;
- split cow veya passage grubu düzeyinde yapılmalı;
- tek-frame baseline'a karşı group-level ve prospective değerlendirme yapılmalı;
- standing ve walking sekansları karıştırılmadan raporlanmalıdır.

BiLSTM/TCN'nin tek-kare modelinden daha iyi olacağı mevcut PoC tarafından
gösterilmemiştir.

## Future phase — RFID / cow re-ID (doğrulanmadı)

Longitudinal analiz için gözlemleri doğru cow'a bağlayan bir identity katmanı
gerekebilir. Aday tasarımda RFID kontrollü birincil kimlik; RFID olmayan veya
kaçırılan olaylarda ayrıca valide edilmiş görsel cow re-identification yardımcı
sinyal olabilir. RFID ile görüntü zaman eşlemesi, duplicate/mismatch oranı,
unknown identity politikası ve gizlilik/erişim kuralları prospective olarak test
edilmelidir. Mevcut PoC RFID okumaz ve cow re-ID yapmaz.

## Future phase — 200 günlük history (doğrulanmadı)

Önerilen sonraki sistem, cow başına zaman damgalı posture probability,
geometri/kalite özeti, passage ve model sürümüyle birlikte **200 günlük history**
tutmayı araştırabilir. Saklama penceresi veri sahipliği, izin, güvenlik, silme ve
yerel mevzuat kontrollerinden sonra kesinleştirilmelidir. Eksik günler sıfır risk
olarak doldurulmamalı; identity belirsiz gözlemler bir cow geçmişine zorla
eklenmemelidir. Mevcut PoC 200 günlük veri saklama veya geçmiş sorgulama özelliği
içermez.

## Future phase — Alert ve clinical feedback (doğrulanmadı)

Offline kapasite-kalibreli triyaj sinyali mevcut olsa da, kullanıcıya dönük alert
ve clinical feedback ancak identity ve longitudinal baseline sahada bağımsız
olarak valide edildikten sonra araştırılabilir. Aday akış:

1. Tek bir frame yerine tekrarlanan passage/günlerde kalıcı sapmayı değerlendirme.
2. Görüntü kalitesi ve identity güveni düşükse alert bastırma veya review kuyruğuna
   alma.
3. Alert'i “klinik inceleme önerisi” olarak sunma; hastalık/lameness tanısı ya da
   otomatik tedavi kararı olarak sunmama.
4. Veteriner/farm reviewer'ın `confirmed`, `dismissed`, `needs follow-up` gibi
   yapılandırılmış clinical feedback vermesi.
5. Feedback'i orijinal posture ground truth'un üzerine yazmadan, ayrı audit trail
   ve model sürümüyle saklama.
6. False-alert yükü, missed event, time-to-review ve cow/farm bazında performansı
   prospective çalışma ile ölçme.

Clinical feedback döngüsü saha verisiyle doğrulanmamıştır.

## Future phase için geçiş kapıları

Aşağıdakiler tamamlanmadan bu taslak özellikler “sistem kabiliyeti” diye raporlanmamalıdır:

- açık provenance/lisans ve yeterli cow/farm/passage çeşitliliği;
- güvenilir cow identity eşlemesi ve hata analizi;
- train/validation/test gruplarının cow veya passage düzeyinde ayrılması;
- tek-frame baseline'a karşı temporal ve group-level değerlendirme;
- 200 günlük history için yönetişim, saklama ve silme politikası;
- önceden tanımlı alert eşiğiyle prospective saha validasyonu;
- veteriner gözetimi, clinical feedback protokolü ve güvenli escalation süreci.

Bu kapılar bir roadmap önerisidir; hiçbiri mevcut PoC'de tamamlanmış veya valide
edilmiş kabul edilmez.
