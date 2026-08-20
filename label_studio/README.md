# CowArch Label Studio integration

Bu dizin Label Studio Community 1.23.0'ı yalnız `127.0.0.1:8080` üzerinde çalıştırır. `data/`
container içine salt-okunur bağlanır; Label Studio görüntüleri silemez veya
değiştiremez. Uygulama veritabanı bir Docker volume'ünde kalıcıdır.

## 1. Servisi başlat

Proje kökünden:

```bash
docker compose -f label_studio/docker-compose.yml up -d
```

Ardından <http://127.0.0.1:8080> adresini aç ve yerel kullanıcı hesabını oluştur.

## 2. Pass A görevlerini üret

Görevler **split başına ayrı** üretilir. Sebebi protokol: test ve validation
kör ve rastgele sırada etiketlenmeli, train ise sonradan active-learning ile
önceliklendirilebilir. Hepsi tek dosyada olursa bu ayrım kaybolur.

```bash
source .venv/bin/activate
for SPLIT in test val train; do
  python scripts/09_label_studio.py export-tasks \
    --phase posture \
    --manifest data/manifest.csv \
    --data-root data \
    --split ${SPLIT} \
    --output label_studio/tasks/pass_a_${SPLIT}.json
done
```

`--approved-sources` bilerek kullanılmıyor: lisansı `unresolved` olan
CattleLameness crop'ları kullanıcı kararıyla PoC'ye dahil edildi. Her task
kendi `license_status` alanını taşır, dolayısıyla yayın aşamasında filtrelemek
mümkün kalır. Yalnız lisanslı alt kümeyle çalışmak istersen bayrağı geri ekle.

Label Studio'da **split başına bir proje** oluştur (`Pass A — test` gibi):

1. Labeling Setup içindeki code görünümüne
   `label_studio/configs/pass_a_posture.xml` içeriğini yapıştır.
2. O split'in task dosyasını import et (önce `pass_a_test.json`).
3. Etiketlemeyi tamamladıktan sonra JSON export al ve
   `label_studio/exports/pass_a_<split>.json` olarak kaydet.

Sonucu yeni bir manifestte birleştir:

```bash
python scripts/09_label_studio.py import-results \
  --phase posture \
  --manifest data/manifest.csv \
  --export label_studio/exports/pass_a_posture.json \
  --output-manifest data/manifest_posture.csv \
  --reviewer posture-reviewer-01
```

Kaynak manifestin üzerine yazılmaz. Yeni CSV'yi kontrol ettikten sonra sonraki
aşamanın girdisi olarak kullan.

## 3. Pass B görevlerini üret

Yalnız `arched` veya `normal` kararı bulunan ve henüz geometri etiketi olmayan
satırlar seçilir:

```bash
python scripts/09_label_studio.py export-tasks \
  --phase geometry \
  --manifest data/manifest_posture.csv \
  --data-root data \
  --approved-sources data/sources_selected.csv \
  --output label_studio/tasks/pass_b_geometry.json
```

İkinci bir Label Studio projesinde
`label_studio/configs/pass_b_geometry.xml` yapılandırmasını ve üretilen görev
dosyasını kullan. Her isimden tam bir nokta bulunmalıdır; importer eksik veya
tekrarlı noktaları reddeder.

Export sonucunu birleştir:

```bash
python scripts/09_label_studio.py import-results \
  --phase geometry \
  --manifest data/manifest_posture.csv \
  --export label_studio/exports/pass_b_geometry.json \
  --output-manifest data/manifest_labeled.csv \
  --reviewer geometry-reviewer-01
```

## Güvenlik kuralları

- İki pass ayrı projedir; Pass A geometri veya weak-label bilgisini göstermez.
- Önerilen komutlar yalnız `license_status=approved` kaynak manifestini kabul
  eder; `check-before-use` videolar pilot görevlere girmez.
- Import varsayılan olarak mevcut, farklı bir etiketi ezmez.
- Çoklu tamamlanmış annotation otomatik çoğunluk oyuna çevrilmez; önce insan
  adjudication gerekir.
- `--allow-overwrite` yalnız açıkça incelenmiş bir düzeltme için kullanılmalıdır.
- Servisi durdurmak veriyi silmez:

```bash
docker compose -f label_studio/docker-compose.yml down
```

`down -v` Docker volume'ündeki Label Studio veritabanını siler; normal kullanımda
çalıştırılmamalıdır.
