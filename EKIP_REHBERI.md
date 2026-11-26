# BIST ScoreCard: Ekip Rehberi

Bu dosya ekip içindir: projeyi kurmak, nasıl çalıştığını anlamak ve sunumda
kendi bölümünü anlatabilmek için. YZ asistanı (Claude, ChatGPT vb.) kullanıyorsan
ona önce bu dosyayı ve `README.md`'yi okut, sonra soru sor.

## 1. Proje ne yapıyor?

BIST100'deki her hisseye 0-100 arası bir puan veriyor ve AL / TUT / SAT diyor.
Ama asıl amaç bir "al" düğmesi değil: kimse buradan tek başına hisse almasın,
fakat bakınca hisse hakkında bir fikri olsun. Bu yüzden her hissenin yanında
şunlar var:

- düz İngilizceyle 2-6 cümlelik bir yorum ("Sakin bir hisse, istikrarlı yükseliş
  trendinde, sektörünün önünde..."),
- güçlü ve zayıf yönleri, risk uyarıları,
- F/K ve PD/DD (sektörüyle karşılaştırmalı, puana girmez, sadece bilgi).

Sonra sistem kendi geçmiş kararlarını test ediyor (backtest): 2021'den beri her
ay o günkü veriyle karar verseydi ne olurdu?

## 2. Kurulum

Python **3.12 veya üstü** gerekir (`python3 --version` ile bak).

**Mac / Linux**
```bash
git clone https://github.com/ozanaktas12/bist-scorecard.git
cd bist-scorecard
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
git clone https://github.com/ozanaktas12/bist-scorecard.git
cd bist-scorecard
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Çalıştırma**
```bash
streamlit run app/streamlit_app.py     # panel tarayıcıda açılır (localhost:8501)
pytest                                 # 39 test, hepsi geçmeli
python scripts/run_backtest.py         # backtest tablolarını results/ klasörüne yazar (~1 dk)
```

İnternet gerekmez: veriler `data/cache/` içinde hazır (fiyatlar 25 Kas 2026'ya
kadar). Taze veri istersen o klasörü silip tekrar çalıştır, Yahoo'dan indirir
(birkaç dakika sürer, sonuçlar rapordakinden biraz farklı çıkar).

Streamlit bir Python kütüphanesi; panel bir web sayfası gibi görünür ama sadece
senin bilgisayarında çalışır, internette yayında değildir. Reklam yoktur.

## 3. Sistem nasıl çalışıyor? (veri akışı)

```
Yahoo fiyatları ──► temizleme ──► 16 gösterge ──► 16 metrik puanı (0-100)
                                                       │
                          5 başlık: Trend, Momentum, Risk, Göreceli Güç, Hacim
                                                       │
                          profil ağırlıklarıyla toplam puan (0-100)
                                                       │
                AL / TUT / SAT kararı + risk uyarıları + yorum cümleleri ──► panel
                                                       │
                          backtest: geçmişteki her ay için aynısı + sonucu ölç
```

- **3 vade:** kısa (1 ay), orta (3 ay), uzun (12 ay). Vade, göstergelerin kaç
  günlük veriye baktığını değiştirir (ör. orta vadede 50 ve 100 günlük ortalama).
- **4 profil:** Conservative, Balanced, Aggressive (elle seçilmiş ağırlıklar) ve
  Data-driven (ağırlıkları backtest seçti: 2021-2023'te arandı, 2024-2026'da test edildi).
- **Karar kuralı:** AL = puan ≥ 60 **ve** o gün en iyi %20 içinde.
  SAT = o gün en kötü %20. Geri kalan TUT.
- **Risk uyarıları kararı değiştirmez**, sadece "AL ama dikkat" der
  (ör. aşırı alım, düşen bıçak, çok sığ işlem).

## 4. İş bölümü

Her kişi sunumda kendi bölümünü anlatır ve o bölümle ilgili sorulara cevap verir.

### Ozan Aktaş: veri ve panel
Dosyalar: `scorecard/data/` (tüm klasör), `app/streamlit_app.py`, `app/ui.py`

- `provider.py`: Yahoo'dan fiyat çekme. `DataProvider` soyut sınıf,
  `YFinanceProvider` onu uygular, `CachedProvider` her sağlayıcıyı sarıp diske
  kaydeder (decorator kalıbı). Yahoo bazı hisseleri atlarsa tek tek tekrar dener.
- `cleaning.py`: Yahoo bedelsiz sermaye artırımlarını bazen kaçırır, fiyat bir
  günde yarıya düşmüş gibi görünür. Günlük ±%10 limitini aşan sıçrama hata sayılıp
  düzeltilir; **ama** endeks de aynı gün sıçradıysa (15 Şubat 2023, deprem sonrası
  borsanın açılışı) gerçek hareket sayılır.
- `universe.py`: Geçmişteki BIST100 listeleri. Bugünün listesinden başlayıp
  Borsa İstanbul'un her çeyrek duyurduğu giren/çıkan hisseleri geri alarak
  çalışır: `önceki üyeler = sonraki üyeler − girenler + çıkanlar`. Her adımda
  endeksin tam 100 hisse olduğu kontrol edilir.
- `valuation.py`: F/K ve PD/DD. Yahoo'nun hazır oranları dolarla raporlayan
  şirketlerde yanlış (THY'nin PD/DD'si 0.38 yerine 18.5 görünüyordu), bu yüzden
  ham verilerden kur çevirerek hesaplanır. Fintables ile kontrol edildi.
- Panel: 4 sekme (Scorecard, Ranking, Compare, Backtest). `ui.py` renkleri,
  grafik stilini ve SAT/TUT/AL barını tutar.

### Demir Bigalı: göstergeler ve puanlama
Dosyalar: `scorecard/indicators.py`, `scorecard/scoring/` (tüm klasör), `scorecard/config.py`

- `indicators.py`: 16 teknik gösterge, hazır teknik analiz kütüphanesi kullanmadan
  pandas/NumPy ile yazıldı: hareketli ortalamalar, RSI, MACD, volatilite, beta, maksimum düşüş,
  OBV, Amihud likidite, sektöre göre getiri vb. Her gösterge sadece o güne kadarki
  veriyi kullanır (gelecek veriye bakmaz). `@register` dekoratörü göstergeleri bir
  listeye ekler.
- `metrics.py`: 16 metrik ve her ham değerin 0-100 puana nasıl çevrildiği
  (parça parça doğrusal eşleme; ör. RSI 55-70 en iyi, 80 üstü aşırı alım cezası).
- `engine.py`: Metrikleri 5 başlıkta toplar, profil ağırlıklarıyla toplam puanı
  ve AL/TUT/SAT kararını üretir; hissenin güçlü/zayıf yönlerini bulur.
- `flags.py`: Risk uyarıları (kırmızı ve turuncu).
- `narrative.py`: Yorum cümleleri. Serbest metin üretilmez; her cümle bir
  metriğe bağlı sabit bir kuraldan gelir, bu yüzden test edilebilir.
- `config.py`: Vadeler, profil ağırlıkları, karar eşikleri; ayarlanabilen her
  sayı burada.

### Arda Özköse: backtest ve testler
Dosyalar: `scorecard/backtest/` (tüm klasör), `scripts/run_backtest.py`, `tests/`

- `engine.py`: 2021'den beri her ayın ilk işlem günü, sadece o güne kadarki
  veriyle karar verilir; işlem ertesi gün yapılır, %0.2 işlem maliyeti düşülür.
  Sonra hissenin vade boyunca BIST100'e göre getirisi ölçülür.
  - **İsabet oranı:** AL dediklerinin yüzde kaçı endeksi geçti?
  - **t-değeri (Newey-West):** fark şans eseri mi? |t| > 2 ise anlamlı sayılır.
    3 ve 12 aylık sonuçlar üst üste bindiği için düz t-testi fazla iyimser olurdu,
    Newey-West bunu düzeltir.
  - **Portföy simülasyonu:** her ay AL listesini alan bir yatırımcının getirisi;
    enflasyondan arındırılmış (reel) getiri de hesaplanır (TCMB TÜFE).
- `calibration.py`: Data-driven profilin ağırlıklarını bulur. 2021-2023
  verisinde en iyi ağırlıkları arar, 2024-2026'da hiç görmediği veride test eder
  (aşırı uyumu önlemek için).
- `tests/`: 39 test. En önemlisi `test_no_lookahead.py`: t günündeki karar,
  t'den sonraki bütün veri silinince de aynı kalmalı. Yani sistem geleceği görmüyor.

## 5. Sonuçlar (results/ klasöründen)

Orta vade (3 ay), 2021-2026, her tarihte o günkü BIST100 listesiyle:

| | Data-driven | Balanced |
|---|---|---|
| AL isabeti (rastgele hisse: %41) | %48 | %47 |
| t-değeri (AL, tüm hisselere karşı) | 4.8 | 3.0 |
| SAT isabeti (endeksin altında kaldı) | %68 | %66 |
| Yıllık getiri (BIST100: %40, eşit ağırlık: %34.5) | %50.4 | %41.2 |
| Enflasyondan arındırılmış yıllık getiri | +%4.4 | −%1.8 |

- 12 profil-vade kombinasyonunun 6'sında AL kararları tüm hisselerden anlamlı
  şekilde iyi (t > 2); AL ile SAT arasındaki fark ise 8'inde anlamlı.
- Puan arttıkça sonuç düzenli iyileşiyor: uzun vadede en düşük puanlı %20'lik
  grup endeksin %15 altında, en yüksek grup %14 üstünde kaldı.
- **Geçmiş listeleri kullanmak önemliydi:** bugünün listesini geçmişe uygulamak
  piyasayı olduğundan iyi gösteriyor (eşit ağırlık %49.8 yerine %34.5) ve sistemin
  başarısını gizliyordu (t 2.5 yerine 4.8), çünkü sonradan çöküp endeksten düşen
  hisseler listede olmuyordu; sistem tam onlara SAT diyordu.

**Dürüstçe söylememiz gerekenler:**
- Kısa vade (1 ay) anlamlı değil; panel de bunu uyarıyor.
- 2026 kötü geçti: Balanced −%27, BIST100 −%7.
- Değerleme puana girmiyor, çünkü Yahoo'da sadece bugünün bilançosu var; geçmişe
  dönük test edilemez.
- Bu bir ders projesi, yatırım tavsiyesi değil.

## 6. YZ asistanıyla çalışırken

Örnek komutlar:
- "EKIP_REHBERI.md'yi oku. Sonra `scorecard/data/universe.py` dosyasını bana
  satır satır, Türkçe ve basitçe anlat."
- "Benim bölümüm Demir'in bölümü. Hoca sunumda neler sorabilir? Cevaplarıyla 15 soru yaz."
- "`pytest` çalıştır ve sonucu açıkla."

Kodu değiştirmek istersen önce Ozan'a haber ver; doğrudan `main` dalına push
yapma, ayrı bir dal aç.

Raporda bir YZ beyanı (Student AI Declaration) var: projede YZ kullanıldı ve bu
açıkça belirtiliyor. Bu yüzden önemli olan, herkesin kendi bölümünü anlayıp
anlatabilmesi.
