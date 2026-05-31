# Special mora outlier examples

Examples are selected from sample-level audit rows to make thresholds inspectable.

## long_vowel
- shortest:
  - ニューイングランド風は、牛乳をベースとした、白いクリームスープであり、ボストンクラムチャウダーとも呼ばれる。: long_vowel ー phones=[u] ratio=0.2026 duration=0.03 decision=too_short warnings=
  - シャンチーの専業プロは、チームから支払われる給料と、対局費を、主な収入としている。: long_vowel ー phones=[u] ratio=0.2051 duration=0.03 decision=too_short warnings=
  - また、東寺のように、五大明王と呼ばれる、主要な明王の中央に配されることも多い。: long_vowel ー phones=[o] ratio=0.2133 duration=0.03 decision=too_short warnings=
  - シルバーサーファー襲撃事件までに、リチャーズは、チーム名と共に、国際的にスーパーヒーロー、および、有名人として、認知されている。: long_vowel ー phones=[e] ratio=0.2175 duration=0.03 decision=too_short warnings=
  - また禰寝氏は、中山王の治める、琉球王国との交易にも参加した。: long_vowel ー phones=[u] ratio=0.2229 duration=0.03 decision=too_short warnings=
- longest:
  - また、東寺のように、五大明王と呼ばれる、主要な明王の中央に配されることも多い。: long_vowel ー phones=[o] ratio=1.6352 duration=0.23 decision=too_long warnings=
  - このときに、浮遊大陸プルヴァマにある、中立国、ビュエルバが、ある情報筋から、バッシュ将軍の処刑と、前王女、アーシェの自害を発表。: long_vowel ー phones=[a] ratio=1.3844 duration=0.19 decision=too_long warnings=
  - また禰寝氏は、中山王の治める、琉球王国との交易にも参加した。: long_vowel ー phones=[o] ratio=1.2629 duration=0.17 decision=too_long warnings=
  - 一方で、漁業と商業で、リャネス港は繁栄していた。: long_vowel ー phones=[e] ratio=1.2526 duration=0.17 decision=too_long warnings=
  - このときに、浮遊大陸プルヴァマにある、中立国、ビュエルバが、ある情報筋から、バッシュ将軍の処刑と、前王女、アーシェの自害を発表。: long_vowel ー phones=[o] ratio=1.1658 duration=0.16 decision=too_long warnings=
- near low threshold:
  - 町域にあった、三根山藩は、長岡藩に、米百俵を送ったことで有名。: long_vowel ー phones=[e] ratio=0.2325 duration=0.03 decision=ok warnings=
  - ベルガートーア前の、ヴェディゲンウーファーパークには、戦争と弾圧の犠牲者のための記念碑が建っている。: long_vowel ー phones=[u] ratio=0.2339 duration=0.03 decision=ok warnings=
  - 全米パブリッシャーズ協会の、ベストストラテジーゲームオブザイヤーを、日本人として受賞。: long_vowel ー phones=[a] ratio=0.2346 duration=0.03 decision=ok warnings=
  - コンピュータゲームのメーカーや、業界団体などに関連する人物のカテゴリ。: long_vowel ー phones=[u] ratio=0.2372 duration=0.03 decision=ok warnings=
  - サービスマネージャー導入駅のため、大井町駅から、遠隔管理している。: long_vowel ー phones=[u] ratio=0.2375 duration=0.03 decision=ok warnings=
- near high threshold:
  - 全米パブリッシャーズ協会の、ベストストラテジーゲームオブザイヤーを、日本人として受賞。: long_vowel ー phones=[e] ratio=1.0948 duration=0.14 decision=ok warnings=
  - 全米パブリッシャーズ協会の、ベストストラテジーゲームオブザイヤーを、日本人として受賞。: long_vowel ー phones=[o] ratio=1.0948 duration=0.14 decision=ok warnings=
  - また禰寝氏は、中山王の治める、琉球王国との交易にも参加した。: long_vowel ー phones=[o] ratio=1.04 duration=0.14 decision=ok warnings=
  - シャンチーの専業プロは、チームから支払われる給料と、対局費を、主な収入としている。: long_vowel ー phones=[ny] ratio=1.0253 duration=0.15 decision=ok warnings=
  - このときに、浮遊大陸プルヴァマにある、中立国、ビュエルバが、ある情報筋から、バッシュ将軍の処刑と、前王女、アーシェの自害を発表。: long_vowel ー phones=[o] ratio=1.1658 duration=0.16 decision=too_long warnings=
- mapping warning examples:

## sokuon
- shortest:
  - ただし、ギャンブル依存症の入院治療を行っている病院は、わずかである。: sokuon ッ phones=[t] ratio=0.2026 duration=0.03 decision=uncertain warnings=
  - このときに、浮遊大陸プルヴァマにある、中立国、ビュエルバが、ある情報筋から、バッシュ将軍の処刑と、前王女、アーシェの自害を発表。: sokuon ッ phones=[cl] ratio=0.2186 duration=0.03 decision=uncertain warnings=
  - 町域にあった、三根山藩は、長岡藩に、米百俵を送ったことで有名。: sokuon ッ phones=[cl] ratio=0.2325 duration=0.03 decision=uncertain warnings=
  - 全米パブリッシャーズ協会の、ベストストラテジーゲームオブザイヤーを、日本人として受賞。: sokuon ッ phones=[cl] ratio=0.2346 duration=0.03 decision=uncertain warnings=
  - 全米パブリッシャーズ協会の、ベストストラテジーゲームオブザイヤーを、日本人として受賞。: sokuon ッ phones=[p] ratio=0.2346 duration=0.03 decision=uncertain warnings=
- longest:
  - ウエットティッシュ: sokuon ッ phones=[] ratio=1.0005 duration=0.1515 decision=uncertain warnings=
  - がっしり: sokuon ッ phones=[] ratio=1.0003 duration=0.1738 decision=uncertain warnings=
  - ばっちり: sokuon ッ phones=[] ratio=1.0 duration=0.195 decision=uncertain warnings=
  - うっとうしい: sokuon ッ phones=[] ratio=1.0 duration=0.15 decision=uncertain warnings=
  - ゆったり: sokuon ッ phones=[] ratio=1.0 duration=0.195 decision=uncertain warnings=
- near low threshold:
- near high threshold:
- mapping warning examples:

## moraic_nasal
- shortest:
  - 一方で、漁業と商業で、リャネス港は繁栄していた。: moraic_nasal ン phones=[N] ratio=0.2211 duration=0.03 decision=too_short warnings=
  - 町域にあった、三根山藩は、長岡藩に、米百俵を送ったことで有名。: moraic_nasal ン phones=[N] ratio=0.2325 duration=0.03 decision=too_short warnings=
  - ただし、ギャンブル依存症の入院治療を行っている病院は、わずかである。: moraic_nasal ン phones=[N] ratio=0.2701 duration=0.04 decision=too_short warnings=
  - また禰寝氏は、中山王の治める、琉球王国との交易にも参加した。: moraic_nasal ン phones=[N] ratio=0.2971 duration=0.04 decision=ok warnings=
  - ベルガートーア前の、ヴェディゲンウーファーパークには、戦争と弾圧の犠牲者のための記念碑が建っている。: moraic_nasal ン phones=[N] ratio=0.3119 duration=0.04 decision=ok warnings=
- longest:
  - ベルガートーア前の、ヴェディゲンウーファーパークには、戦争と弾圧の犠牲者のための記念碑が建っている。: moraic_nasal ン phones=[N] ratio=1.5596 duration=0.2 decision=too_long warnings=
  - 時間領域と、空間領域で共通する処理手法は、フィルタリングによる、入力信号の強化である。: moraic_nasal ン phones=[N] ratio=1.2654 duration=0.17 decision=too_long warnings=
  - このときに、浮遊大陸プルヴァマにある、中立国、ビュエルバが、ある情報筋から、バッシュ将軍の処刑と、前王女、アーシェの自害を発表。: moraic_nasal ン phones=[N] ratio=1.1658 duration=0.16 decision=too_long warnings=
  - ただし、ギャンブル依存症の入院治療を行っている病院は、わずかである。: moraic_nasal ン phones=[w] ratio=1.0128 duration=0.15 decision=ok warnings=
  - ファンタジーゲーム: moraic_nasal ン phones=[] ratio=1.0005 duration=0.1557 decision=uncertain warnings=
- near low threshold:
  - ただし、ギャンブル依存症の入院治療を行っている病院は、わずかである。: moraic_nasal ン phones=[N] ratio=0.2701 duration=0.04 decision=too_short warnings=
  - また禰寝氏は、中山王の治める、琉球王国との交易にも参加した。: moraic_nasal ン phones=[N] ratio=0.2971 duration=0.04 decision=ok warnings=
  - ベルガートーア前の、ヴェディゲンウーファーパークには、戦争と弾圧の犠牲者のための記念碑が建っている。: moraic_nasal ン phones=[N] ratio=0.3119 duration=0.04 decision=ok warnings=
  - 町域にあった、三根山藩は、長岡藩に、米百俵を送ったことで有名。: moraic_nasal ン phones=[N] ratio=0.2325 duration=0.03 decision=too_short warnings=
  - コンピュータゲームのメーカーや、業界団体などに関連する人物のカテゴリ。: moraic_nasal ン phones=[N] ratio=0.3162 duration=0.04 decision=ok warnings=
- near high threshold:
  - このときに、浮遊大陸プルヴァマにある、中立国、ビュエルバが、ある情報筋から、バッシュ将軍の処刑と、前王女、アーシェの自害を発表。: moraic_nasal ン phones=[N] ratio=1.1658 duration=0.16 decision=too_long warnings=
  - 時間領域と、空間領域で共通する処理手法は、フィルタリングによる、入力信号の強化である。: moraic_nasal ン phones=[N] ratio=1.2654 duration=0.17 decision=too_long warnings=
  - ただし、ギャンブル依存症の入院治療を行っている病院は、わずかである。: moraic_nasal ン phones=[w] ratio=1.0128 duration=0.15 decision=ok warnings=
  - ファンタジーゲーム: moraic_nasal ン phones=[] ratio=1.0005 duration=0.1557 decision=uncertain warnings=
  - 婚約指輪: moraic_nasal ン phones=[] ratio=1.0001 duration=0.2343 decision=uncertain warnings=
- mapping warning examples:

## yoon
- shortest:
  - このときに、浮遊大陸プルヴァマにある、中立国、ビュエルバが、ある情報筋から、バッシュ将軍の処刑と、前王女、アーシェの自害を発表。: yoon ピョ phones=[o] ratio=0.51 duration=0.07 decision=uncertain warnings=
  - シャンチーの専業プロは、チームから支払われる給料と、対局費を、主な収入としている。: yoon ニュ phones=[u u] ratio=0.5468 duration=0.08 decision=uncertain warnings=
  - シャンチーの専業プロは、チームから支払われる給料と、対局費を、主な収入としている。: yoon シュ phones=[u u] ratio=0.7519 duration=0.11 decision=uncertain warnings=
  - ただし、ギャンブル依存症の入院治療を行っている病院は、わずかである。: yoon ビョ phones=[o o] ratio=0.8777 duration=0.13 decision=uncertain warnings=
  - シャンチーの専業プロは、チームから支払われる給料と、対局費を、主な収入としている。: yoon リョ phones=[ry o] ratio=0.8886 duration=0.13 decision=uncertain warnings=
- longest:
  - ただし、ギャンブル依存症の入院治療を行っている病院は、わずかである。: yoon リョ phones=[ry o] ratio=3.8485 duration=0.57 decision=uncertain warnings=
  - 町域にあった、三根山藩は、長岡藩に、米百俵を送ったことで有名。: yoon ピョ phones=[py o] ratio=3.1002 duration=0.4 decision=uncertain warnings=
  - この、ニューサウスウェールズ代表チームが、ワラビーズの中核となって行く。: yoon チュ phones=[ch u] ratio=2.2939 duration=0.31 decision=uncertain warnings=
  - また、東寺のように、五大明王と呼ばれる、主要な明王の中央に配されることも多い。: yoon チュ phones=[ch u] ratio=2.2751 duration=0.32 decision=uncertain warnings=
  - 町域にあった、三根山藩は、長岡藩に、米百俵を送ったことで有名。: yoon ヒャ phones=[hy a] ratio=2.1701 duration=0.28 decision=uncertain warnings=
- near low threshold:
  - ただし、ギャンブル依存症の入院治療を行っている病院は、わずかである。: yoon ビョ phones=[o o] ratio=0.8777 duration=0.13 decision=uncertain warnings=
  - シャンチーの専業プロは、チームから支払われる給料と、対局費を、主な収入としている。: yoon リョ phones=[ry o] ratio=0.8886 duration=0.13 decision=uncertain warnings=
  - 時間領域と、空間領域で共通する処理手法は、フィルタリングによる、入力信号の強化である。: yoon リョ phones=[ry o] ratio=0.8933 duration=0.12 decision=uncertain warnings=
  - ツュレンハルト領は、ヴュルテンベルク領に編入された。: yoon リョ phones=[ry o] ratio=0.9043 duration=0.12 decision=uncertain warnings=
  - シャンチーの専業プロは、チームから支払われる給料と、対局費を、主な収入としている。: yoon キョ phones=[ky o] ratio=0.9569 duration=0.14 decision=uncertain warnings=
- near high threshold:
  - また、東寺のように、五大明王と呼ばれる、主要な明王の中央に配されることも多い。: yoon チュ phones=[ch u] ratio=2.2751 duration=0.32 decision=uncertain warnings=
  - この、ニューサウスウェールズ代表チームが、ワラビーズの中核となって行く。: yoon チュ phones=[ch u] ratio=2.2939 duration=0.31 decision=uncertain warnings=
  - 町域にあった、三根山藩は、長岡藩に、米百俵を送ったことで有名。: yoon ヒャ phones=[hy a] ratio=2.1701 duration=0.28 decision=uncertain warnings=
  - ニューイングランド風は、牛乳をベースとした、白いクリームスープであり、ボストンクラムチャウダーとも呼ばれる。: yoon ニュ phones=[ny u] ratio=2.0935 duration=0.31 decision=uncertain warnings=
  - コンピュータゲームのメーカーや、業界団体などに関連する人物のカテゴリ。: yoon ギョ phones=[gy o] ratio=1.9763 duration=0.25 decision=uncertain warnings=
- mapping warning examples:

