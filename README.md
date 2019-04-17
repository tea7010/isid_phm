# ISIDのデータコンペ用レポジトリ・管理 

優勝目指せ！が第一目標かもしれませんが、BID全体の機械学習知識/経験の底上げが浅田さんの意図だと思ってます。
* 現段階ではデータの理解度等に差があると思うので、3人で一つのモデルを良くしていくのは、なかなか難しいです。
* そこで、個人個人でまずデータを眺める・単純でもいいので、モデリングしてみることをおすすめしたいです。
* その後、チームで同じモデルを改善していくなり、違うアプローチを分担して試していったり考えましょう。

## 内部評価・submitスコアボード
[docs/submit_history.md](./docs/submit_history.md)

## フォルダ構成
```
docs: スコア履歴、カラムのコピペ用ファイルなど
src
 |- {team_member}_lab: 各自の実験フォルダ、ここに各自のチャレンジを記録
 |- libs: 共通で使えそうなモジュールを置く
 |- notebook: チュートリアルや周知したい実験結果などを置く
```
実行rootはこのレポジトリの第一階層、`./isid_phm`を想定してます。

## Getting started
1. notebookフォルダにhowto_EDAなど簡単なチュートリアルを用意しました。

1. github上(ブラウザ）で閲覧できるので、一回読んでみる
![](./docs/img/2019-04-10-16-45-36.png)

1. 自分でjupyter notebook/labを立ち上げて、そのファイルを実行してみる（jupyter notebook/labで開く）

1. 自分で簡単な予測モデルを作ってみたりしてみる


## 関連URL
サイト|URL
-|-
メインページ|https://industrial-big-data.io/phmc2019-datachallenge/
データDLページ|https://industrial-big-data.io/phmc2019-datachallenge_dl/
元ネタ？の論文|https://scholarcommons.usf.edu/cgi/viewcontent.cgi?article=7252&context=etd
関連論文|https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/publications/#turbofan
RULの論文|https://www.researchgate.net/publication/271921403_Remaining_useful_life_prediction_using_prognostic_methodology_based_on_logical_analysis_of_data_and_Kaplan-Meier_estimation
有名そうな論文|http://or.nsfc.gov.cn/bitstream/00001903-5/93202/1/1000004637516.pdf