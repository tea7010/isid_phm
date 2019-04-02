# ISIDのデータコンペ用レポジトリ・管理 

## submit履歴
No|概要|validation結果|sunbit_score
-|-|-|-
1|trainの各エンジンの最終フライト数の平均を、予測フライト数とする|32.3|31
2|回帰アプローチ|35.16|37.27

## ToDo
1. 検証（validation)データの作成
1. アプローチ・特徴量・モデリングのPDCA <- いまここ

## 関連URL
サイト|URL
-|-
メインページ|https://industrial-big-data.io/phmc2019-datachallenge/
データDLページ|https://industrial-big-data.io/phmc2019-datachallenge_dl/
元ネタ？の論文|https://scholarcommons.usf.edu/cgi/viewcontent.cgi?article=7252&context=etd
関連論文|https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/publications/#turbofan
RULの論文|https://www.researchgate.net/publication/271921403_Remaining_useful_life_prediction_using_prognostic_methodology_based_on_logical_analysis_of_data_and_Kaplan-Meier_estimation
有名そうな論文|http://or.nsfc.gov.cn/bitstream/00001903-5/93202/1/1000004637516.pdf

## Getting started
最初にやることは、
1. condaのenvからpythonの環境を作成
1. terminalで環境を開き、好きなコードを実行

### 1. condaのenvからpythonの環境を作成
anacondaは入ってなかったら多分最新のやつinstallしとけば大丈夫です。

`isid_phm\conda_env`へ移動して、次を実行
```
conda env create -f env.yml
```

各種パッケージのinstallが出来たら、windowsなら
```
activate isid
```
で環境が立ち上がるはずです。
* 分からなかったら誰かにきくかドキュメントを参照 https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
* 多分まだ必要なものが入ってないかもしれないので、適宜ymlもアップデートしていきましょう

### 2. 端末でコードを実行
`isid_phm`下で以下を実行すると
```
python src/try_model.py
```
こんな流れを一通り実行してくれる。
1. データDL
1. データ前処理
1. モデル学習
1. 内部評価値の表示
1. 提出用ファイルを出力
