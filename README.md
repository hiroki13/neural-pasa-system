# Neural Models for Japanese Predicate Argument Structure Analysis

### Data Format: NAIST Text Corpus 1.5
```
\# S-ID:950112002-001 KNP:98/05/19 MOD:98/07/14
* 0 1D
地球 ちきゅう * 名詞 普通名詞 * * _
から から * 助詞 格助詞 * * _
* 1 2D
二千万 にせんまん * 名詞 数詞 * * _
光年 こうねん * 接尾辞 名詞性名詞助数辞 * * _
かなた かなた * 名詞 普通名詞 * * id="1"
に に * 助詞 格助詞 * * _
* 2 3D
ある ある * 動詞 * 子音動詞ラ行 基本形 alt="active"/ga="2"/ga_type="dep"/ni="1"/
ni_type="dep"/type="pred"
```

### Dependencies
To run the code, you need the following extra packages installed:
  - Numpy and Theano

#### Example Command
	`python -m pasa.api.main -mode train --train_data /path/to/data --dev_data /path/to/data --test_data /path/to/data --vocab_cut_off 1 --save 1 --model grid --layers 2 --batch_size 2 --reg 0.0005`

