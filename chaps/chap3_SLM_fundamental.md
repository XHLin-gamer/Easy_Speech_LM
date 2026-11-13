<!-- tokenizer and RVQ -> Fast & Slow AR

Mel Specturm

prompt pack

Slow AR -> Repetition Aware Sampling (Additional Sampling Strategy)

Fast AR -> visualization

Insturctional Inference -->


めたん：これでは、一番肝心な部分、トークンから更なるトークンを予測する仕組み、言語モデルの言語モデルの部分について、話をします。

つむぎ：確か、それはNeural　Codecを使って、音声をトークンに変換して、言語モデルのような普通の推理をするですよね。

めたん：大体的なイメージは確かにこんな風にはなるですが、言語と全く同じように、音声を処理するモデルは少ないです。

ずんだもん：それは何でですか？

めたん：前に紹介したRVQの概念はまだ覚えていますか？一節の音声に対応するトークンは、一個になるかもしれないし、何個が同時に必要かもしれない。そこで、一回きり一つのトークンを生成して、それだけで音声を出力するやり方があれば、他のモデルを介して、一回に数個のトークンを生成するモデルも存在しています。前者は普通の言語モデルと変わらないが、後者は言語モデルの構造と推理時の流れを少しだけ改造する必要があります。

つむぎ：ほかのモデルを介するのは変なやり方だな。なぜ一つのモデルにまとまらないか？

めたん：それは、実際推理するときの所要時間を削減するためと、RVQの性質上、段数が多いほど最終的に結果にもたらす影響が有限になるので、そんなに深いモデルを使わなくとも普通にいい性能を出してくれるからです。これからは、いくつかの実験を通して、このやり方を紹介したいなっと思います。


## RVQの場合のプロンプトを見てみよう

めたん：例えばRVQ使わない場合、実際にSLMを使うと、以下のなプロンプトが使われる。

```Bash
<|im_start|>system
convert the provided text to speech reference to the following:

Text:
<|speaker:0|>また、東寺のように、五大明王と呼ばれる、主要な明王の中央に配されることも多い。

Speech:
{VQPart: reference.codes, shape=(1, 149)}<|im_end|>　# <- ここは1段階の、149フレームの音声トークンのテンセルを意味する

<|im_start|>user
<|speaker:0|>吾輩は猫である。名前はまだない。<|im_end|>

<|im_start|>assistant
<|voice|>
```

ずんだもん：結構直観てきなやつだな。マジて、お手本の音声のテキストとコードをシステムプロダクトに入れて、モデルを似たようなコードを続いて生成するそのままじゃないか。

めたん：じゃもし、RVQを使って、十段階の音声トークンに拡張すれば

```Bash
<|im_start|>system
convert the provided text to speech reference to the following:

Text:
<|speaker:0|>また、東寺のように、五大明王と呼ばれる、主要な明王の中央に配されることも多い。

Speech:
{VQPart: reference.codes, shape=(10, 149)}<|im_end|>　# <- ここは10段階になります

<|im_start|>user
<|speaker:0|>吾輩は猫である。名前はまだない。<|im_end|>

<|im_start|>assistant
<|voice|>
```

つむぎ：待って、このプロンプトだとそのまま使えないよ。このプロンプトの形がおかしいわ、音声のコードのところ、でこぼこになっているじゃない。文字のプロンプトは基本、一行で完結できるに対して、ここの音声tokenが十行必要。だからそのまま埋め込みできるはずがないです。これを一列に整列するつもりなのか？

## トークンを一列に並べると何が困るのか

めたん：それは、一列に整列すると、いろいろ不都合が生じるですよ。

ずんだもん：でも、普通の言語モデルみたいに一個ずつ予測するほうが、シンプルでいいと思うのだ。

めたん：では、計算してみましょう。前の章で聞いた Codec は、1秒の音声を12.5個の区間（フレーム）に分けて、1フレームごとに16段のコードブックを使っていました。
<!-- TODO: 使用する tokenizer（Qwen3-TTS-Tokenizer-12Hz）のフレームレート・段数を再確認 -->

つむぎ：1秒あたり 12.5 × 16 = 200 トークンか。WaveNet の1万6千個よりはずっとましだけど、テキストに比べるとまだ多いよね。

めたん：そうです。16段のトークンを全部一列に並べて、一つの大きなモデルで順番に予測すると、10秒の音声で2000回も大きなモデルを動かすことになります。

![alt text](assets/chap3_SLM_fundamental/images/flatten.png)
<!-- TODO: 図：[フレーム1: cb0 cb1 ... cb15][フレーム2: cb0 cb1 ... cb15] と一列に並べた図 -->

ずんだもん：さっきの WaveNet みたいに、待たされるのはいやなのだ。

めたん：しかも、16個のトークンは同じ時刻の音を表しているので、性質がバラバラです。cb0 は「何を話しているか」、後ろの段に行くほど細かい音色の補正。役割の違うものを同じ列に並べて、同じように扱うのは、ちょっともったいないんです。

つむぎ：なるほど。「時間方向のつながり」と「同じ時刻の中での段のつながり」は、別物ってことね。

めたん：その通りです。そこで、このコードブックたちを足し算して、融合させます。出力するは、別のモデルを介して、RVQの多重構造を復活させる戦法よ。
 
## Semantic Model と Accoustic Model

![alt text](assets/chap3_SLM_fundamental/images/slow_fast_ar.png)
<!-- TODO: 図：横方向（時間）に Slow AR、各フレームの縦方向（段）に Fast AR が走る二次元の図 -->

めたん：図の横方向が時間、縦方向が RVQ の段です。

- **Slow AR**：大きな言語モデル。**1フレームに1回だけ**動いて、そのフレームの cb0 を予測する。
- **Fast AR**：小さなモデル。Slow AR の結果を受け取って、同じフレームの cb1〜cb15 を**下から順番に**予測する。

ずんだもん：ゆっくりな方と速い方なのだ？

つむぎ：名前の由来は、動く頻度だよね。Slow AR は時間方向に1フレームずつ「ゆっくり」進んで、Fast AR はその1フレームの間に15回も「せかせか」動く。

めたん：ええ。Qwen3-TTS では、Slow AR にあたる部分を **Talker**、Fast AR にあたる部分を **Code Predictor** と呼んでいます。
<!-- TODO: Qwen3-TTS の論文・実装上の正式名称（talker / code_predictor / MTP module）を確認 -->

つむぎ：じゃあ、さっき言ってた「ほかのモデルを介する」って、この Code Predictor のことなんだ。

めたん：そうです。前の章の RVQ の表を思い出してください。

| 段 | 残差 | 量子化 | 合計 |
|:---:|---:|---:|---:|
| 1 | 3.14159 | 3 | 3 |
| 2 | 0.14159 | 0.1 | 3.1 |
| 3 | 0.04159 | 0.04 | 3.14 |

めたん：一番大事な「3」を決めるのは難しい仕事なので、大きなモデルにじっくり考えてもらう。でも「0.1」や「0.04」は、「3」さえ決まっていれば、もう選択肢がだいぶ絞られています。

ずんだもん：小さい桁は、小さいモデルで十分なのだな！

めたん：その通り。しかも、後ろの段ほど音全体に与える影響が小さいので、多少間違えても、聞いた感じはほとんど変わりません。

### コードで見る流れ

つむぎ：コードにするとどうなるの？

めたん：細かいところを省くと、こんな流れになります。

```python
NUM_CODEBOOKS = 16

codes_history = []
for t in range(max_frames):
    # --- Slow AR：1フレームに1回 ---
    h = talker(inputs)                      # 大きいモデル
    c0 = sample(talker.head(h[-1]))         # cb0 を予測
    if c0 == EOS:
        break

    # --- Fast AR：1フレームに15回 ---
    frame = [c0]
    for k in range(1, NUM_CODEBOOKS):
        logits = code_predictor(h[-1], frame)   # 小さいモデル
        frame.append(sample(logits))
    codes_history.append(frame)

    # 16段ぶんの埋め込みを「足し合わせて」、次のフレームの入力にする
    next_input = sum(codebook_embed[k](frame[k]) for k in range(NUM_CODEBOOKS))
    inputs = torch.cat([inputs, next_input[None]], dim=0)

audio = codec.decode(codes_history)
```

ずんだもん：最後の「足し合わせる」ところが気になるのだ。16個のトークンを並べないで、一個にまとめちゃうのだ？

つむぎ：あ、これ RVQ と同じ発想じゃない？　RVQ でも、各段のベクトルを全部足して、一つの潜在ベクトルにしてたよね。

めたん：鋭いですね。Slow AR から見れば、**1フレームはあくまで1個の入力**です。だから、Slow AR が扱う列の長さは、1秒あたり12.5個で済むんです。

つむぎ：テキストのトークンと同じくらいの密度だ。それなら、普通の言語モデルの仕組みがそのまま使えそう。

めたん：実際、Qwen3-TTS の Talker は、Qwen3 という言語モデルをベースにしています。テキストを読む能力を、そのまま音声の生成に活かせるわけです。

### 実験：Fast AR は何をしているのか

ずんだもん：Fast AR が本当に細かいところしか担当してないのか、確かめてみたいのだ。

めたん：いい提案です。Slow AR が出した cb0 の列は固定したまま、Fast AR だけ乱数のシードを変えて、何回か生成し直してみましょう。

<!-- TODO: 実験スクリプトのリンク -->

| | cb0〜cb15 のヒートマップ | 合成音声 |
|---|---|---|
| シード 0 | ![alt text](assets/chap3_SLM_fundamental/images/fast_ar_seed0.png) | <audio controls src="assets/chap3_SLM_fundamental/audio/fast_ar_seed0.wav"></audio> |
| シード 1 | ![alt text](assets/chap3_SLM_fundamental/images/fast_ar_seed1.png) | <audio controls src="assets/chap3_SLM_fundamental/audio/fast_ar_seed1.wav"></audio> |
| シード 2 | ![alt text](assets/chap3_SLM_fundamental/images/fast_ar_seed2.png) | <audio controls src="assets/chap3_SLM_fundamental/audio/fast_ar_seed2.wav"></audio> |

<!-- TODO: 縦軸=段（cb0 が一番下）、横軸=フレーム。cb0 の行は全シードで同一、上の段ほど色がバラつく図を想定 -->

つむぎ：一番下の cb0 の行は全部同じで、上の段に行くほどバラバラになってる。

ずんだもん：でも、聞いてみると、ほとんど同じなのだ！

めたん：トークンの番号はかなり違うのに、音はほぼ同じ。これが「後ろの段ほど影響が小さい」ということの実感です。逆に、cb0 を変えてしまうと……

<audio controls src="assets/chap3_SLM_fundamental/audio/slow_ar_resample.wav"></audio>
<!-- TODO: cb0 だけ再サンプリングした音声 -->

ずんだもん：話し方のリズムやイントネーションが、ガラッと変わったのだ！

めたん：つまり、「どう話すか」の大部分は Slow AR が決めていて、Fast AR はそれを仕上げる職人さん、というわけです。

## プロンプトの組み立て

つむぎ：仕組みはわかったけど、テキストはどこから入るの？　それに、誰の声でしゃべるかはどうやって決めるの？

めたん：そこが、プロンプトの組み立て方です。言語モデルのプロンプトは、テキストを並べるだけでしたよね。音声言語モデルでは、いくつかの種類の情報を一つの列に詰め込みます。

![alt text](assets/chap3_SLM_fundamental/images/prompt_pack.png)
<!-- TODO: 図：[話者埋め込み][参照音声のテキスト + 読み上げたいテキスト][参照音声のトークン] → [生成する音声トークン] -->

めたん：声をまねる（ボイスクローン）場合、Qwen3-TTS のプロンプトにはだいたい次のものが入ります。

1. **話者埋め込み**：参照音声から計算した、「声色」を表す1本のベクトル
2. **テキスト**：参照音声で話している内容と、これから読み上げたい内容
3. **参照音声のトークン**：参照音声を Codec でトークンにしたもの

ずんだもん：参照音声を、2回も別の形で入れているのだ？

めたん：そうです。それぞれ得意なことが違うんです。話者埋め込みは「どんな声か」を大まかに、参照音声のトークンは「どんな話し方か」まで細かく伝えます。

つむぎ：言語モデルでいう **Few-shot** プロンプトみたいだね。「この人はこういうふうに話します。では続きをどうぞ」って、お手本を見せてから続きを書かせる感じ。

めたん：まさにそれです。参照音声のテキストとトークンが対になっているので、モデルは「この文字はこの人だとこう発音する」という対応をその場で読み取って、続きを同じ調子で話してくれます。

### メルスペクトログラム

ずんだもん：話者埋め込みは、どうやって計算するのだ？　波形のまま入れるのだ？

めたん：波形のままだと長すぎるのは、もうご存じですよね。そこで、**メルスペクトログラム**という形に変換してから、話者を見分けるための小さなモデル（話者エンコーダー）に入れます。

![alt text](assets/chap3_SLM_fundamental/images/mel.png)
<!-- TODO: 波形 → スペクトログラム → メルスペクトログラムの比較図 -->

つむぎ：横が時間、縦が周波数、色が音の強さ、っていうグラフだよね。

めたん：ええ。音を短い時間ごとに区切って、「どの高さの音が、どれくらいの強さで含まれているか」を調べたものがスペクトログラムです。そこに、人間の耳の感じ方に合わせて周波数の目盛りを付け直したものが、メルスペクトログラムです。

ずんだもん：人間の耳に合わせるのだ？

めたん：前の章で、「人間の感覚は意外と大ざっぱ」という話をしましたね。人間の耳は、低い音の違いには敏感ですが、高い音の違いにはかなり鈍いんです。100Hz と 200Hz の違いはすぐわかっても、10000Hz と 10100Hz の違いはほとんどわかりません。

つむぎ：だから低い音は細かく、高い音はざっくりまとめちゃうってことか。これも、聞こえにくい情報を捨てる工夫なんだね。

```python
import torchaudio

wav, sr = torchaudio.load("reference.wav")
mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=sr,
    n_fft=1024,       # 何サンプルずつ周波数を調べるか
    hop_length=256,   # 何サンプルずつずらすか
    n_mels=80,        # メル目盛りで何本に分けるか
)(wav)
print(wav.shape, mel.shape)
# 1秒・24kHz なら (1, 24000) → (1, 80, 94) くらい
```
<!-- TODO: Qwen3-TTS の話者エンコーダーが実際に使うメルの設定値に合わせる -->

ずんだもん：2万4千個が、80本 × 94フレームになったのだ。

めたん：話者エンコーダーはこれを読んで、最後に時間方向を平均して、1本のベクトルにまとめます。時間の情報を捨てるので、「何を話しているか」は消えて、「誰の声か」だけが残る、という仕組みです。

### 実験：話者埋め込みだけ vs お手本付き

めたん：では、プロンプトの中身を変えると結果がどう変わるか、試してみましょう。

```python
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")

# 話者埋め込みだけを使う
prompt_xvec = model.create_voice_clone_prompt(
    ref_audio="reference.wav", ref_text=REF_TEXT, x_vector_only_mode=True)

# 話者埋め込み + 参照音声のトークン（お手本付き）
prompt_icl = model.create_voice_clone_prompt(
    ref_audio="reference.wav", ref_text=REF_TEXT, x_vector_only_mode=False)

for name, prompt in [("xvec", prompt_xvec), ("icl", prompt_icl)]:
    wavs, sr = model.generate_voice_clone(
        text=TARGET_TEXT, language="Japanese", voice_clone_prompt=prompt)
```
<!-- TODO: qwen_tts の実際の API・引数名を確認 -->

| プロンプト | 合成音声 |
|---|---|
| 参照音声（元の声） | <audio controls src="audio/VOICEACTRESS100_001.wav"></audio> |
| 話者埋め込みだけ | <audio controls src="assets/chap3_SLM_fundamental/audio/clone_xvec.wav"></audio> |
| お手本付き | <audio controls src="assets/chap3_SLM_fundamental/audio/clone_icl.wav"></audio> |

ずんだもん：どっちもつくよみちゃんっぽいけど、お手本付きのほうが、しゃべり方まで似てるのだ！

つむぎ：話者埋め込みだけだと、声色は近いけど、話し方は「平均的な人」っぽくなるんだね。

めたん：その代わり、お手本付きは参照音声のテキストを正しく用意する必要があり、プロンプトも長くなります。用途に合わせて使い分けるといいですね。

## サンプリングの工夫

つむぎ：生成のところで `sample()` って書いてたけど、ここは普通の言語モデルと同じ？　温度とか Top-p とか。

めたん：基本は同じです。第1章で、「予測結果の中から一つを選ぶ」と言いましたよね。この「選び方」がサンプリングです。

- **貪欲法**：一番確率の高いものを毎回選ぶ
- **温度**：確率の分布を尖らせたり、なだらかにしたりする
- **Top-k / Top-p**：確率の低い候補を最初から除外してから選ぶ

ずんだもん：一番確率の高いのを選べば、一番いい声になるんじゃないのだ？

めたん：そう思いますよね。でも、音声で貪欲法を使うと、こうなることがあります。

<audio controls src="assets/chap3_SLM_fundamental/audio/greedy_loop.wav"></audio>
<!-- TODO: 貪欲法で同じ音が繰り返したり、無音が続いたりする例 -->

ずんだもん：同じところで、ずっと「あー」って伸びてるのだ！　壊れたのだ！

めたん：音声は、隣り合うフレームがとても似ています。だから「直前と同じトークン」の確率がいつも高くなりやすく、一度ループに入ると抜け出せなくなるんです。

つむぎ：テキストの LLM でも、同じ文を何度も繰り返しちゃうことがあるよね。それの、もっとひどいバージョンか。

### Repetition Aware Sampling

めたん：そこで、VALL-E 2 という研究で提案されたのが、**Repetition Aware Sampling（繰り返しを意識したサンプリング）**です。考え方はとてもシンプルです。

1. まず普通に Top-p などでトークンを選ぶ
2. 選んだトークンが、直近の数フレームの中で何回出てきたかを数える
3. 出てきすぎていたら、「ループに入りかけている」と判断して、元の確率分布から選び直す

```python
def repetition_aware_sampling(logits, history, top_p=0.8, window=10, threshold=0.1):
    token = top_p_sampling(logits, top_p)

    recent = history[-window:]
    if len(recent) > 0 and recent.count(token) / window > threshold:
        # 繰り返しすぎ → 候補を絞らずに、確率どおりにランダムに選び直す
        token = torch.multinomial(logits.softmax(-1), 1).item()

    return token
```
<!-- TODO: 実際に使う window / threshold の値を、Qwen3-TTS の設定に合わせる -->

ずんだもん：普段はちゃんと選んで、怪しいときだけサイコロを振り直すのだな。

めたん：ええ。普段は品質を保ちつつ、ループに入りかけたときだけ、そこから抜け出すきっかけを作るんです。

つむぎ：これって、Slow AR と Fast AR のどっちに使うの？

めたん：主に Slow AR の cb0 です。話の流れを決めるのは cb0 で、ループに陥るのも cb0 の列だからです。Fast AR のほうは、さっきの実験のとおり多少ぶれても問題ないので、普通のサンプリングで十分です。

<audio controls src="assets/chap3_SLM_fundamental/audio/ras.wav"></audio>
<!-- TODO: 同じ条件で Repetition Aware Sampling を使った結果 -->

ずんだもん：今度は最後まで、ちゃんとしゃべれたのだ！

## 指示による推論

つむぎ：最後にもう一つ。最近の TTS って、「もっと元気に」とか「ささやくように」とか、言葉で指示できるのもあるよね？

めたん：ええ。Qwen3-TTS にも、そういう使い方ができるモデルがあります。プロンプトの中に、テキストで書いた**指示文**を一緒に入れるんです。

```python
model = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")

wavs, sr = model.generate_custom_voice(
    text="今日はワークショップに来てくれて、ありがとう！",
    language="Japanese",
    speaker=SPEAKER,
    instruct="とても嬉しそうに、少し早口で話してください。",
)
```
<!-- TODO: API・話者名を確認 -->

| 指示 | 合成音声 |
|---|---|
| （指示なし） | <audio controls src="assets/chap3_SLM_fundamental/audio/instruct_none.wav"></audio> |
| とても嬉しそうに、少し早口で | <audio controls src="assets/chap3_SLM_fundamental/audio/instruct_happy.wav"></audio> |
| 眠そうに、ささやくように | <audio controls src="assets/chap3_SLM_fundamental/audio/instruct_whisper.wav"></audio> |

ずんだもん：同じ文なのに、全然違う感じになるのだ！

つむぎ：Slow AR が言語モデルだから、指示の文章も「読める」ってことね。

めたん：その通りです。もともと文章を理解する能力を持った言語モデルを土台にしているからこそ、こういう使い方ができるんです。さらに、話者を指定する代わりに「低くて落ち着いた声の中年男性」のように、**声そのものを言葉で設計する**モデル（VoiceDesign）もあります。

ずんだもん：もう何でもありなのだ……。

めたん：ただし、どんな指示でもきちんと従ってくれるわけではありません。学習データにあまりない指示や、矛盾した指示は、うまく反映されないこともあります。

つむぎ：そこを良くするのが、第1章で言ってた SFT とか DPO の出番ってわけか。

めたん：鋭いですね。それは次の章でお話しします。

## まとめ

めたん：最後に、今回の内容を振り返りましょう。

- **Slow AR / Fast AR**：RVQ の全段を一列に並べると長くなりすぎるので、時間方向は大きなモデル（Slow AR）、段方向は小さなモデル（Fast AR）に分担させる。Slow AR は1フレームに1回、cb0 を予測するだけで済む。
- **Fast AR の役割**：後ろの段ほど音への影響が小さいので、小さなモデルで十分。トークンが多少変わっても、聞いた感じはほとんど変わらない。
- **プロンプトの組み立て**：話者埋め込み、テキスト、参照音声のトークンを一つの列に詰め込む。お手本を見せるほど、話し方まで似せられる。
- **メルスペクトログラム**：人間の耳の感じ方に合わせて周波数を並べ直した、音の表現。話者埋め込みの計算に使う。
- **Repetition Aware Sampling**：音声は同じトークンを繰り返しやすいので、繰り返しすぎたときだけ選び直して、ループを防ぐ。
- **指示による推論**：言語モデルを土台にしているので、言葉で話し方や声色を指示できる。

ずんだもん：トークンの作り方から、しゃべらせ方まで、一通りつながったのだ！

つむぎ：次は、このモデルを自分の好きな声やしゃべり方に合わせる方法だね。

めたん：ええ。次の章では、ファインチューニングについてお話しします。お楽しみに！

参考文献

- Qwen Team, "Qwen3-TTS Technical Report", 2026. <!-- TODO: URL -->
- https://github.com/QwenLM/Qwen3-TTS <!-- TODO: URL を確認 -->
- Chen et al., "VALL-E 2: Neural Codec Language Models are Human Parity Zero-Shot Text to Speech Synthesizers", 2024. https://arxiv.org/abs/2406.05370
- Lee et al., "Autoregressive Image Generation using Residual Quantization" (RQ-Transformer), 2022. https://arxiv.org/abs/2203.01941
- Défossez et al., "Moshi: a speech-text foundation model for real-time dialogue", 2024. https://arxiv.org/abs/2410.00037

# 音声言語モデルの構造

本節では、実際に音声言語モデルを動かして、何かを生成しながら、その構造を見てみようと思います。

使うモデルはこちら、「Fish Audio S2」となります。このモデルは特徴として、完全に自己回帰モデルで、Difussionを使わずに、高品質の音声を合成することができます。ちなみに、「Qwen3TTS」などのモデルは、同じ完全たる自己回帰モデルで、「Irodori-TTS」などのモデルは、ある程度Difussionを使うので言語モデルとの類似性がFishほどではない。

細かいディテールを省けて、大体的な流れを紹介するつもりです。

前の節で紹介した、Codecにして、音声言語モデルのTokenizerにもなるVQVAE