<div class="dialogue">
  <div class="talk metan">
    <div class="person">
      <span class="avatar">
        <img
          src="pics\四国めたん\22_実況_announcing.png"
          alt=""
          style="transform: scaleX(-1);"
        >
      </span>
      <span>めたん</span>
    </div>
    <div class="bubble">
      <p>皆様、こんにちは。<strong>ゆる音声合成ワークショップへようこそ！</strong></p>
      <p>わたくしこと、四国メタンがこのワークショップのチェアーを担当させていただきます。</p>
    </div>
  </div>

  <div class="talk zundamon">
    <div class="person">
      <span class="avatar">
        <img src="pics\ずんだもん\04_嬉しい_happy.png" alt="">
      </span>
      <span>ずんだもん</span>
    </div>
    <div class="bubble">
      <p>ずんだもんだ。音声合成について全然詳しくないが、よろしくなのだ。</p>
      <p>初めて学会に来たのだ、めっちゃ人がおおいのだ</p>
    </div>
  </div>

  <div class="talk tsumugi">
    <div class="person">
      <span class="avatar">
        <img src="pics\春日部つむぎ\04_キメ顔_posing.png" alt="">
      </span>
      <span>つむぎ</span>
    </div>
    <div class="bubble">
      <p>春日部つむぎです。</p>
      <p>ワークショップのような気軽な雰囲気の方が好きなので、キーノートをさぼってきた。</p>
    </div>
  </div>

  <div class="talk metan">
    <div class="person">
      <span class="avatar">
        <img
          src="pics\四国めたん\22_実況_announcing.png"
          alt=""
          style="transform: scaleX(-1);"
        >
      </span>
      <span>めたん</span>
    </div>
    <div class="bubble">
      <p>それでは、本日の本題、
        <strong>音声言語モデル</strong>に入ろうと思います。</p>
    </div>
  </div>
</div>

<style>
.dialogue {
  --avatar-size: clamp(2.75rem, 10vw, 5.625rem);
  --gap: clamp(.75rem, 2vw, 1.5rem);
  color: #34343d;
  font-family: "Hiragino Kaku Gothic ProN", "Yu Gothic", Meiryo, sans-serif;
  margin-block: 2rem;
  line-height: 1.85;
}

.dialogue,
.dialogue * {
  box-sizing: border-box;
}

/* Each character supplies their own theme. */
.dialogue .metan {
  --bubble-color: #fcecf4;
  --avatar-color: #f4d4e5;
  --accent: #933d70;
  --highlight: #e9a9cc66;
}

.dialogue .zundamon {
  --bubble-color: #edf7e7;
  --avatar-color: #d7ebc7;
  --accent: #456d30;
  --highlight: #acd18466;
}

.dialogue .tsumugi {
  --bubble-color: #fff1e3;
  --avatar-color: #f8ddbd;
  --accent: #99551d;
  --highlight: #f4bc7666;
}

.dialogue .talk {
  display: flex;
  align-items: flex-start;
  gap: var(--gap);
  margin-block: clamp(1.25rem, 3vw, 2rem);
}

/* めたん stays left; both other characters stay right. */
.dialogue .talk.zundamon,
.dialogue .talk.tsumugi {
  flex-direction: row-reverse;
}

.dialogue .person {
  flex: 0 0 var(--avatar-size);
  min-inline-size: 0;
  color: var(--accent);
  text-align: center;
  font-size: clamp(.65rem, 1.5vw, .8rem);
  font-weight: 700;
  overflow-wrap: anywhere;
}

.dialogue .avatar {
  display: grid;
  place-items: center;
  inline-size: 100%;
  aspect-ratio: 1;
  margin-block-end: .35rem;
  overflow: hidden;
  border-radius: 50%;
  background: var(--avatar-color);
}

.dialogue .avatar img {
  display: block;
  inline-size: 100%;
  block-size: 100%;
  object-fit: contain;
}

.dialogue .bubble {
  position: relative;
  flex: 0 1 auto;
  min-inline-size: 0;
  padding: clamp(.85rem, 2.5vw, 1.5rem);
  border-radius: 4px 22px 22px 22px;
  background: var(--bubble-color);
  box-shadow: 0 3px 0 #263e3b0a;
  overflow-wrap: anywhere;
}

.dialogue .bubble::before {
  content: "";
  position: absolute;
  top: 1rem;
  left: -.625rem;
  border-block: .5rem solid transparent;
  border-right: .6875rem solid var(--bubble-color);
}

.dialogue :is(.zundamon, .tsumugi) .bubble {
  border-radius: 22px 4px 22px 22px;
}

.dialogue :is(.zundamon, .tsumugi) .bubble::before {
  left: auto;
  right: -.625rem;
  border-right: 0;
  border-left: .6875rem solid var(--bubble-color);
}

.dialogue .bubble p {
  margin: 0;
}

.dialogue .bubble p + p {
  margin-block-start: .625rem;
}

.dialogue .bubble strong {
  color: var(--accent);
  background: linear-gradient(
    transparent 65%,
    var(--highlight) 65%
  );
}

.dialogue .bubble code {
  padding: .125rem .3rem;
  border-radius: 5px;
  background: #ffffffa6;
  color: var(--accent);
}

.dialogue .bubble pre {
  max-inline-size: 100%;
  overflow-x: auto;
}
</style>