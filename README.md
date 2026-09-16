# Easy_Speech_LM
【C109】誰でも簡単に理解できる音声生成AI (工事中)


Table of Content

|   No. | Section                                    | Subsection                                       | Progress      | Reviewer |
| ----: | ------------------------------------------ | ------------------------------------------------ | ------------- | -------- |
| **1** | **Introduction**                           |                                                  |               |          |
|   1.1 |                                            | What Is a Speech Language Model?                 |                |                   |
|   1.2 |                                            | Why Speech Language Models Matter                |                |                   |
|   1.3 |                                            | Two Representative Categories: TTS & Full-Duplex |                |                   |
|   1.4 |                                            | Overview of Modern Speech-Language Architectures |                |                   |
| **2** | **Legal and Ethical Issues**               |                                                  |               |          |W
|   2.1 |                                            | 文化庁の議論・ガイドライン                         |                |                   |
|   2.2 |                                            | 著作権法と創作物の認定                             |                |                   |
|   2.3 |                                            | Training Data and Copyright                      |                |                   |W
| **3** | **How Modern Speech Language Models Work** |                                                  |               |          |
|   3.1 |                                            | Speech as a Continuous Signal                    |                |                   |
|   3.2 |                                            | From Waveforms to Discrete Tokens                |                |                   |
|   3.3 |                                            | Neural Codec + Language Model + Decoder          |                |                   |
|   3.4 |                                            | Text Tokens vs. Speech Tokens                    |                |                   |
| **4** | **Preliminaries**                          |                                                  |               |          |
|   4.1 |                                            | Conv&ResNet$SEANet                               |                |                   |
|   4.2 |                                            | Transformer                                      |                |                   |
|   4.3 |                                            | VAE                                              |                |                   |
|   4.4 |                                            | Autoregressive Modeling                          |                |                   |
| **5** | **Neural Audio Codec**                     |                                                  |               |          |
|   5.1 |                                            | VQ-VAE                                           |                |                   |
|   5.2 |                                            | Residual Vector Quantization (RVQ)               |                |                   |
|   5.3 |                                            | Encoder–Quantizer–Decoder                        |                |                   |
|   5.4 |                                            | Reconstruction Loss                              |                |                   |
|   5.5 |                                            | GAN / Adversarial Loss                           |                |                   |
|   5.6 |                                            | Speech Tokens                                    |                |                   |
| **6** | **TTS Model — Qwen3-TTS**                  |                                                  |               |          |
|   6.1 |                                            | Overall Architecture                             |                |                   |
|   6.2 |                                            | Backbone Network                                 |                |                   |
|   6.3 |                                            | Speech Token Generation                          |                |                   |
|   6.4 |                                            | Training                                         |                |                   |
|   6.5 |                                            | Fine-tuning                                      |                |                   |
|   6.6 |                                            | Inference                                        |                |                   |
| **7** | **Full-Duplex Model — Moshi**              |                                                  |               |          |
|   7.1 |                                            | Overall Architecture                             |                |                   |
|   7.2 |                                            | Streaming Speech Modeling                        |                |                   |
|   7.3 |                                            | Temporal Alignment                               |                |                   |
|   7.4 |                                            | Low-Latency Real-Time Generation                 |                |                   |
|   7.5 |                                            | Training                                         |                |                   |
|   7.6 |                                            | Fine-tuning                                      |                |                   |
|   7.7 |                                            | Duplex Conversation                              |                |                   |
| **8** | **Comparison and Future Directions**       |                                                  |               |          |
|   8.1 |                                            | TTS vs. Full-Duplex Models                       |                |                   |
|   8.2 |                                            | Latency                                          |                |                   |
|   8.3 |                                            | Speech Quality                                   |                |                   |
|   8.4 |                                            | Controllability                                  |                |                   |
|   8.5 |                                            | Remaining Challenges                             |                |                   |
| **9** | **Conclusion**                             |                                                  |               |          |
|   9.1 |                                            | Summary and Takeaways                            |                |                   |
