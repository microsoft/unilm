<!--# 跨任务、跨语言与跨模态的大规模自监督预训练-->
## [aka.ms/GeneralAI](https://aka.ms/GeneralAI)

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

# 人才招聘

我们正在招聘各级别优秀人才（包括全职全员研究员 FTE 与实习生）！如果您有兴趣与我们共同在基座大模型（Foundation Models，即大规模预训练模型）与通用人工智能（General AI）、自然语言处理（NLP）、机器翻译（MT）、语音处理（Speech）、文档智能（Document AI）以及多模态 AI 等领域开展前沿探索，欢迎将简历投递至 <a href="mailto:fuwei@microsoft.com" class="x-hidden-focus">fuwei@microsoft.com</a>。

# 基座架构 (Foundation Architecture)

### TorchScale - 基座架构算法库 ([代码库](https://github.com/microsoft/torchscale))

致力于研发面向基座模型与通用 AI 的下一代基础新架构，重点聚焦建模通用性与综合能力，以及训练过程的稳定性与计算效率。

> 稳定性 - [**DeepNet**](https://github.com/microsoft/unilm/tree/master/deepnet)：将 Transformer 成功扩展至 1,000 层及以上规模

> 通用性 - [**Foundation Transformers (Magneto)**](https://arxiv.org/abs/2210.06423)：迈向跨任务与跨模态（涵盖语言、视觉、语音及多模态）的真正通用化建模

> 泛化拓展能力 - [**Length-Extrapolatable**](https://arxiv.org/abs/2212.10554)：具备长度外推能力的 Transformer 架构

> 高效性与迁移能力 - [**X-MoE**](https://github.com/microsoft/unilm/tree/master/xmoe)：兼具高扩展性与微调能力的稀疏混合专家系统（MoE）

### 模型架构范式演进

> [**BitNet**](https://arxiv.org/abs/2310.11453)：面向大语言模型的 1-bit Transformer 架构

> [**RetNet**](https://arxiv.org/abs/2307.08621)：保留网络（Retentive Network）——大语言模型 Transformer 的下一代继承者

> [**LongNet**](https://arxiv.org/abs/2307.02486)：将 Transformer 序列建模能力扩展至 10 亿（1,000,000,000）Tokens

# 基座大模型 (Foundation Models)

### 多模态大语言模型 (MLLM) 演进脉络

> [**Kosmos-2.5**](https://github.com/microsoft/unilm/tree/master/kosmos-2.5)：**多模态读写通识大模型 (Multimodal Literate Model)**

> [**Kosmos-2**](https://github.com/microsoft/unilm/tree/master/kosmos-2)：**将多模态大语言模型与物理世界实体感知锚定 (Grounding)**

> [**Kosmos-1**](https://arxiv.org/abs/2302.14045)：**多模态大语言模型 (MLLM)**

> [**MetaLM**](https://github.com/microsoft/unilm/tree/master/metalm)：**语言模型作为通用的通用交互界面**

**大融合 (The Big Convergence)** - 跨越```任务```（预测型与生成型）、```语言```（覆盖 100+ 种语言）以及```模态```（文本、图像、音频、排版/格式 + 语言、视觉 + 语言、音频 + 语言等）的大规模自监督预训练体系。

<!--## 跨任务、跨语言与跨模态的大规模自监督预训练-->

### 语言与多语言大模型 (Language & Multilingual)
> [**UniLM**](https://github.com/microsoft/unilm/tree/master/unilm)：面向自然语言理解（NLU）与自然语言生成（NLG）的统一预训练模型

> [**InfoXLM/XLM-E**](https://github.com/microsoft/unilm/tree/master/infoxlm)：覆盖 100+ 种语言的多语言/跨语言预训练模型

> [**DeltaLM/mT6**](https://github.com/microsoft/unilm/tree/master/deltalm)：面向 100+ 语言的语言生成与机器翻译的 Encoder-Decoder 预训练模型

> [**MiniLM**](https://github.com/microsoft/unilm/tree/master/minilm)：面向语言理解与生成的轻量极速预训练小模型

> [**AdaLM**](https://github.com/microsoft/unilm/tree/master/adalm)：面向领域、语言与具体任务的预训练模型适配方法

> [**EdgeLM**](https://github.com/microsoft/unilm/tree/master/edgelm) (```NEW```)：部署于边缘计算/端侧设备的高效预训练模型

> [**SimLM**](https://github.com/microsoft/unilm/tree/master/simlm) (```NEW```)：面向语义相似度匹配的大规模自监督预训练模型

> [**E5**](https://github.com/microsoft/unilm/tree/master/e5) (```NEW```)：高质量通用文本向量嵌入模型 (Text Embeddings)

> [**MiniLLM**](https://arxiv.org/abs/2306.08543) (```NEW```)：面向大语言模型的高效知识蒸馏算法

### 计算机视觉 (Vision)
> [**BEiT**](https://github.com/microsoft/unilm/tree/master/beit)/[**BEiT-2**](https://github.com/microsoft/unilm/tree/master/beit2)：图像 Transformer 的生成式自监督预训练 / BERT 式视觉预训练

> [**DiT**](https://github.com/microsoft/unilm/tree/master/dit)：面向文档图像 Transformer 的自监督预训练模型

> [**TextDiffuser**](https://github.com/microsoft/unilm/tree/master/textdiffuser)/[**TextDiffuser-2**](https://github.com/microsoft/unilm/tree/master/textdiffuser-2) (```NEW```)：基于扩散模型（Diffusion Models）的文本绘制与图像生成

### 语音处理 (Speech)
> [**WavLM**](https://github.com/microsoft/unilm/tree/master/wavlm)：面向全栈全流程语音任务的自监督语音预训练模型

> [**VALL-E**](https://github.com/microsoft/unilm/tree/master/valle)：基于神经编解码器的大语言模型文本转语音（TTS）系统

### 多模态 (X + 语言)
> [**LayoutLM**](https://github.com/microsoft/unilm/tree/master/layoutlm)/[**LayoutLMv2**](https://github.com/microsoft/unilm/tree/master/layoutlmv2)/[**LayoutLMv3**](https://github.com/microsoft/unilm/tree/master/layoutlmv3)：多模态（文本 + 版面排版/格式 + 图像）**文档基座大模型**，赋能[文档智能 (Document AI)](https://www.microsoft.com/en-us/research/project/document-ai/)（如扫描件、PDF 文档理解等）

> [**LayoutXLM**](https://github.com/microsoft/unilm/tree/master/layoutxlm)：面向多语言文档智能的多模态（文本 + 版面排版/格式 + 图像）**文档基座大模型**

> [**MarkupLM**](https://github.com/microsoft/unilm/tree/master/markuplm)：面向富视觉网页标记语言（HTML/XML）理解的标记语言模型预训练

> [**XDoc**](https://github.com/microsoft/unilm/tree/master/xdoc)：面向跨格式文档理解的统一预训练模型

> [**UniSpeech**](https://arxiv.org/abs/2101.07597)：结合自监督学习与监督学习的自动语音识别（ASR）统一预训练模型

> [**UniSpeech-SAT**](https://arxiv.org/pdf/2110.05752.pdf)：具有说话人感知能力的通用语音表征学习

> [**SpeechT5**](https://arxiv.org/abs/2110.07205)：面向语音语言处理的 Encoder-Decoder 统一预训练模型

> [**SpeechLM**](https://arxiv.org/abs/2209.15329)：利用非配对文本数据增强的语音预训练模型

> [**VLMo**](https://github.com/microsoft/unilm/tree/master/vlmo)：统一视觉-语言预训练大模型

> [**VL-BEiT**](https://github.com/microsoft/unilm/tree/master/vl-beit) (```NEW```)：生成式视觉-语言预训练——**BEiT** 架构向多模态领域的全新演进

> [**BEiT-3**](https://github.com/microsoft/unilm/tree/master/beit3) (```NEW```)：通用通用多模态基座大模型，跨任务、跨语言与跨模态大规模预训练**“大融合 (The Big Convergence)”**的重大里程碑成果。

### 工具集 (Toolkits)
> [**s2s-ft**](https://github.com/microsoft/unilm/tree/master/s2s-ft)：Sequence-to-Sequence 序列到序列微调工具包

> [**Aggressive Decoding**](https://arxiv.org/pdf/2205.10350.pdf) (```NEW```)：无损且极致高效的 Sequence-to-Sequence 快速解码算法

### 代表性应用 (Applications)
> [**TrOCR**](https://github.com/microsoft/unilm/tree/master/trocr)：基于 Transformer 与预训练模型的端到端文字识别（OCR）系统

> [**LayoutReader**](https://github.com/microsoft/unilm/tree/master/layoutreader)：结合文本与排版预训练的通用阅读顺序智能检测模型

> [**XLM-T**](https://github.com/microsoft/unilm/tree/master/xlmt)：基于预训练跨语言编码器的多语言神经机器翻译（NMT）

## 相关技术链接
### LLMOps ([代码库](https://github.com/microsoft/lmops))
基于大语言模型（LLM）与多模态大语言模型（MLLM）构建和赋能 AI 应用的通用核心技术体系。

### RedStone ([代码库](https://github.com/microsoft/redstone))
面向大语言模型精选的高质量通用文本、代码、数学以及问答（QA）数据构建工具。

## 最新动态 (News)
- 2024年12月：[**RedStone**](https://github.com/microsoft/redstone) 数据构建库正式开源发布！
- 2023年12月：开源发布 [**LongNet**](https://github.com/microsoft/unilm/tree/master/longnet) 与 [**LongViT**](https://github.com/microsoft/unilm/tree/master/longvit)
- [模型发布] 2023年12月：[**TextDiffuser-2**](https://github.com/microsoft/unilm/tree/master/textdiffuser-2) 模型、源码与 [在线体验 Demo](https://huggingface.co/spaces/JingyeChen22/TextDiffuser-2)。
- 2023年9月：[**Kosmos-2.5**](https://arxiv.org/abs/2309.11419) - 面向高密度文本图像机器阅读的多模态读写大模型。
- [模型发布] 2023年5月：[**TextDiffuser**](https://github.com/microsoft/unilm/tree/master/textdiffuser) 模型与完整源代码。
- [模型发布] 2023年3月：[**BEiT-3**](https://github.com/microsoft/unilm/tree/master/beit3) 预训练模型与源代码。
- 2023年3月：[**Kosmos-1**](https://arxiv.org/abs/2302.14045) - 能够感知通用模态、具备上下文少样本学习（Few-shot）和零样本指令遵循（Zero-shot）能力的多模态大语言模型（MLLM）。
- 2023年1月：[**VALL-E**](https://arxiv.org/abs/2301.02111) 基于大语言模型范式的文本转语音（TTS）合成方法，实现前沿 SOTA 的零样本文本转语音能力，实测样例详见 https://aka.ms/valle。
- [模型发布] 2023年1月：[**E5**](https://github.com/microsoft/unilm/tree/master/e5) - 基于弱监督对比预训练的通用文本向量嵌入模型。
- 2022年11月：[**TorchScale 0.1.1**](https://github.com/microsoft/torchscale) 正式发布！
- 2022年11月：[**TrOCR**](https://arxiv.org/abs/2109.10282) 被 AAAI 2023 顶会正式录用。
- [模型发布] 2022年11月：面向跨格式文档理解的 [**XDoc**](https://github.com/microsoft/unilm/tree/master/xdoc) **BASE** 基础模型发布。
- [模型发布] 2022年9月：面向场景文字识别（STR）的 [**TrOCR**](https://github.com/microsoft/unilm/tree/master/trocr) **BASE** 与 **LARGE** 模型发布。
- [模型发布] 2022年9月：[**BEiT v2**](https://github.com/microsoft/unilm/tree/master/beit2) 源码与预训练模型发布。
- 2022年8月：[**BEiT-3**](https://arxiv.org/abs/2208.10442) 通用多模态基座模型发布，在视觉与图文多模态各项下游任务上取得全新突破。
- 2022年7月：[**SimLM**](https://github.com/microsoft/unilm/tree/master/simlm) - 面向相似度匹配的大规模自监督预训练模型。
- 2022年6月：[**DiT**](https://arxiv.org/abs/2203.02378) 与 [**LayoutLMv3**](https://arxiv.org/abs/2204.08387) 被 ACM Multimedia 2022 顶会正式录用。
- 2022年6月：[**MetaLM**](https://github.com/microsoft/unilm/tree/master/metalm) - 将语言模型构建为基座模型通往语言/多语言、视觉、语音及多模态世界的统一交互界面。
- 2022年6月：[**VL-BEiT**](https://github.com/microsoft/unilm/tree/master/vl-beit) - 基于单阶段端到端训练与统一预训练目标从零构建的双向多模态 Transformer，兼具强大视觉与视觉-语言处理能力。
- [模型发布] 2022年6月：[**LayoutLMv3 中文版**](https://github.com/microsoft/unilm/tree/master/layoutlmv3#form-understanding-on-xfund) - LayoutLMv3 中文专属基座模型。
- [代码发布] 2022年5月：[**Aggressive Decoding**](https://github.com/microsoft/unilm/tree/master/decoding) - 面向 Seq2seq 文本生成的无损加速算法。
- 2022年4月：**超大规模 Transformer 基础框架** = [DeepNet](https://arxiv.org/abs/2203.00555) + [X-MoE](https://arxiv.org/abs/2204.09179)
- [模型发布] 2022年4月：[**LayoutLMv3**](https://github.com/microsoft/unilm/tree/master/layoutlmv3) - 结合统一文本与图像掩码建模的文档智能预训练大模型。
- [模型发布] 2022年3月：[**EdgeFormer**](https://github.com/microsoft/unilm/tree/master/edgelm) - 面向端侧设备 Seq2seq 生成的高效参数轻量 Transformer。
- [模型发布] 2022年3月：[**DiT**](https://github.com/microsoft/unilm/tree/master/dit) - 自监督文档图像 Transformer。在线体验 Demo：[文档版面分析](https://huggingface.co/spaces/nielsr/dit-document-layout-analysis)、[文档图像分类](https://huggingface.co/spaces/microsoft/document-image-transformer)。
- 2022年1月：[**BEiT**](https://openreview.net/forum?id=p-BhZSz59o4) 入选 **ICLR 2022 口头汇报论文 (Oral presentation)**（录用率仅 54 / 3391）。
- [模型发布] 2021年12月16日：面向手写体与印刷体文本识别的 [**TrOCR**](https://github.com/microsoft/unilm/tree/master/trocr) **small** 轻量模型发布，推理速度提升 3 倍。
- 2021年11月24日：[**VLMo**](https://github.com/microsoft/unilm/tree/master/vlmo) 刷新 [视觉问答竞赛 (VQA Challenge)](https://eval.ai/web/challenges/challenge-page/830/leaderboard/2278) SOTA 世界纪录。
- 2021年11月：[大规模多语言机器翻译：迈向 10000 语言对及更远](https://www.microsoft.com/en-us/translator/blog/2021/11/22/multilingual-translation-at-scale-10000-language-pairs-and-beyond/)。
- [模型发布] 2021年11月：[**MarkupLM**](https://github.com/microsoft/unilm/tree/master/markuplm) - 面向文本与标记语言（如 HTML/XML）的预训练模型。
- [模型发布] 2021年11月：[**VLMo**](https://github.com/microsoft/unilm/tree/master/vlmo) - 融合 [**BEiT**](https://github.com/microsoft/unilm/tree/master/beit) 架构的统一视觉-语言预训练大模型。
- 2021年10月：[**WavLM**](https://github.com/microsoft/unilm/tree/master/wavlm) Large 在 [SUPERB](https://superbbenchmark.org/leaderboard) 语音基准榜单刷新 SOTA。
- [模型发布] 2021年10月：[**WavLM**](https://github.com/microsoft/unilm/tree/master/wavlm) - 大规模自监督语音预训练模型。
- [模型发布] 2021年10月：[**TrOCR**](https://huggingface.co/transformers/master/model_doc/trocr.html) 正式上线 [HuggingFace](https://github.com/huggingface/transformers)。
- 2021年9月28日：T-ULRv5（即 <a href="https://arxiv.org/abs/2106.16138" target="_blank">XLM-E</a>/<a href="https://arxiv.org/abs/2007.07834" target="_blank">InfoXLM</a>）登顶 <a href="https://sites.research.google/xtreme" target="_blank">XTREME</a> 多语言权威榜单第一，训练提速达 100 倍。// <a href="https://www.microsoft.com/en-us/research/blog/microsoft-turing-universal-language-representation-model-t-ulrv5-tops-xtreme-leaderboard-and-trains-100x-faster/" target="_blank">官方博客</a>
- [模型发布] 2021年9月：[**LayoutLM-cased**](https://huggingface.co/microsoft/layoutlm-base-cased) 上线 [HuggingFace](https://github.com/huggingface/transformers)。
- [模型发布] 2021年9月：[**TrOCR**](https://github.com/microsoft/unilm/tree/master/trocr) - 基于预训练 [**BEiT**](https://github.com/microsoft/unilm/tree/master/beit) 与 RoBERTa 的纯 Transformer 式 OCR 模型。
- 2021年8月：[**LayoutLMv2**](https://huggingface.co/transformers/master/model_doc/layoutlmv2.html) 与 [**LayoutXLM**](https://huggingface.co/transformers/master/model_doc/layoutxlm.html) 上线 [HuggingFace](https://github.com/huggingface/transformers)。
- [模型发布] 2021年8月：[**LayoutReader**](https://github.com/microsoft/unilm/tree/master/layoutreader) - 基于 LayoutLM 构建的通用文本阅读顺序智能排定模型。
- [模型发布] 2021年8月：[**DeltaLM**](https://github.com/microsoft/unilm/tree/master/deltalm) - 面向多语言生成与翻译的 Encoder-Decoder 预训练模型。
- 2021年8月：[**BEiT**](https://huggingface.co/transformers/master/model_doc/beit.html) 上线 [HuggingFace](https://github.com/huggingface/transformers)。
- [模型发布] 2021年7月：[**BEiT**](https://github.com/microsoft/unilm/tree/master/beit) - 开启计算机视觉领域的“BERT 时刻”。
- [模型发布] 2021年6月：[**LayoutLMv2**](https://github.com/microsoft/unilm/tree/master/layoutlmv2)、[**LayoutXLM**](https://github.com/microsoft/unilm/tree/master/layoutxlm)、[**MiniLMv2**](https://github.com/microsoft/unilm/tree/master/minilm) 以及 [**AdaLM**](https://github.com/microsoft/unilm/tree/master/adalm)。
- 2021年5月：[LayoutLMv2](https://github.com/microsoft/unilm/tree/master/layoutlmv2)、InfoXLMv2、MiniLMv2、UniLMv3 和 AdaLM 被 ACL 2021 顶会正式录用。
- 2021年4月：将 LayoutLM 扩展至多语言的全新 [LayoutXLM](https://github.com/microsoft/unilm/tree/master/layoutxlm) 面世！同时推出包含中、日、西、法、意、德、葡 7 种语言多语言表单理解权威基准评测数据集 [XFUND](https://github.com/doc-analysis/XFUND)。
- 2021年3月：[InfoXLM](https://github.com/microsoft/unilm/tree/master/infoxlm) 被 NAACL 2021 顶会录用。
- 2020年12月29日：[LayoutLMv2](https://arxiv.org/abs/2012.14740) 发布，刷新包括 [DocVQA](https://rrc.cvc.uab.es/?ch=17&com=evaluation&task=1) 与 [SROIE](https://rrc.cvc.uab.es/?ch=13&com=evaluation&task=3) 等多项文档智能顶级榜单记录。
- 2020年10月8日：T-ULRv2（即 [InfoXLM](https://arxiv.org/abs/2007.07834)）登顶 [XTREME](https://sites.research.google/xtreme) 多语言综合评测榜单第一。// [官方博客](https://www.microsoft.com/en-us/research/blog/microsoft-turing-universal-language-representation-model-t-ulrv2-tops-xtreme-leaderboard/)
- 2020年9月：[MiniLM](https://github.com/microsoft/unilm/tree/master/minilm) 被 NeurIPS 2020 顶会正式录用。
- 2020年7月16日：[**InfoXLM** (多语言版 UniLM)](https://github.com/microsoft/unilm/tree/master/infoxlm) 论文正式公布：[arXiv](https://arxiv.org/pdf/2007.07834.pdf)。
- 2020年6月：[UniLMv2](https://github.com/microsoft/unilm/tree/master/unilm) 被 ICML 2020 录用；[LayoutLM](https://github.com/microsoft/unilm/tree/master/layoutlm) 被 KDD 2020 录用。
- 2020年4月5日：[**多语言版 MiniLM (Multilingual MiniLM)**](https://github.com/microsoft/unilm/tree/master/minilm) 正式发布！
- 2019年9月：[UniLMv1](https://github.com/microsoft/unilm/tree/master/unilm-v1) 被 NeurIPS 2019 顶会录用。

<!--
## 发布历史记录

**\*\*\*\*\* ```2022年10月新发布```: [XDoc](https://github.com/microsoft/unilm/tree/master/xdoc) 发布 \*\*\*\*\***

- [x] [**XDoc**](https://github.com/microsoft/unilm/tree/master/xdoc) (2022年10月7日): XDoc, 跨格式文档理解统一预训练大模型。

**\*\*\*\*\* ```2022年5月新发布```: [Aggressive Decoding](https://github.com/microsoft/unilm/tree/master/decoding) 发布 \*\*\*\*\***

- [x] [**Aggressive Decoding**](https://github.com/microsoft/unilm/tree/master/decoding) (2022年5月20日): 面向 Seq2seq 生成的无损极速解码新范式。

**\*\*\*\*\* ```2022年4月新发布```: [LayoutLMv3](https://github.com/microsoft/unilm/tree/master/layoutlmv3) 发布 \*\*\*\*\***

- [x] [**LayoutLM 3.0**](https://github.com/microsoft/unilm/tree/master/layoutlmv3) (2022年4月19日): 统一文本与图像掩码的文档智能多模态预训练 Transformer。

**\*\*\*\*\* ```2022年3月```: [EdgeFormer](https://github.com/microsoft/unilm/tree/master/edgelm) 发布 \*\*\*\*\***

- [x] [**EdgeFormer**](https://github.com/microsoft/unilm/tree/master/edgelm) (2022年3月18日): 面向端侧 Seq2seq 文本生成的轻量参数高效 Transformer。

**\*\*\*\*\* ```2022年3月```: [DiT](https://github.com/microsoft/unilm/tree/master/dit) 发布 \*\*\*\*\***

- [x] [**DiT**](https://github.com/microsoft/unilm/tree/master/dit) (2022年3月4日): 自监督文档图像 Transformer。

**\*\*\*\*\* ```2021年10月```: [WavLM](https://github.com/microsoft/unilm/tree/master/wavlm) 发布 \*\*\*\*\***

- [x] [**WavLM**](https://github.com/microsoft/unilm/tree/master/wavlm) (2021年10月27日): 面向全栈语音下游任务的预训练大模型。

**\*\*\*\*\* ```2021年10月```: [MarkupLM](https://github.com/microsoft/unilm/tree/master/markuplm) 发布 \*\*\*\*\***

- [x] [**MarkupLM**](https://github.com/microsoft/unilm/tree/master/markuplm) (2021年10月19日): 文本与网页标记语言的联合预训练。

**\*\*\*\*\* ```2021年9月```: [TrOCR](https://github.com/microsoft/unilm/tree/master/trocr) 发布 \*\*\*\*\***

- [x] [**TrOCR**](https://github.com/microsoft/unilm/tree/master/trocr) (2021年9月22日): 基于预训练模型的端到端 Transformer OCR。

**\*\*\*\*\* ```2021年8月```: [LayoutReader](https://github.com/microsoft/unilm/tree/master/layoutreader) 发布 \*\*\*\*\***

- [x] [**LayoutReader**](https://github.com/microsoft/unilm/tree/master/layoutreader) (2021年8月26日): 基于文本与排版预训练的阅读顺序识别。

**\*\*\*\*\* ```2021年8月```: [DeltaLM](https://github.com/microsoft/unilm/tree/master/deltalm) 发布 \*\*\*\*\***

- [x] [**DeltaLM**](https://github.com/microsoft/unilm/tree/master/deltalm) (2021年8月): 面向语言生成与机器翻译的 Encoder-Decoder 预训练。

**\*\*\*\*\* ```2021年7月```: [BEiT](https://github.com/microsoft/unilm/tree/master/beit) 发布 \*\*\*\*\***

- [x] [**BEiT**](https://github.com/microsoft/unilm/tree/master/beit) (2021年6月15日): 图像 Transformer 的 BERT 式预训练。

**\*\*\*\*\* ```2021年6月```: [LayoutXLM](https://github.com/microsoft/unilm/tree/master/layoutxlm) | [AdaLM](https://github.com/microsoft/unilm/tree/master/adalm) | [MiniLMv2](https://github.com/microsoft/unilm/tree/master/minilm) 发布 \*\*\*\*\***

- [x] [**LayoutXLM**](https://github.com/microsoft/unilm/tree/master/layoutxlm) (2021年4月17日): 面向多语言文档理解的多模态预训练。
- [x] [**AdaLM**](https://github.com/microsoft/unilm/tree/master/adalm) (2021年6月): 预训练模型领域自适应算法。
- [x] [**MiniLMv2**](https://github.com/microsoft/unilm/tree/master/minilm) (2020年12月): 压缩预训练 Transformer 的自注意力关系蒸馏算法。

**\*\*\*\*\* ```2021年5月```: [LayoutLMv2](https://github.com/microsoft/unilm/tree/master/layoutlmv2) | [LayoutXLM](https://github.com/microsoft/unilm/tree/master/layoutxlm) 发布 \*\*\*\*\***

- [x] [**LayoutLM 2.0**](https://github.com/microsoft/unilm/tree/master/layoutlmv2) (2020年12月29日): 融合文本、布局与图像信息的多模态文档理解预训练。

**\*\*\*\*\* ```2020年2月```: [UniLM v2](https://github.com/microsoft/unilm/tree/master/unilm) | [MiniLM v1](https://github.com/microsoft/unilm/tree/master/minilm) | [LayoutLM v1](https://github.com/microsoft/unilm/tree/master/layoutlm) | [s2s-ft v1](https://github.com/microsoft/unilm/tree/master/s2s-ft) 发布 \*\*\*\*\***

- [x] [**LayoutLM 1.0**](https://github.com/microsoft/unilm/tree/master/layoutlm) (2020年2月18日): 面向文档图像理解的预训练模型。
- [x] [**s2s-ft 1.0**](https://github.com/microsoft/unilm/tree/master/s2s-ft) (2020年2月26日): 序列到序列语言生成微调工具包。
- [x] [**MiniLM 1.0**](https://github.com/microsoft/unilm/tree/master/minilm) (2020年2月26日): 深度自注意力蒸馏模型。
- [x] [**UniLM 2.0**](https://github.com/microsoft/unilm/tree/master/unilm) (2020年2月28日): 统一双向与自回归语言模型预训练。

**\*\*\*\*\* 2019年10月1日: UniLM v1 发布 \*\*\*\*\***

- [x] [**UniLM v1**](https://github.com/microsoft/unilm/tree/master/unilm-v1) (2019年9月30日): NeurIPS 2019 论文《Unified Language Model Pre-training for Natural Language Understanding and Generation》源码与预训练模型。

-->

## 开源许可证

本项目基于源代码根目录下的 LICENSE 文件所述许可证开源。
部分源代码基于 [transformers](https://github.com/huggingface/transformers) 项目扩展构建。

[Microsoft 开源行为准则](https://opensource.microsoft.com/codeofconduct)

### 联系方式

若在使用预训练模型时遇到任何问题或疑问，欢迎直接在 GitHub 上提交 Issue。

如需进行学术合作或其他商务沟通，请联系 [韦福如博士 (Furu Wei)](https://thegenerality.com)（邮箱：`fuwei@microsoft.com`）。

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年09月13日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
