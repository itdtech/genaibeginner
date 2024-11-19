# 初学者生成性 AI 小语言模型简介

生成性 AI 是人工智能的一个迷人领域，专注于创造能够生成新内容的系统。这些内容可以包括文本、图像、音乐，甚至整个虚拟环境。生成性 AI 最令人激动的应用之一是在语言模型领域。

## 什么是小语言模型？

小语言模型（SLM）是大语言模型（LLM）的缩小版，利用许多大型语言模型的架构原理和技术，同时显著减少了计算资源的占用。

SLM 是一类设计用来生成类人文本的语言模型。不像 GPT-4 这样的更大模型，SLM 更加紧凑和高效，适合计算资源有限的应用。尽管它们体积较小，但仍能执行各种任务。通常，SLM 是通过压缩或提取 LLM 构建的，旨在保留原模型的大部分功能和语言能力。这种模型大小的缩减降低了整体复杂性，使得 SLM 在内存使用和计算需求方面更加高效。尽管做了这些优化，SLM 仍然能够执行多种自然语言处理（NLP）任务：

- 文本生成：创建连贯且上下文相关的句子或段落。
- 文本补全：基于给定的提示预测和补全句子。
- 翻译：将文本从一种语言转化为另一种语言。
- 摘要：将长篇文本缩短为更易消化的摘要。

尽管性能或理解深度上可能不及更大的模型。

## 小语言模型如何工作？

SLM 在大量文本数据上进行训练。在训练过程中，它们学习语言的模式和结构，使其能够生成语法正确且上下文适当的文本。训练过程包括：

- 数据收集：从各种来源收集大型文本数据集。
- 预处理：清理和组织数据，使其适合训练。
- 训练：使用机器学习算法教模型如何理解和生成文本。
- 微调：调整模型以提高其在特定任务上的表现。

SLM 的发展符合在资源受限环境中部署模型的需求，如移动设备或边缘计算平台，在这些环境中，由于资源需求较高，完全规模的 LLM 可能不切实际。通过专注于效率，SLM 平衡了性能与可访问性，使其在不同领域中得到更广泛应用。

![slm](./img/slm.png?WT.mc_id=academic-105485-koreyst)

## 学习目标

在本课程中，我们希望介绍 SLM 的知识并结合 Microsoft Phi-3 学习文本内容、视觉和 MoE 等不同场景。

到本课程结束时，您应能够回答以下问题：

- 什么是 SLM
- SLM 和 LLM 的区别是什么
- 什么是 Microsoft Phi-3/3.5 系列
- 如何推理 Microsoft Phi-3/3.5 系列

准备好了吗？让我们开始吧。

## 大语言模型（LLM）和小语言模型（SLM）的区别

LLM 和 SLM 都基于概率机器学习的基础原理，采用类似的架构设计、训练方法、数据生成过程和模型评估技术。然而，这两种模型之间有几个关键因素的区别。

## 小语言模型的应用

SLM 有广泛的应用，包括：

- 聊天机器人：提供客户支持并以对话方式与用户互动。
- 内容创作：帮助作家生成想法甚至整个文章草稿。
- 教育：帮助学生完成写作作业或学习新语言。
- 无障碍：创建为残障人士设计的工具，如文本转语音系统。

**规模**

LLM 和 SLM 之间的主要区别在于模型的规模。LLM，如 ChatGPT（GPT-4），可能包含约 1.76 万亿参数，而开源 SLM，如 Mistral 7B 的设计目标则显著少于这些参数，约为 70 亿。此差异主要源于模型架构和训练过程的不同。例如，ChatGPT 使用自注意力机制在一个编码器-解码器框架中，而 Mistral 7B 使用滑动窗口注意力，仅在解码器模型中进行更高效的训练。这种架构差异对这些模型的复杂性和性能有深远影响。

**理解能力**

SLM 通常针对特定领域的性能进行优化，使其高度专业化，但在跨多个知识领域提供广泛上下文理解的能力上可能有限。相反，LLM 旨在模拟更全面的人类智能。LLM 在大型多样化数据集上进行训练，设计目的是在各种领域中表现出色，具有更大的多功能性和适应性。因此，LLM 更适合执行更广泛的下游任务，如自然语言处理和编程。

**计算**

LLM 的训练和部署是资源密集型过程，通常需要大型 GPU 集群等显著的计算基础设施。例如，从头开始训练 ChatGPT 这样的模型可能需要数千个 GPU，持续时间较长。相反，SLM 由于其较少的参数数量，在计算资源方面更为可访问。像 Mistral 7B 这样的模型可以在配备有中等 GPU 能力的本地机器上进行训练和运行，尽管训练仍需要多个 GPU 数小时的时间。

**偏见**

偏见是 LLM 中的已知问题，主要是由于训练数据的性质。这些模型通常依赖于来自互联网的公开可用的原始数据，可能会低估或错误代表某些群体，介绍错误标签或反映受方言、地理变化和语法规则影响的语言偏见。此外，LLM 架构的复杂性可能无意中加剧偏见，若没有仔细的微调，这种偏见可能不会被察觉。另一方面，SLM 在更受限的特定领域数据集上进行训练，本质上不太容易出现这种偏见，尽管它们并非免疫。

**推理速度**

SLM 较小的规模使其在推理速度方面具有显著优势，能够在本地硬件上高效地生成输出，无需大量的并行处理。相反，由于规模和复杂性，LLM 通常需要大量并行计算资源以获得可接受的推理时间。当广泛部署时，同时有多个并发用户进一步减慢 LLM 的响应时间，尤其是在规模化部署时。

总之，尽管 LLM 和 SLM 都在机器学习的基础上，但它们在模型规模、资源需求、上下文理解、易受偏见影响和推理速度等方面存在显著差异。这些区别反映了它们在不同用例中的适用性，LLM 更加多功能但资源密集，而 SLM 提供了更特定领域的效率，同时计算需求减少。

***注意：在本章中，我们将以 Microsoft Phi-3 / 3.5 为例介绍 SLM。***

## Phi-3 / Phi-3.5 系列简介

Phi-3 / 3.5 系列主要针对文本、视觉和代理（MoE）应用场景：

### Phi-3 / 3.5 Instruct

主要用于文本生成、聊天完成和内容信息提取等。

**Phi-3-mini**

3.8B 语言模型可在 Microsoft Azure AI Studio、Hugging Face 和 Ollama 上使用。Phi-3 模型在关键基准测试上显著优于同等和更大规模的语言模型（参见下面的基准测试数据，数值越高越好）。Phi-3-mini 表现超过了两倍规模的模型，而 Phi-3-small 和 Phi-3-medium 则超过了更大的模型，包括 GPT-3.5。

**Phi-3-small & medium**

仅有 70 亿参数的 Phi-3-small 在各种语言、推理、编码和数学基准测试中击败了 GPT-3.5T。

拥有 140 亿参数的 Phi-3-medium 延续了这一趋势，表现优于 Gemini 1.0 Pro。

**Phi-3.5-mini**

我们可以将其视为 Phi-3-mini 的升级版。虽然参数保持不变，但它增强了对多种语言的支持（支持 20 多种语言：阿拉伯语、中文、捷克语、丹麦语、荷兰语、英语、芬兰语、法语、德语、希伯来语、匈牙利语、意大利语、日语、韩语、挪威语、波兰语、葡萄牙语、俄语、西班牙语、瑞典语、泰语、土耳其语、乌克兰语）并且增强了对长文本的支持。

Phi-3.5-mini 具有 3.8B 参数，表现优于同等大小的语言模型，甚至与其两倍大小的模型相当。

### Phi-3 / 3.5 Vision

我们可以将 Phi-3/3.5 的 Instruct 模型视为 Phi 的理解能力，Vision 则赋予 Phi 理解世界的眼睛。

**Phi-3-Vision**

Phi-3-vision 仅有 42 亿参数，在一般视觉推理任务、OCR 和表格与图表理解任务上表现优于更大的模型，如 Claude-3 Haiku 和 Gemini 1.0 Pro V。

**Phi-3.5-Vision**

Phi-3.5-Vision 也是 Phi-3-Vision 的升级版，增加了对多张图像的支持。您可以将其视为视觉能力的提升，不仅能看图片，还能看视频。

Phi-3.5-vision 表现优于更大的模型，如 Claude-3.5 Sonnet 和 Gemini 1.5 Flash，在 OCR、表格和图表理解任务上表现优越，在一般视觉知识推理任务上则相当。支持多帧输入，即对多个输入图像进行推理。

### Phi-3.5-MoE

***混合专家模型（MoE）*** 允许模型在预训练时使用更少的计算资源，这意味着可以在与密集模型相同的计算预算下显著扩大模型或数据集的规模。特别是，MoE 模型应在预训练期间更快地达到与其密集模型对等的质量。

Phi-3.5-MoE 由 16x3.8B 专家模块组成。Phi-3.5-MoE 仅用 66 亿活动参数就达到了与更大模型相当的推理、语言理解和数学水平。

我们可以根据不同的场景使用 Phi-3/3.5 系列模型。与 LLM 不同，您可以在边缘设备上部署 Phi-3/3.5-mini 或 Phi-3/3.5-Vision。

## 如何使用 Phi-3/3.5 系列模型

我们希望在不同的场景中使用 Phi-3/3.5。接下来，我们将基于不同的场景使用 Phi-3/3.5。

![phi3](./img/phi3.png?WT.mc_id=academic-105485-koreyst)

### 推理差异 Cloud 的 API

**GitHub 模型**

GitHub 模型是最直接的方法。您可以通过 GitHub 模型快速访问 Phi-3/3.5-Instruct 模型。结合 Azure AI 推理 SDK / OpenAI SDK，您可以通过代码访问 API 完成 Phi-3/3.5-Instruct 调用。您还可以通过 Playground 测试不同的效果。

- 演示：在中文场景中对比 Phi-3-mini 和 Phi-3.5-mini 的效果

![phi3](./img/gh1.png?WT.mc_id=academic-105485-koreyst)

![phi35](./img/gh2.png?WT.mc_id=academic-105485-koreyst)

**Azure AI Studio**

或者如果我们想使用视觉和 MoE 模型，您可以使用 Azure AI Studio 完成调用。如果您感兴趣，可以阅读 Phi-3 Cookbook 以了解如何通过 Azure AI Studio 调用 Phi-3/3.5 Instruct、Vision 和 MoE [点击此链接](https://github.com/microsoft/Phi-3CookBook/blob/main/md/02.QuickStart/AzureAIStudio_QuickStart.md?WT.mc_id=academic-105485-koreyst)

**NVIDIA NIM**

除了 Azure 和 GitHub 提供的基于云的模型目录解决方案，您还可以使用 [Nivida NIM](https://developer.nvidia.com/nim?WT.mc_id=academic-105485-koreyst) 完成相关调用。您可以访问 NIVIDA NIM 完成 Phi-3/3.5 系列的 API 调用。NVIDIA NIM（NVIDIA 推理微服务）是一组加速的推理微服务，旨在帮助开发者在不同环境中高效部署 AI 模型，包括云、数据中心和工作站。

以下是 NVIDIA NIM 的一些关键特性：

- **部署简便性**：NIM 允许通过单个命令部署 AI 模型，便于集成到现有工作流中。
- **优化性能**：它利用 NVIDIA 的预优化推理引擎，如 TensorRT 和 TensorRT-LLM，确保低延迟和高吞吐量。
- **可扩展性**：NIM 支持 Kubernetes 上的自动扩展，能有效处理不同的工作负载。
- **安全性与控制**：组织可以通过在自有托管基础设施上自行托管 NIM 微服务，维护对其数据和应用的控制。
- **标准 API**：NIM 提供行业标准的 API，使得构建和集成 AI 应用如聊天机器人、AI 助手等变得简单。

NIM 是 NVIDIA AI Enterprise 的一部分，旨在简化 AI 模型的部署和运营化，确保其在 NVIDIA GPU 上高效运行。

- 演示：使用 Nividia NIM 调用 Phi-3.5-Vision-API [[点击此链接](./python/Phi-3-Vision-Nividia-NIM.ipynb?WT.mc_id=academic-105485-koreyst)]

### 在本地环境推理 Phi-3/3.5
推理是指生成响应或预测的过程，基于 Phi-3 或任何类似 GPT-3 的语言模型接收到的输入。当你向 Phi-3 提供提示或问题时，它使用其已训练的神经网络，通过分析其训练数据中的模式和关系来推断最可能且相关的响应。

**Hugging face Transformer**
Hugging Face Transformers 是一个强大的库，设计用于自然语言处理（NLP）和其他机器学习任务。以下是一些关键点：

1. **预训练模型**：它提供了数千个预训练模型，可用于文本分类、命名实体识别、问题回答、摘要、翻译和文本生成等任务。
2. **框架互通性**：该库支持多个深度学习框架，包括 PyTorch、TensorFlow 和 JAX。这允许你在一个框架中训练模型，并在另一个框架中使用。
3. **多模态能力**：除了 NLP，Hugging Face Transformers 还支持计算机视觉（例如图像分类、目标检测）和音频处理（例如语音识别、音频分类）。
4. **使用简便**：该库提供了 API 和工具，便于下载和微调模型，使初学者和专家都易于使用。
5. **社区和资源**：Hugging Face 拥有一个充满活力的社区，以及详尽的文档、教程和指南，帮助用户上手并充分利用该库。
可以参考其[官方网站](https://huggingface.co/docs/transformers/index?WT.mc_id=academic-105485-koreyst)或其[GitHub 仓库](https://github.com/huggingface/transformers?WT.mc_id=academic-105485-koreyst)。

这是最常用的方法，但也需要 GPU 加速。毕竟，像 Vision 和 MoE 这样的场景需要大量计算，如果不进行量化，使用 CPU 计算会非常有限。

- 演示：使用 Transformer 调用 Phi-3.5-Instuct [点击此链接](./python/phi35-instruct-demo.ipynb?WT.mc_id=academic-105485-koreyst)

- 演示：使用 Transformer 调用 Phi-3.5-Vision [点击此链接](./python/phi35-vision-demo.ipynb?WT.mc_id=academic-105485-koreyst)

- 演示：使用 Transformer 调用 Phi-3.5-MoE [点击此链接](./python/phi35_moe_demo.ipynb?WT.mc_id=academic-105485-koreyst)

**Ollama**
[Ollama](https://ollama.com/?WT.mc_id=academic-105485-koreyst) 是一个平台，旨在使您能够在本地机器上更轻松地运行大型语言模型（LLM）。它支持如 Llama 3.1、Phi 3、Mistral 和 Gemma 2 等各种模型。该平台通过将模型权重、配置和数据打包为单个包，使用户更容易自定义和创建自己的模型。Ollama 可在 macOS、Linux 和 Windows 上运行。如果您希望在无需依赖云服务的情况下实验或部署 LLM，这是一个非常好的工具。Ollama 是最直接的方式，您只需要执行以下语句。

```bash

ollama run phi3.5

```

**ONNX Runtime for GenAI**

[ONNX Runtime](https://github.com/microsoft/onnxruntime-genai?WT.mc_id=academic-105485-koreyst) 是一个跨平台的推理和训练机器学习加速器。适用于 ONNX 模型的 ONNX Runtime for Generative AI (GENAI) 库提供了生成 AI 循环，包括带有 ONNX Runtime 的推理，logits 处理、搜索和采样以及 KV 缓存管理。

## 什么是 ONNX Runtime？
ONNX Runtime 是一个开源项目，支持高性能的机器学习模型推理。它支持 Open Neural Network Exchange (ONNX) 格式的模型，这是一个机器学习模型表示的标准。ONNX Runtime 可以通过利用硬件加速器进行优化，提供最佳性能支持不同的硬件、驱动程序和操作系统。

## 什么是生成性 AI？
生成性 AI 指的是能够生成新内容（例如文本、图像或音乐）的 AI 系统，基于它们被训练