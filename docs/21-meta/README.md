# 基于Meta家族模型构建

## 介绍

本课将涵盖：

- 探索Meta家族的两大主要模型——Llama 3.1和Llama 3.2
- 了解每个模型的使用场景和适用情况
- 展示每个模型独特特性的代码示例

## Meta家族模型

在本课中，我们将探索Meta家族或称为“Llama群”中的两个模型——Llama 3.1和Llama 3.2

这些模型有不同的变体，且可在Github Model marketplace获得。以下是使用Github Models进行[原型设计与AI模型](https://docs.github.com/en/github-models/prototyping-with-ai-models?WT.mc_id=academic-105485-koreyst)的更多详细信息。

模型变体:
- Llama 3.1 - 70B 指令
- Llama 3.1 - 405B 指令
- Llama 3.2 - 11B 视觉指令
- Llama 3.2 - 90B 视觉指令

*注意: Llama 3也可以在Github Models上找到，但本课不做覆盖*

## Llama 3.1

Llama 3.1具有4050亿参数，属于开源LLM类别。

该模型是对早期版本Llama 3的升级，提供：

- 更大的上下文窗口——128k tokens vs 8k tokens
- 更大的最大输出Token数——4096 vs 2048
- 更好的多语言支持——由于训练token数量增加

这些使得Llama 3.1能够在构建GenAI应用时处理更复杂的用例，包括：
- 本地函数调用——能够调用LLM工作流之外的外部工具和函数
- 更好的RAG性能——由于更高的上下文窗口
- 合成数据生成——能够创建有效的数据用于诸如微调的任务

### 本地函数调用

Llama 3.1已被微调以更有效地进行函数或工具调用。它还具有两个内置工具，模型可以根据用户的提示自行决定是否需要使用这些工具。这些工具包括：

- **Brave Search**——可以通过进行网页搜索获取最新信息，如天气情况
- **Wolfram Alpha**——可以用于更复杂的数学计算，因此无需编写自己的函数

你也可以创建自己的自定义工具供LLM调用。

在下面的代码示例中：

- 我们在系统提示中定义了可用的工具（brave_search，wolfram_alpha）。
- 发送一个用户提示，询问某个城市的天气。
- LLM将通过调用Brave Search工具回复，这看起来像这样`<|python_tag|>brave_search.call(query="Stockholm weather")`

*注意：此示例仅进行工具调用。如果您希望获取结果，则需要在Brave API页面上创建一个免费账户并定义函数本身`*

```python 
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import AssistantMessage, SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.inference.ai.azure.com"
model_name = "meta-llama-3.1-405b-instruct"

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)


tool_prompt=f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Environment: ipython
Tools: brave_search, wolfram_alpha
Cutting Knowledge Date: December 2023
Today Date: 23 July 2024

You are a helpful assistant<|eot_id|>
"""

messages = [
    SystemMessage(content=tool_prompt),
    UserMessage(content="What is the weather in Stockholm?"),

]

response = client.complete(messages=messages, model=model_name)

print(response.choices[0].message.content)
```

## Llama 3.2

尽管Llama 3.1是一种LLM，但其一个局限性是多模态支持。也就是说，无法使用诸如图像等不同类型的输入作为提示并提供响应。这种能力是Llama 3.2的主要特性之一。这些特性还包括：

- 多模态——能够评估文本和图像提示
- 小到中等大小的变体（11B和90B）——提供灵活的部署选项
- 仅文本变体（1B和3B）——允许模型部署在边缘/移动设备上并提供低延迟

多模态支持代表了开源模型领域的一大进步。下面的代码示例使用Llama 3.2 90B接受图像和文本提示以获得图像分析。

### Llama 3.2的多模态支持

```python 
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import (
    SystemMessage,
    UserMessage,
    TextContentItem,
    ImageContentItem,
    ImageUrl,
    ImageDetailLevel,
)
from azure.core.credentials import AzureKeyCredential

token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.inference.ai.azure.com"
model_name = "Llama-3.2-90B-Vision-Instruct"

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

response = client.complete(
    messages=[
        SystemMessage(
            content="You are a helpful assistant that describes images in details."
        ),
        UserMessage(
            content=[
                TextContentItem(text="What's in this image?"),
                ImageContentItem(
                    image_url=ImageUrl.load(
                        image_file="sample.jpg",
                        image_format="jpg",
                        detail=ImageDetailLevel.LOW)
                ),
            ],
        ),
    ],
    model=model_name,
)

print(response.choices[0].message.content)
```

## 学习不会止步于此，继续学习之旅

完成本课后，请浏览我们的[生成式AI学习合集](https://aka.ms/genai-collection?WT.mc_id=academic-105485-koreyst)，继续提升您的生成式AI知识！