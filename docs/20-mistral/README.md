# 使用Mistral模型构建

## 介绍

本课程将涵盖:
- 探索不同的Mistral模型
- 理解每个模型的用例和场景
- 代码示例展示每个模型的独特功能

## Mistral模型

在本课程中，我们将探索三种不同的Mistral模型：**Mistral Large**、**Mistral Small** 和 **Mistral Nemo**。

这些模型都可以在Github Model市场上免费获取。本笔记本中的代码将使用这些模型运行代码。更多关于使用Github模型进行[原型设计与AI模型](https://docs.github.com/en/github-models/prototyping-with-ai-models?WT.mc_id=academic-105485-koreyst)的详细信息，请查看链接。

## Mistral Large 2 (2407)
Mistral Large 2目前是Mistral的旗舰模型，设计用于企业级使用。

该模型是对原始Mistral Large的升级，提供以下提升：
- 更大的上下文窗口 - 128k vs 32k
- 数学和编程任务上更好的性能 - 平均准确率 76.9% vs 60.4%
- 增强的多语言性能 - 支持语言包括英语、法语、德语、西班牙语、意大利语、葡萄牙语、荷兰语、俄语、中文、日语、韩语、阿拉伯语和印地语。

凭借这些特性，Mistral Large擅长
- *检索增强生成 (RAG)* - 由于更大的上下文窗口
- *功能调用* - 该模型具有原生功能调用功能，允许与外部工具和API集成。调用可以并行进行或顺序进行。
- *代码生成* - 该模型在Python、Java、TypeScript和C++的生成方面表现出色。

### 使用Mistral Large 2的RAG示例

在本示例中，我们使用Mistral Large 2在文本文档上运行RAG模式。问题用韩语编写，询问作者在上大学前的活动。

它使用Cohere Embeddings Model来创建文本文档和问题的嵌入。对于此示例，它使用faiss Python包作为向量存储。

发送给Mistral模型的提示包括问题和检索到的与问题相似的文本块。然后模型提供自然语言的回答。

```python 
pip install faiss-cpu
```

```python 
import requests
import numpy as np
import faiss
import os

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference import EmbeddingsClient

endpoint = "https://models.inference.ai.azure.com"
model_name = "Mistral-large"
token = os.environ["GITHUB_TOKEN"]

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

response = requests.get('https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt')
text = response.text

chunk_size = 2048
chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
len(chunks)

embed_model_name = "cohere-embed-v3-multilingual" 

embed_client = EmbeddingsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(token)
)

embed_response = embed_client.embed(
    input=chunks,
    model=embed_model_name
)

text_embeddings = []
for item in embed_response.data:
    length = len(item.embedding)
    text_embeddings.append(item.embedding)
text_embeddings = np.array(text_embeddings)

d = text_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(text_embeddings)

question = "저자가 대학에 오기 전에 주로 했던 두 가지 일은 무엇이었나요?"

question_embedding = embed_client.embed(
    input=[question],
    model=embed_model_name
)

question_embeddings = np.array(question_embedding.data[0].embedding)

D, I = index.search(question_embeddings.reshape(1, -1), k=2) # 距离，索引
retrieved_chunks = [chunks[i] for i in I.tolist()[0]]

prompt = f"""
上下文信息如下。
---------------------
{retrieved_chunks}
---------------------
根据上下文信息而非已有知识来回答查询。
查询：{question}
回答：
"""

chat_response = client.complete(
    messages=[
        SystemMessage(content="您是一位乐于助人的助手。"),
        UserMessage(content=prompt),
    ],
    temperature=1.0,
    top_p=1.0,
    max_tokens=1000,
    model=model_name
)

print(chat_response.choices[0].message.content)
```

## Mistral Small
Mistral Small是Mistral高级/企业类别中的另一个模型。顾名思义，该模型是一个小型语言模型（SLM）。使用Mistral Small的优点是：
- 相比Mistral LLMs如Mistral Large和NeMo可节省成本- 价格下降80%
- 低延迟 - 相比Mistral的LLMs响应速度更快
- 灵活 - 可以跨不同环境部署，对所需资源的限制更少

Mistral Small适用于：
- 文本任务，例如摘要、情感分析和翻译
- 由于成本效益高，适用于频繁请求的应用
- 低延迟代码任务，如代码审查和建议

## 比较Mistral Small和Mistral Large

为了展示Mistral Small和Large的延迟差异，请运行以下单元。

你应该可以看到响应时间在3-5秒之间的差异。同时注意相同提示下的响应长度和风格。

```python 

import os 
endpoint = "https://models.inference.ai.azure.com"
model_name = "Mistral-small"
token = os.environ["GITHUB_TOKEN"]

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

response = client.complete(
    messages=[
        SystemMessage(content="您是一位乐于助人的编程助手。"),
        UserMessage(content="你能写一个Python函数来做Fizz Buzz测试吗？"),
    ],
    temperature=1.0,
    top_p=1.0,
    max_tokens=1000,
    model=model_name
)

print(response.choices[0].message.content)

```

```python

import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

endpoint = "https://models.inference.ai.azure.com"
model_name = "Mistral-large"
token = os.environ["GITHUB_TOKEN"]

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

response = client.complete(
    messages=[
        SystemMessage(content="您是一位乐于助人的编程助手。"),
        UserMessage(content="你能写一个Python函数来做Fizz Buzz测试吗？"),
    ],
    temperature=1.0,
    top_p=1.0,
    max_tokens=1000,
    model=model_name
)

print(response.choices[0].message.content)

```

## Mistral NeMo

与本课讨论的其他两个模型相比，Mistral NeMo是唯一拥有Apache2许可证的免费模型。

它被视为早期Mistral开源LLM（Mistral 7B）的升级版。

NeMo模型的其他一些特点包括：

- *更高效的分词:* 该模型使用Tekken分词器而不是更常用的tiktoken。这样可以在更多语言和代码上获得更好的性能。

- *微调:* 基础模型可以进行微调。这使得在需要微调的用例中具有更大的灵活性。

- *原生功能调用* - 像Mistral Large一样，该模型在训练中包含了功能调用。使其成为首批具有此功能的开源模型之一。

### 比较分词器

在这个示例中，我们将比较Mistral NeMo与Mistral Large在分词方面的处理。

两个示例都采用相同的提示，你应看到NeMo返回的标记数比Mistral Large少。

```bash
pip install mistral-common
```

```python 
# 导入所需的包：
from mistral_common.protocol.instruct.messages import (
    UserMessage,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import (
    Function,
    Tool,
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

# 加载Mistral分词器

model_name = "open-mistral-nemo"

tokenizer = MistralTokenizer.from_model(model_name)

# 对消息列表进行分词
tokenized = tokenizer.encode_chat_completion(
    ChatCompletionRequest(
        tools=[
            Tool(
                function=Function(
                    name="get_current_weather",
                    description="获取当前天气",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "城市和州，例如：San Francisco, CA",
                            },
                            "format": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "温度单位。根据用户位置推断。",
                            },
                        },
                        "required": ["location", "format"],
                    },
                )
            )
        ],
        messages=[
            UserMessage(content="今天巴黎的天气怎么样"),
        ],
        model=model_name,
    )
)
tokens, text = tokenized.tokens, tokenized.text

# 计算标记数量
print(len(tokens))
```

```python
# 导入所需的包：
from mistral_common.protocol.instruct.messages import (
    UserMessage,
)
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.protocol.instruct.tool_calls import (
    Function,
    Tool,
)
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

# 加载Mistral分词器

model_name = "mistral-large-latest"

tokenizer = MistralTokenizer.from_model(model_name)

# 对消息列表进行分词
tokenized = tokenizer.encode_chat_completion(
    ChatCompletionRequest(
        tools=[
            Tool(
                function=Function(
                    name="get_current_weather",
                    description="获取当前天气",
                    parameters={
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "城市和州，例如：San Francisco, CA",
                            },
                            "format": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "温度单位。根据用户位置推断。",
                            },
                        },
                        "required": ["location", "format"],
                    },
                )
            )
        ],
        messages=[
            UserMessage(content="今天巴黎的天气怎么样"),
        ],
        model=model_name,
    )
)
tokens, text = tokenized.tokens, tokenized.text

# 计算标记数量
print(len(tokens))
```

## 学习并不停下，继续前行

完成本节课程后，请查看我们的[生成性AI学习课程](https://aka.ms/genai-collection?WT.mc_id=academic-105485-koreyst)继续提升您的生成性AI知识！