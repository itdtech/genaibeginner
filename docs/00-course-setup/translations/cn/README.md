# 设置您的开发环境

我们使用[开发容器](https://containers.dev?WT.mc_id=academic-105485-koreyst)设置了此存储库和课程，该容器具有支持 Python3、.NET、Node.js 和 Java 开发的通用运行时。相关配置在存储库根目录下的 `.devcontainer/` 文件夹中的 `devcontainer.json` 文件中定义。

要激活开发容器，请在 [GitHub Codespaces](https://docs.github.com/en/codespaces/overview?WT.mc_id=academic-105485-koreyst)（用于云托管运行时）或 [Docker Desktop](https://docs.docker.com/desktop/?WT.mc_id=academic-105485-koreyst)（用于本地设备托管运行时）中启动它。阅读[此文档](https://code.visualstudio.com/docs/devcontainers/containers?WT.mc_id=academic-105485-koreyst)了解 VS Code 中的开发容器的工作原理。 

> [!提示]  
> 我们推荐使用 GitHub Codespaces 以最小的努力实现快速启动。它为个人账户提供了慷慨的[免费使用配额](https://docs.github.com/billing/managing-billing-for-github-codespaces/about-billing-for-github-codespaces#monthly-included-storage-and-core-hours-for-personal-accounts?WT.mc_id=academic-105485-koreyst)。配置[超时设置](https://docs.github.com/codespaces/setting-your-user-preferences/setting-your-timeout-period-for-github-codespaces?WT.mc_id=academic-105485-koreyst)以停止或删除不活动的代码空间，以最大化配额使用。

## 1. 执行任务

每节课将有 _可选_ 的任务，可能提供一种或多种编程语言，包括：Python、.NET/C#、Java 和 JavaScript/TypeScript。本节提供了与执行这些任务相关的一般指导。

### 1.1 Python 任务

Python 任务将作为应用程序（`.py` 文件）或 Jupyter notebook（`.ipynb` 文件）提供。
- 要运行 notebook， 在 Visual Studio Code 中打开它，然后点击 _选择内核_（右上角），选择显示的默认 Python 3 选项。现在可以 _全部运行_ 以执行 notebook。
- 要从命令行运行 Python 应用程序，请按照特定任务说明进行操作，确保选择正确的文件并提供所需的参数。

## 2. 配置提供商

任务 **可能** 还可以设置为通过支持的服务提供商（如 OpenAI、Azure 或 Hugging Face）在一个或多个大型语言模型(LLM)部署中工作。这些提供我们可以使用正确凭据（API 密钥或令牌）以编程方式访问的 _托管端点_（API）。在本课程中，我们讨论这些提供商：

- [OpenAI](https://platform.openai.com/docs/models?WT.mc_id=academic-105485-koreyst) 提供包括核心 GPT 系列在内的多样化模型。
- 专注于企业准备度的 [Azure OpenAI](https://learn.microsoft.com/azure/ai-services/openai/?WT.mc_id=academic-105485-koreyst) 提供 OpenAI 模型。
- 提供开源模型和推理服务器的 [Hugging Face](https://huggingface.co/docs/hub/index?WT.mc_id=academic-105485-koreyst)。

**您需要使用自己的账户进行这些练习**。任务是可选的，因此您可以根据兴趣选择设置一个、全部或没有任何提供商。注册的一些指导：

| 注册 | 费用 | API 密钥 | Playground | 注释 |
|:---|:---|:---|:---|:---|
| [OpenAI](https://platform.openai.com/signup?WT.mc_id=academic-105485-koreyst)| [定价](https://openai.com/pricing#language-models?WT.mc_id=academic-105485-koreyst)| [基于项目](https://platform.openai.com/api-keys?WT.mc_id=academic-105485-koreyst) | [无代码，Web](https://platform.openai.com/playground?WT.mc_id=academic-105485-koreyst) | 多种模型可用 |
| [Azure](https://aka.ms/azure/free?WT.mc_id=academic-105485-koreyst)| [定价](https://azure.microsoft.com/pricing/details/cognitive-services/openai-service/?WT.mc_id=academic-105485-koreyst)| [SDK 快速入门](https://learn.microsoft.com/azure/ai-services/openai/quickstart?WT.mc_id=academic-105485-koreyst)| [Studio 快速入门](https://learn.microsoft.com/azure/ai-services/openai/quickstart?WT.mc_id=academic-105485-koreyst) |  [必须提前申请访问权限](https://learn.microsoft.com/azure/ai-services/openai/?WT.mc_id=academic-105485-koreyst)|
| [Hugging Face](https://huggingface.co/join?WT.mc_id=academic-105485-koreyst) | [定价](https://huggingface.co/pricing) | [访问令牌](https://huggingface.co/docs/hub/security-tokens?WT.mc_id=academic-105485-koreyst) | [Hugging Chat](https://huggingface.co/chat/?WT.mc_id=academic-105485-koreyst)| [Hugging Chat 的模型有限](https://huggingface.co/chat/models?WT.mc_id=academic-105485-koreyst) |
| | | | | 

按照下面的指示将此存储库配置为与不同的提供商一起使用。需要特定提供商的任务将包含以下标签之一：

 - `aoai` - 需要 Azure OpenAI 端点和密钥
 - `oai` - 需要 OpenAI 端点和密钥
 - `hf` - 需要 Hugging Face 令牌

您可以配置一个、没有或全部提供商。相关的任务将在缺少凭据时简单地出错。

### 2.1 创建 `.env` 文件

我们假设您已经阅读了上述指导并注册了相关提供商，并获得了所需的认证凭据（API_KEY 或令牌）。在 Azure OpenAI 的情况下，我们假设您还具有有效的 Azure OpenAI 服务部署（端点），其中至少有一个 GPT 模型部署用于聊天完成。

下一步是配置您的**本地环境变量**，如下所示：

1. 在根文件夹中查找一个 `.env.copy` 文件，里面应该有如下内容：

   ```bash
   # OpenAI 提供商
   OPENAI_API_KEY='<在此添加您的 OpenAI API 密钥>'

   ## Azure OpenAI
   AZURE_OPENAI_API_VERSION='2024-02-01' # 默认已设置！
   AZURE_OPENAI_API_KEY='<在此添加您的 AOAI 密钥>'
   AZURE_OPENAI_ENDPOINT='<在此添加您的 AOIA 服务端点>'
   AZURE_OPENAI_DEPLOYMENT='<在此添加您的聊天完成模型名称>' 
   AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT='<在此添加您的嵌入模型名称>'

   ## Hugging Face
   HUGGING_FACE_API_KEY='<在此添加您的 HuggingFace API 或令牌>'
   ```

2. 使用下面的命令将该文件复制到 `.env`。这个文件被 _gitignore_ 了，确保秘密信息安全。

   ```bash
   cp .env.copy .env
   ```

3. 按照下一节中的描述填写这些值（替换 `=` 右侧的占位符）。

4.（可选）如果您使用 GitHub Codespaces，您可以将环境变量保存为与此存储库相关的 _Codespaces secret_。在这种情况下，您不需要设置本地 .env 文件。**但是，请注意，此选项仅在您使用 GitHub Codespaces 时有效。** 如果您使用 Docker Desktop，则仍需设置 .env 文件。

### 2.2 填写 `.env` 文件

让我们快速了解变量名称，以理解它们代表的含义：

| 变量  | 描述  |
| :--- | :--- |
| HUGGING_FACE_API_KEY | 这是您在个人资料中设置的用户访问令牌 |
| OPENAI_API_KEY | 这是非 Azure OpenAI 端点使用该服务的授权密钥 |
| AZURE_OPENAI_API_KEY | 这是使用该服务的授权密钥 |
| AZURE_OPENAI_ENDPOINT | 这是 Azure OpenAI 资源的部署端点 |
| AZURE_OPENAI_DEPLOYMENT | 这是 _文本生成_ 模型部署端点 |
| AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT | 这是 _文本嵌入_ 模型部署端点 |
| | |

注意：最后两个 Azure OpenAI 变量分别表示聊天完成（文本生成）和向量搜索（嵌入）的默认模型。设置它们的说明将在相关任务中定义。

### 2.3 在 Azure 门户中配置 Azure

Azure OpenAI 端点和密钥值将会在 [Azure 门户](https://portal.azure.com?WT.mc_id=academic-105485-koreyst) 中找到，让我们从这里开始。

1. 访问 [Azure 门户](https://portal.azure.com?WT.mc_id=academic-105485-koreyst)
2. 点击侧边栏（左侧菜单）中的 **密钥和端点** 选项。
3. 点击 **显示密钥** - 您应该会看到以下内容：KEY 1、KEY 2 和端点。
4. 使用 KEY 1 的值填入 AZURE_OPENAI_API_KEY
5. 使用端点的值填入 AZURE_OPENAI_ENDPOINT

接下来，我们需要特定模型的端点。

1. 点击 Azure OpenAI 资源的侧边栏（左侧菜单）中的 **模型部署** 选项。
2. 在目标页面，点击 **管理部署**

这将把您带到 Azure OpenAI Studio 网站，我们将在下面描述的步骤中找到其他值。

### 2.4 在 Studio 中配置 Azure

1. 按上述描述从您的资源导航到 [Azure OpenAI Studio](https://oai.azure.com?WT.mc_id=academic-105485-koreyst)。
2. 点击侧边栏（左侧）的 **部署** 标签以查看当前部署的模型。
3. 如果没有部署所需模型，请使用 **创建新部署** 进行部署。
4. 您将需要一个 _文本生成_ 模型 - 我们推荐：**gpt-35-turbo**
5. 您将需要一个 _文本嵌入_ 模型 - 我们推荐：**text-embedding-ada-002**

现在更新环境变量以反映使用的 _部署名称_。这通常与模型名称相同，除非您明确更改它。因此，例如，您可能有：

```bash
AZURE_OPENAI_DEPLOYMENT='gpt-35-turbo'
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT='text-embedding-ada-002'
```

**完成后不要忘记保存 .env 文件**。您现在可以退出文件并返回到 notebook 的运行指示。

### 2.5 从个人资料中配置 OpenAI

您的 OpenAI API 密钥可以在您的 [OpenAI 账户](https://platform.openai.com/api-keys?WT.mc_id=academic-105485-koreyst) 中找到。如果没有，您可以注册一个账户并创建 API 密钥。一旦有了密钥，您可以在 `.env` 文件中填写 `OPENAI_API_KEY` 变量。

### 2.6 从个人资料中配置 Hugging Face

您的 Hugging Face 令牌可以在个人资料中的 [访问令牌](https://huggingface.co/settings/tokens?WT.mc_id=academic-105485-koreyst) 下找到。不要公开发布或分享这些令牌。相反，为此项目使用创建一个新令牌，并将其复制到 `.env` 文件中的 `HUGGING_FACE_API_KEY` 变量下。_注意：_ 这实际上不是 API 密钥，但用于认证，因此为了统一，我们保留了这个命名约定。