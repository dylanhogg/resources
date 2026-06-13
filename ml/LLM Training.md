# AI Training

## Overview

Resources for LLM training.

## Resources

- 2026 Sebastian Raschka: [LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/)
- 2026 Alex Wa: [Frontier model training methodologies](https://djdumpling.github.io/2026/01/31/frontier_training.html)
- 2026 Tom Aarsen: [Training and Finetuning Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-multimodal-sentence-transformers)
- 2025 Deepmind: [How to Scale Your Model A Systems View of LLMs on TPUs](https://jax-ml.github.io/scaling-book/index)
- 2025 HuggingFace: [The Smol Training Playbook: The Secrets to Building World-Class LLMs](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook)
- 2025 HuggingFace: [Tricks from OpenAI gpt-oss YOU 🫵 can use with transformers](https://huggingface.co/blog/faster-transformers)
- 2025 Meta: [LLaMA Cookbook: Finetuning Llama](https://github.com/meta-llama/llama-cookbook/tree/main/getting-started/finetuning)
- 2024 Google: [A playbook for systematically maximizing the performance of deep learning models.](https://github.com/dylanhogg/google_tuning_playbook)

## Libraries

- https://www.awesomepython.org/?q=fine-tuning
- https://www.awesomepython.org/?q=quantization
- https://www.awesomepython.org/?q=training

- LLaMA-Factory:
  - https://www.awesomepython.org/?q=llama-factory
  - https://github.com/hiyouga/LLaMA-Factory
  - https://www.datacamp.com/tutorial/llama-factory-web-ui-guide-fine-tuning-llms
  - https://arxiv.org/abs/2403.13372

- Unsloth:
  - https://www.awesomepython.org/?q=unsloth
  - https://github.com/unslothai/unsloth
  - https://www.datacamp.com/tutorial/unsloth-web-ui-guide-fine-tuning-llms

- Pytorch
  - https://mlechner.substack.com/p/why-we-started-with-jax-but-moved

- Nanochat (simplest experimental harness for training LLMs.)
  - https://github.com/karpathy/nanochat

- TODO: more

## Code / Notebooks

- https://colab.research.google.com/github/arman-bd/guppylm/blob/main/train_guppylm.ipynb

## Theory

## Physics of Language Models

Physics of Language Models is a research framework and emerging field of study, pioneered by Zeyuan Allen-Zhu and collaborators, that seeks a scientific, principled understanding of how and why large language models (LLMs) behave the way they do — analogous to how physics explains the natural world with universal laws rather than surface-level observations.

https://physics.allen-zhu.com/home

https://www.youtube.com/playlist?list=PLIZhMKKbVX6JmdngPRKvAS4u4L97odbGp

https://arxiv.org/search/?searchtype=all&query=Allen-Zhu+Physics+of+Language+Models&abstracts=show&size=50&order=-submitted_date

- Physics of Language Models: Part 1, Learning Hierarchical Language Structures

Summary: Shows that transformer-based models can learn hierarchical context-free grammars and that their hidden states and attention patterns reflect dynamic programming-like structure understanding.
[https://arxiv.org/abs/2305.13673](https://arxiv.org/abs/2305.13673)

- Physics of Language Models: Part 2.1, Grade-School Math and the Hidden Reasoning Process

Summary: Investigates how language models solve grade-school math, identifying hidden model processes and distinguishing genuine reasoning from memorization.
[https://arxiv.org/abs/2407.20311](https://arxiv.org/abs/2407.20311)

- Physics of Language Models: Part 2.2, How to Learn From Mistakes on Grade-School Math Problems

Summary: Studies how incorporating _error-correction data_ into pretraining helps language models improve reasoning accuracy directly, without multi-round prompting.
[https://arxiv.org/abs/2408.16293](https://arxiv.org/abs/2408.16293)

- Physics of Language Models: Part 3.1, Knowledge Storage and Extraction

Summary: Shows that reliable knowledge extraction by LLMs depends on diversity in pretraining data and that knowledge must be sufficiently augmented during training to be practically retrievable.
[https://arxiv.org/abs/2309.14316](https://arxiv.org/abs/2309.14316)

- Physics of Language Models: Part 3.2, Knowledge Manipulation

Summary: Demonstrates that while LLMs can retrieve stored knowledge well, they struggle with basic manipulation tasks like classification or comparison unless augmented with chain-of-thought mechanisms.
[https://arxiv.org/abs/2309.14402](https://arxiv.org/abs/2309.14402)

- Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws

Summary: Establishes _scaling laws_ for LLM knowledge storage, finding a ~2 bits of factual knowledge capacity per parameter and how architecture, data, and training affect it.
[https://arxiv.org/abs/2404.05405](https://arxiv.org/abs/2404.05405)

- Physics of Language Models: Part 4.1, Architecture Design and the Magic of Canon Layers

Summary: Introduces _Canon layers_ — lightweight components that improve information flow in sequence models — and shows how they enhance reasoning, knowledge manipulation, and architectures through controlled synthetic pretraining tasks.
[https://arxiv.org/abs/2512.17351](https://arxiv.org/abs/2512.17351)
