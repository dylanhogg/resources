# LLM Study

## karpathy

✅ [Youtube: Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) (1hr) [Blog notes](https://ppaolo.substack.com/p/introduction-to-large-language-models-llms)

[State of GPT - Microsoft Build 2023](https://www.youtube.com/watch?v=bZQun8Y4L2A&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) (42m)
Learn about the training pipeline of GPT assistants like ChatGPT, from tokenization to pretraining, supervised finetuning, and Reinforcement Learning from Human Feedback (RLHF). Dive deeper into practical techniques and mental models for the effective use of these models, including prompting strategies, finetuning, the rapidly growing ecosystem of tools, and their future extensions.

[Youtube: Stanford CS25: V2 I Introduction to Transformers w/ Andrej Karpathy](https://www.youtube.com/watch?v=XfpMkf4rD6E) [Full CS25 playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM)

[Youtube: Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) (4hrs) [nanoGPT video code](https://github.com/karpathy/ng-video-lecture) [nanoGPT code](https://github.com/karpathy/nanoGPT)
We reproduce the GPT-2 (124M) from scratch. This video covers the whole process: First we build the GPT-2 network, then we optimize its training to be really fast, then we set up the training run following the GPT-2 and GPT-3 paper and their hyperparameters, then we hit run, and come back the next morning to see our results, and enjoy some amusing model generations. Keep in mind that in some places this video builds on the knowledge from earlier videos in the Zero to Hero Playlist (see my channel). You could also see this video as building my nanoGPT repo, which by the end is about 90% similar.

[Playlist: Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) (10 videos) [karpathy.ai/zero-to-hero.html](https://karpathy.ai/zero-to-hero.html)
- [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) [micrograd code](https://github.com/karpathy/micrograd) [micrograd notebooks](https://github.com/karpathy/nn-zero-to-hero/tree/master/lectures/micrograd)
This is the most step-by-step spelled-out explanation of backpropagation and training of neural networks. It only assumes basic knowledge of Python and a vague recollection of calculus from high school.
- [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) [makemore code](https://github.com/karpathy/makemore) [makemore notebooks](https://github.com/karpathy/nn-zero-to-hero/tree/master/lectures/makemore)
We implement a bigram character-level language model, which we will further complexify in followup videos into a modern Transformer language model, like GPT. In this video, the focus is on (1) introducing torch.Tensor and its subtleties and use in efficiently evaluating neural networks and (2) the overall framework of language modeling that includes model training, sampling, and the evaluation of a loss (e.g. the negative log likelihood for classification).
- [Building makemore Part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
We implement a multilayer perceptron (MLP) character-level language model. In this video we also introduce many basics of machine learning (e.g. model training, learning rate tuning, hyperparameters, evaluation, train/dev/test splits, under/overfitting, etc.).
- [Building makemore Part 3: Activations & Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
We dive into some of the internals of MLPs with multiple layers and scrutinize the statistics of the forward pass activations, backward pass gradients, and some of the pitfalls when they are improperly scaled. We also look at the typical diagnostic tools and visualizations you'd want to use to understand the health of your deep network. We learn why training deep neural nets can be fragile and introduce the first modern innovation that made doing so much easier: Batch Normalization. Residual connections and the Adam optimizer remain notable todos for later video.
- [Building makemore Part 4: Becoming a Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
We take the 2-layer MLP (with BatchNorm) from the previous video and backpropagate through it manually without using PyTorch autograd's loss.backward(): through the cross entropy loss, 2nd linear layer, tanh, batchnorm, 1st linear layer, and the embedding table. Along the way, we get a strong intuitive understanding about how gradients flow backwards through the compute graph and on the level of efficient Tensors, not just individual scalars like in micrograd. This helps build competence and intuition around how neural nets are optimized and sets you up to more confidently innovate on and debug modern neural networks.
- [Building makemore Part 5: Building a WaveNet](https://www.youtube.com/watch?v=t3YJ5hKiMQ0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
We take the 2-layer MLP from previous video and make it deeper with a tree-like structure, arriving at a convolutional neural network architecture similar to the WaveNet (2016) from DeepMind. In the WaveNet paper, the same hierarchical architecture is implemented more efficiently using causal dilated convolutions (not yet covered). Along the way we get a better sense of torch.nn and what it is and how it works under the hood, and what a typical deep learning development process looks like (a lot of reading of documentation, keeping track of multidimensional tensor shapes, moving between jupyter notebooks and repository code, ...).
- [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
We build a Generatively Pretrained Transformer (GPT), following the paper "Attention is All You Need" and OpenAI's GPT-2 / GPT-3. We talk about connections to ChatGPT, which has taken the world by storm. We watch GitHub Copilot, itself a GPT, help us write a GPT (meta :D!) . I recommend people watch the earlier makemore videos to get comfortable with the autoregressive language modeling framework and basics of tensors and PyTorch nn, which we take for granted in this video.
- [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
The Tokenizer is a necessary and pervasive component of Large Language Models (LLMs), where it translates between strings and tokens (text chunks). Tokenizers are a completely separate stage of the LLM pipeline: they have their own training sets, training algorithms (Byte Pair Encoding), and after training implement two fundamental functions: encode() from strings to tokens, and decode() back from tokens to strings. In this lecture we build from scratch the Tokenizer used in the GPT series from OpenAI. In the process, we will see that a lot of weird behaviors and problems of LLMs actually trace back to tokenization. We'll go through a number of these issues, discuss why tokenization is at fault, and why someone out there ideally finds a way to delete this stage entirely.


## 3b1b

[Neural networks and Transformers](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
- But what is a neural network? | Chapter 1, Deep learning
- Gradient descent, how neural networks learn | Chapter 2, Deep learning
- What is backpropagation really doing? | Chapter 3, Deep learning
- Backpropagation calculus | Chapter 4, Deep learning
- But what is a GPT? Visual intro to transformers | Chapter 5, Deep Learning
- Attention in transformers, visually explained | Chapter 6, Deep Learning
  
## rasbt

[PLaylist: LLMs](https://www.youtube.com/playlist?list=PLTKMiZHVd_2Licpov-ZK24j6oUnbhiPkm) (4 videos)
- Developing an LLM: Building, Training, Finetuning
- Understanding PyTorch Buffers
- Finetuning Open-Source LLMs
- Insights from Finetuning LLMs with Low-Rank Adaptation

[github.com/rasbt/LLM-workshop-2024](https://github.com/rasbt/LLM-workshop-2024)

## CS25: Transformers United V4

https://web.stanford.edu/class/cs25/

[Playlist: Stanford CS25 - Transformers United](https://www.youtube.com/playlist?list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM) (33 videos)

[Playlist: Stanford CS25 - Transformers United V3](https://www.youtube.com/playlist?list=PLNQo_x2EPWCkrhwatKK8t0q1HKad9o1Ye) (7 videos)

[Playlist: CS25 Transformers United 23](https://www.youtube.com/playlist?list=PLVVTN-yNn8rvEwlY8ClxDUWeVPVfdifYj)

## Blogs / Demos

https://amatria.in/blog/transformer-models-an-introduction-and-catalog-2d1e9039f376/

https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html

https://jalammar.github.io/illustrated-transformer/

https://poloclub.github.io/transformer-explainer/

https://bbycroft.net/llm

https://sebastianraschka.com/blog/2024/using-finetuning-transformers.html

https://sebastianraschka.com/blog/2023/llm-reading-list.html

https://github.com/naklecha/llama3-from-scratch

## Also see

https://blog.infocruncher.com/study/#ml

