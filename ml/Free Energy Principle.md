# Free Energy Principle / Active Inference

## Overview

Free energy principle is a theoretical framework for understanding the brain, cognition, and behavior. It is based on the idea that the brain is an inference machine that minimizes the free energy of sensory signals by changing its internal states. This principle is inspired by the work of Karl Friston, a neuroscientist and theoretical physicist.

## Resources

### Books / Articles

https://direct.mit.edu/books/oa-monograph/5299/Active-InferenceThe-Free-Energy-Principle-in-Mind

https://github.com/ActiveInferenceInstitute/Parr_et_al_2022_ActInf_Textbook

https://link.springer.com/article/10.1007/s00422-018-0753-2

https://arxiv.org/abs/2201.06387 - The free energy principle made simpler but not too simple

https://arxiv.org/abs/1906.10184 - A free energy principle for a particular physics, Friston

https://philsci-archive.pitt.edu/18974/ - The math is not the territory: navigating the free energy
principle (Mel Andrews)

https://discovery.ucl.ac.uk/id/eprint/1570070/1/Friston_Active%20Inference%20Curiosity%20and%20Insight.pdf - Active Inference, Curiosity and Insight (Friston, Lin et al, 2017)

https://medium.com/@solopchuk/tutorial-on-active-inference-30edcf50f5dc - Tutorial on Active Inference

### Websites

https://en.wikipedia.org/wiki/Free_energy_principle

https://www.fil.ion.ucl.ac.uk/~karl/The%20free-energy%20principle%20-%20a%20rough%20guide%20to%20the%20brain.pdf

https://osf.io/preprints/psyarxiv/daf5n

### Active Inference Institute

https://www.activeinference.org/ - Active Inference Institute

https://welcome.activeinference.institute/

https://www.activeinference.institute/ (TODO: see readings)

https://coda.io/d/AII_do5M5iYau8H/Textbook-Group_suWRa5oa#_luQzKZu_ - Textbook Group

https://coda.io/d/ActInf-Textbook-Group_d4CkUI-iA_K/Chapters-Notes-Questions_suBGcJ6p#_lu5NHylY - Textbook notes

https://coda.io/d/AII_do5M5iYau8H/Courses_suUDddog#_lu8wn866 - Courses

https://github.com/ActiveInferenceInstitute/Active_Inference_Ontology

https://github.com/ActiveInferenceInstitute/ActiveBlockference

### Podcasts

Friston -free energy principle playlist: https://www.youtube.com/playlist?list=PLgK-3nhN7lDhQiLBgelOvn23I65b26GGU

https://www.youtube.com/watch?v=KkR24ieh5Ow - MLST #033 Karl Friston - The Free Energy Principle

https://www.youtube.com/watch?v=xKQ-F2-o8uM - MLST #67 Prof. KARL FRISTON 2.0

https://www.youtube.com/watch?v=V_VXOdf1NMw - MLST #106 Prof. KARL FRISTON 3.0

https://www.youtube.com/watch?v=NwzuibY5kUs - Lex Fridman #99 Karl Friston: Neuroscience and the Free Energy Principle | Lex Fridman Podcast #99

https://www.youtube.com/watch?v=TcFLQvz5uEg - Mindscape 87 | Karl Friston on Brains, Predictions, and Free Energy

https://www.youtube.com/watch?v=bk_xCikDUDQ - Dr. THOMAS PARR - Active Inference

https://www.youtube.com/watch?v=XohuuIi7Jt8 - MLDawn: Active inference explained with Prof. Karl Friston

https://www.youtube.com/@ActiveInference/playlists - Active Inference Institute

https://www.youtube.com/playlist?list=PLNm0u2n1IwdozjlqtVSAVC5_mNIpkHdub - Active Inference Insights

### Courses

https://www.youtube.com/playlist?list=PLNm0u2n1IwdqaxGwvFjQd9_KqOu3nlMae - Background & Context videos for Active Inference

https://www.youtube.com/watch?v=RpOrRw4EhTo&list=PLNm0u2n1Iwdq0UnnnnkUr446lUz00x6E7 - Physics as Information Processing

https://github.com/ActiveInferenceInstitute/ActiveInferenceCategoryTheory - Active Inference Category Theory

### People

#### Karl Friston

https://arxiv.org/search/?searchtype=author&query=Friston%2C+K

https://en.wikipedia.org/wiki/Karl_J._Friston

#### Thomas Parr

https://tejparr.github.io/Talks.html

https://www.youtube.com/watch?v=L_zDbFdyos8

---

### ChatGPT Summary

https://chatgpt.com/share/66e955e5-ea34-8004-836f-422d968ea338

The **Free Energy Principle (FEP)** is a theoretical framework that explains how biological systems, including humans, maintain a stable and adaptive state in an ever-changing and uncertain environment. It is based on principles from Bayesian inference, information theory, and thermodynamics, and applies broadly to any system that seeks to minimize surprise or uncertainty about its interactions with the world.

### Core Concepts:

1. **Organisms as Inference Machines**:
   Biological systems can be viewed as "inference machines" that try to model their environment. Instead of passively responding to stimuli, organisms maintain internal models (or beliefs) about the world and continuously update these models to minimize prediction errors.

2. **Free Energy**:
   Free energy, in this context, refers to a mathematical quantity borrowed from thermodynamics, but interpreted in terms of probabilistic inference. It is a bound on the "surprise" (a.k.a. negative log evidence) an organism experiences from sensory input. Minimizing free energy is equivalent to minimizing prediction errors between what the organism expects and what it senses.

3. **Active Inference**:
   Active Inference is a dynamic process where organisms reduce prediction error not just by updating internal beliefs (perception), but also by acting in the world (action) to bring about predicted states. For example, if you predict you’ll see a certain object when you turn your head, you can either update your beliefs if you don’t see it, or act by turning your head to confirm the prediction.

### Mathematical Formulation:

FEP uses Bayesian inference to model the brain’s behavior. It posits that the brain makes predictions about sensory input based on a prior model (beliefs) and updates these predictions by integrating new evidence (sensory input). In essence, the brain is constantly performing a form of approximate Bayesian inference.

The objective is to minimize **variational free energy**, which can be thought of as the difference between the brain's internal model and the actual sensory input:
\[ F = \text{KL}[q(x) || p(x | y)] \]
where:

- \( q(x) \) is the organism’s internal beliefs or model about the causes of sensory input.
- \( p(x | y) \) is the true posterior probability of those causes given sensory input \( y \).
- **KL** is the Kullback-Leibler divergence, which measures how far the internal model is from the true posterior.

Minimizing free energy essentially means adjusting the organism’s internal model to better reflect reality or acting to change the environment to fit the model’s predictions.

### Connection to Machine Learning:

- **Predictive Coding**: In machine learning terms, the FEP is akin to a hierarchical predictive coding system where higher-level beliefs generate predictions about lower-level sensory input. Errors between predicted and actual input propagate up the hierarchy to update beliefs.
- **Reinforcement Learning**: The active component of inference resembles reinforcement learning, where actions are chosen to minimize expected future prediction error.

### Biological Perspective:

- **Homeostasis**: The FEP offers a general explanation for how organisms maintain homeostasis. By minimizing free energy, organisms ensure that their internal states remain within viable limits, avoiding states of high surprise (i.e., stress or danger).
- **Perception and Action**: It unifies perception and action under the same principle. Perception is about reducing free energy through belief updates, while action is about reducing it by changing the external world.

### Summary:

In summary, the Free Energy Principle explains how living systems survive and adapt by continuously minimizing the discrepancy between their internal models and external sensory input. Active Inference extends this by suggesting that organisms don't just passively perceive the world but actively influence it to align with their predictions. This principle bridges neuroscience, biology, and machine learning by proposing that the brain functions as a Bayesian inference engine.
