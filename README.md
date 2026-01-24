
## Epistemic Stance Classification and Evaluation Experimentation

This repository presents my work in the [Apart Research Hackathon on AI Manipulation](https://apartresearch.com/sprints/ai-manipulation-hackathon-2026-01-09-to-2026-01-11). The goal of my work was to build an evaluation to measure the capability and tendency of LLMs to identify and adapt to the epistemic stance of the user they are interacting with. If this is the case, then this points towards promising further research in theory of mind, persuasion, and manipulation evaluations based in epistemological theories. In other words, if LLMs can and do use the epistemic stance (or other epistemic models) of the user they are interacting with, how and when do they do this, and does this make their attempts at persuasion more or less successful?

Rather than focusing on technical setup and use of this repository, think of this read me as an informal documentary story of my research journey that explains the process, experiments, and learnings gained from fine-tuning a model for epistemic stance detection.

## Background

#### Hackathon

The Apart Research Hackathon set out its mission as this:

>    The line between authentic interaction and strategic manipulation is disappearing as AI systems master deception, sycophancy, sandbagging, and psychological exploitation at scale. Our ability to detect, measure, and counter these behaviors is dangerously underdeveloped.

With potential projects falling under these (and other) categories:

>    **Manipulation benchmarks** that measure persuasive capabilities, deception, and strategic behavior with real ecological validity
>    **Detection systems** that identify sycophancy, reward hacking, sandbagging, and dark patterns in deployed AI systems

The tool I built, a fully fine-tuned [model](https://huggingface.co/johnclund/epistemic-stance-analyzer) capable of identifying and analyzing the use of epistemic stance in dialogue has applications in these areas by measuring the ability of LLMs to identify and use epistemic-based persuasion in benchmarks and real-world interactions.

#### Research Priors

The primary source for my understanding of epistemic stance is (Kuhn et al., 2000). Kuhn identifies three primary stances:

- **Absolutist**: Knowledge is CERTAIN. Claims presented as objective truth without qualification. Dismisses opposing views.
- **Multiplist**: Knowledge is SUBJECTIVE. All opinions equally valid. Avoids evaluation.
- **Evaluativist**: Knowledge is UNCERTAIN but some claims have MORE MERIT. Weighs evidence, engages counterarguments, shows calibrated confidence.

In addition, I drew from other sources (Nussbaum et al. 2008; Hofer & Pintrich 1997; Wu et al. 2025) to understand more nuance in the these areas as regards to reasoning and belief.

As mentioned above, epistemic stance is a gateway to broader understanding of the potential capabilities and alignment of epistemology in LLMs, leading to further study in:

- **Adaptive Persuasion**: Can and do LLMs adapt their dialogue to appeal to certain reasoning/epistemic frameworks held by the humans using them?
- **Theory of Mind**: How comprehensive is this understanding, and what models of epistemology are exhibited?
- **Evaluation Gaps**: Are there currently gaps in evaluations and benchmarks that are missing more nuanced and strategic forms of manipulation and persuasion?

## Research Process

#### Dataset Selection and Preparation

I initially wanted a dataset that would represent the target evaluations well, i.e. interactions between humans and LLMs, so I started by looking at [WildChat](https://huggingface.co/datasets/allenai/WildChat) and [OpenAssistant](https://huggingface.co/datasets/OpenAssistant/oasst1). After reviewing this, I found that the majority of the interactions with LLMs were transactional and in they did include reasoning / epistemic markers, they were high likely to be absolutist as the LLM presented knowledge to the user as facts. While this is useful to know for evaluations, it didn't help me to build a dataset that would be useful for training a model to recognize different expressions of epistemic stance.

I then looked at the [PERSUADE](https://github.com/scrosseye/PERSUADE_corpus) dataset, which is a corpus of grade school persuasive / argumentative essays. As you might expect, this also leaned heavily absolutist, with some evaluativist markers showing up in small sizes. Also, the writing tended to be similar in style and tone, representing only a small example of human dialogue.

Finally, I turned to the Reddit [dataset](https://huggingface.co/datasets/HuggingFaceGECLM/REDDIT_comments). There are number of large datasets drawing from different subreddits. This proved to be a valuable resource for finding different styles of thinking and reasoning, though of course, it is itself still a small and specific set of dialogue on the internet. The main challenge was to find a significant number of truly multiplist thinking in any of these datasets.

This was one of the most interesting findings of my research: that multiplist reasoning appears to be very rare on the internet, at least in a cursory examination. Another research project could attempt to find more examples of multiplist thinking (using the classifier and analyzer models that I've trained perhaps) and explore the results in terms of psychology, sociology and epistemology.

The subreddit that had the most multiplist examples was r/socialskills, which makes some sense, as these discussions lend themselves to reasoning based on subjective preference.

#### Dataset Preparation

My approach to labeling the data was to create a relatively small set (~3000) of high-quality (gold) samples using Claude Sonnet, and then using those samples to train a Longformer classifier [model](https://huggingface.co/johnclund/epistemic-stance-classifier). This model would then classify about 25,000 (silver) samples that I would use to fine-tune a larger instruct model.

I first filtered the dataset to eliminate low-quality comments with little to no epistemic/argumentative value. I decided to filter the comments on a fairly large set of words that are correlated with reasoning and argumentation. In order to get enough multiplist samples in each of these datasets, I had to add another set of filters that selected for strings of words that matched examples of multiplist reasoning. This, of course, biased the data coming out of the dataset, but I was running out of time and compute, so I had to move forward. It would be very useful to spend a good deal more time searching for high-quality datasets that exhibit each of the different stances (coming from different sources) in order to get a truly balanced dataset for training.

In the end, I had to manually rebalance the dataset by weighting the multiplist examples more heavily in the training run. And I did end up including a ~3000 sample subset from WildChat selected for reasoning markers to ensure I had LLM-generated dialogue as well.

#### Model Training

The Hackathon provided $400 in compute budget from [Lambda](https://lambda.ai/) Cloud, and I wanted to make as much use of that budget as possible. I debated the pros and cons of large and smaller models and finally settled on [Mistral-Small-24B-Instruct-2501](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501). Claude estimated that this would use one-third to one-half of my training budget, which I felt would be a good outcome and still leave room for inference to run some initial evals.

I wanted to use a larger model for a couple reasons. First, I have a decently well-trained classifier model already, so I didn't just want another small model that could only classify dialogue. And second, I recognize that the more nuanced and strategic aspect of this research would require a more capable model for analysis.

Completing a full fine-tune of that model took a lot of work as I learned the relationships between models / hardware / environments. I went through a couple different 24B models, learning that the cutting edge models are not well supported in existing workflows, and that cutting edge hardware (B100s) are also not well supported. Eventually I dialed back to a more common hardware / model setup and things progressed smoothly from that point on.

But the entire [process](https://wandb.ai/modern-masters/epistemic-stance) ended up using most of my compute budget. By the end of this, I felt a little disappointed that I didn't have more room in my budget left over, and felt that I should have started with a smaller (7B) model in order to develop and test the entire pipeline, then switch to a larger model if it was needed.

#### Training Results

**Classifier Model**

Best Run (Longformer, 6 epochs, focal γ=2.0, lr=2e-5)

| Metric                               | Score  |
| ------------------------------------ | ------ |
| **Test Accuracy**                    | 0.7651 |
| **Test F1-macro**                    | 0.6272 |
| **Test F1-weighted**                 | 0.7574 |
| **Test F1-evaluativist**             | 0.8354 |
| **Test F1-absolutist**               | 0.6294 |
| **Test F1-multiplist**               | 0.4167 |
| **Expected Calibration Error (ECE)** | 0.0943 |

**Key Observations**

- **Evaluativist detection** was strongest (F1 0.8354), reflecting the majority class in the dataset.
- **Multiplist F1** (0.4167) was challenging due to limited samples (n=15 in test set), with recall particularly low (0.33).
- **Calibration (ECE 0.0943)** was reasonable, making confidence scores usable for silver labeling.

**Data Statistics**

- **Total labeled samples:** 2,806 (no missing values)
- **High-confidence labels:** 86.5%
- **Multiplist samples:** 149 (5.3% of total), with 99% high confidence but lower reasoning quality (aligned with theory).

---

**24B Model (Mistral Small 24B Instruct 2501) Final Training Run**

|Metric|Score|
|---|---|
|**Accuracy**|0.9300|
|**F1 Macro**|0.7865|
|**F1 Weighted**|0.9306|
|**Eval Loss**|0.025|

**Per-Class Metrics (100-sample evaluation)**

|Class|Precision|Recall|F1-score|Support|
|---|---|---|---|---|
|**Absolutist**|0.86|0.96|0.91|26|
|**Evaluativist**|0.97|0.93|0.95|72|
|**Multiplist**|0.50|0.50|0.50|2|

**Key Observations**

- **Overall accuracy (93%)** and **F1-weighted (93.06%)** were excellent, reflecting strong performance on the majority classes.
- **Evaluativist F1 (0.95)** was the highest, consistent with the classifier results.
- **Multiplist F1 (0.50)** was limited by the small sample size (n=2 in the eval set), but precision/recall were balanced.
- **Eval loss (0.025)** was very low, indicating the model learned the task effectively.

**Training Details**

- **Total steps:** 583
- **Final checkpoint:** `checkpoint-583`
- **Context window:** 32k
- **Optimization:** Liger kernel (rope, rms_norm, swiglu, fused_linear_cross_entropy)

## Summary

**What did I learn about epistemic stance classification?**

The most striking finding was how rare genuine multiplist reasoning is, both in human internet discourse and in LLM outputs. In the Reddit data, multiplist represented only about 5% of samples, and in LLM outputs (WildChat), it dropped to just 0.7%. This suggests that RLHF-trained models are optimized away from "I can't tell you what to do" responses toward either committing to positions with reasoning (evaluativist) or stating things as facts (absolutist).

The distinction between "deceptive multiplists" and genuine multiplism proved challenging. Many responses that appear multiplist on the surface ("it depends," "everyone is different") are actually evaluativist once you look at whether they weigh options or engage with trade-offs. True multiplism involves refusing to evaluate—tolerating inconsistency and treating all positions as equally valid without judgment. This is surprisingly rare in advice-giving contexts.

**What would I do differently next time?**

1. **Start with a smaller model.** I spent most of my compute budget wrestling with cutting-edge models (Magistral-Small-2509) and hardware (B100s) that weren't well-supported in existing workflows. Starting with a well-documented 7B model would have let me validate the entire pipeline faster, then scale up if needed.
2. **Invest more time in dataset diversity.** The r/socialskills dataset was a significant improvement over PERSUADE for multiplist examples, but still represents a narrow slice of human dialogue. Future work should draw from multiple subreddits, other platforms, and perhaps educational or philosophical discussions where multiplist thinking might be more prevalent.
3. **Build domain adaptation into the plan from the start.** The gap between human Reddit posts and LLM outputs was noticeable. Adding WildChat data late in the process helped, but a more principled approach to multi-domain training would improve generalization.

**Open questions and future directions:**

- **Adaptive persuasion research:** Can LLMs detect a user's epistemic stance and adjust their communication strategy accordingly? The classifier I built could serve as a probe to measure this.
- **The sociology of multiplism:** Why is multiplist reasoning so rare online? Is it suppressed by platform dynamics, or is it genuinely uncommon in human cognition around contested topics?
- **Evaluativist mimicry:** RLHF-trained models appear systematically evaluativist in form. But does this reflect genuine epistemic virtue, or is it performative hedging that doesn't correspond to actual uncertainty calibration?
- **Cross-domain generalization:** How well does the classifier perform on academic writing, news articles, or political discourse versus social media advice?

## Resources and References

**Research Papers:**

- Kuhn, D., Cheney, R., & Weinstock, M. (2000). The development of epistemological understanding. _Cognitive Development, 15_(3), 309-328.
- Nussbaum, E. M., Sinatra, G. M., & Poliquin, A. (2008). Role of epistemic beliefs and scientific argumentation in science learning. _International Journal of Science Education, 30_(15), 1977-1999.
- Hofer, B. K., & Pintrich, P. R. (1997). The development of epistemological theories: Beliefs about knowledge and knowing and their relation to learning. _Review of Educational Research, 67_(1), 88-140.
- Wu, S. et al. (2025). Strengthening human epistemic agency in the symbiotic learning partnership with generative AI.

**Datasets:**

- [Reddit Comments Dataset](https://huggingface.co/datasets/HuggingFaceGECLM/REDDIT_comments) - Source for r/socialskills samples
- [WildChat](https://huggingface.co/datasets/allenai/WildChat) - LLM dialogue data for domain adaptation
- [PERSUADE Corpus](https://github.com/scrosseye/PERSUADE_corpus) - Initial exploration (not used in final model)

**Models:**

- Fine-tuned model: [johnclund/epistemic-stance-analyzer](https://huggingface.co/johnclund/epistemic-stance-analyzer)
- Classifier model: [johnclund/epistemic-stance-classifier](https://huggingface.co/johnclund/epistemic-stance-classifier)
- Base model: [Mistral-Small-24B-Instruct-2501](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501)

**Tools:**

- [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) - Fine-tuning framework
- [Lambda Cloud](https://lambda.ai/) - GPU compute
- [Weights & Biases](https://wandb.ai/modern-masters/epistemic-stance) - Training monitoring

**Acknowledgments:**

- [Apart Research](https://apartresearch.com/) for organizing the AI Manipulation Hackathon and providing the compute budget
- Claude (Anthropic) for assistance with labeling, code development, and debugging throughout the project

## License and Contact

**License:** This work is released under the MIT License. The fine-tuned models inherit licensing terms from their base models (Mistral Small is Apache 2.0).

**Contact:**

- GitHub: [johnlund](https://github.com/johnlund)
- Hugging Face: [johnclund](https://huggingface.co/johnclund)

For questions about this research or collaboration inquiries, please open an issue on the repository or reach out via Hugging Face.
