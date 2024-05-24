# Citing calibrated-explanations


If you use `calibrated-explanations` for a scientific publication, you are kindly requested to cite one of the following papers:

- [Löfström, H](https://github.com/Moffran)., [Löfström, T](https://github.com/tuvelofstrom)., Johansson, U., and Sönströd, C. (2024). [Calibrated Explanations: with Uncertainty Information and Counterfactuals](https://doi.org/10.1016/j.eswa.2024.123154). Expert Systems with Applications, 1-27.

The extensions for regression are introduced in the paper:

- [Löfström, T](https://github.com/tuvelofstrom)., [Löfström, H](https://github.com/Moffran)., Johansson, U., Sönströd, C., and [Matela, R](https://github.com/rudymatela). [Calibrated Explanations for Regression](https://arxiv.org/abs/2308.16245). arXiv preprint arXiv:2308.16245.

The paper that originated the idea of `calibrated-explanations` is:

- [Löfström, H.](https://github.com/Moffran), [Löfström, T.](https://github.com/tuvelofstrom), Johansson, U., & Sönströd, C. (2023). [Investigating the impact of calibration on the quality of explanations](https://link.springer.com/article/10.1007/s10472-023-09837-2). Annals of Mathematics and Artificial Intelligence, 1-18. [Code and results](https://github.com/tuvelofstrom/calibrating-explanations).

Bibtex entry for the original paper:

```bibtex
@article{lofstrom2024ce_classification,
	title = 	{Calibrated explanations: With uncertainty information and counterfactuals},
	journal = 	{Expert Systems with Applications},
	pages = 	{123154},
	year = 		{2024},
	issn = 		{0957-4174},
	doi = 		{https://doi.org/10.1016/j.eswa.2024.123154},
	url = 		{https://www.sciencedirect.com/science/article/pii/S0957417424000198},
	author = 	{Helena Löfström and Tuwe Löfström and Ulf Johansson and Cecilia Sönströd},
	keywords = 	{Explainable AI, Feature importance, Calibrated explanations, Venn-Abers, Uncertainty quantification, Counterfactual explanations},
	abstract = 	{While local explanations for AI models can offer insights into individual predictions, such as feature importance, they are plagued by issues like instability. The unreliability of feature weights, often skewed due to poorly calibrated ML models, deepens these challenges. Moreover, the critical aspect of feature importance uncertainty remains mostly unaddressed in Explainable AI (XAI). The novel feature importance explanation method presented in this paper, called Calibrated Explanations (CE), is designed to tackle these issues head-on. Built on the foundation of Venn-Abers, CE not only calibrates the underlying model but also delivers reliable feature importance explanations with an exact definition of the feature weights. CE goes beyond conventional solutions by addressing output uncertainty. It accomplishes this by providing uncertainty quantification for both feature weights and the model’s probability estimates. Additionally, CE is model-agnostic, featuring easily comprehensible conditional rules and the ability to generate counterfactual explanations with embedded uncertainty quantification. Results from an evaluation with 25 benchmark datasets underscore the efficacy of CE, making it stand as a fast, reliable, stable, and robust solution.}
}
```
Bibtex entry for the regression paper:

```bibtex
@misc{lofstrom2023ce_regression,
      title = 	      	{Calibrated Explanations for Regression},
      author =          {L\"ofstr\"om, Tuwe and L\"ofstr\"om, Helena and Johansson, Ulf and S\"onstr\"od, Cecilia and Matela, Rudy},
      year =            {2023},
      eprint =          {2308.16245},
      archivePrefix =   {arXiv},
      primaryClass =    {cs.LG}
}
```

To cite this software, use the following bibtex entry:

```bibtex
@software{lofstrom2024ce_repository,
	author = 	{Löfström, Helena and Löfström, Tuwe and Johansson, Ulf and Sönströd, Cecilia and Matela, Rudy},
	license = 	{BSD-3-Clause},
	title = 	{Calibrated Explanations},
	url = 		{https://github.com/Moffran/calibrated_explanations},
	version = 	{v0.3.2},
	month = 	April,
	year = 		{2024}
}
```
