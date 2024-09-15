771005-0316# Citing calibrated-explanations


If you use `calibrated-explanations` for a scientific publication, you are kindly requested to cite one of the following papers:

- [Löfström, H](https://github.com/Moffran). (2023). [Trustworthy explanations: Improved decision support through well-calibrated uncertainty quantification](https://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1810440&dswid=6197) (Doctoral dissertation, Jönköping University, Jönköping International Business School).
- [Löfström, H](https://github.com/Moffran)., [Löfström, T](https://github.com/tuvelofstrom)., Johansson, U., and Sönströd, C. (2024). [Calibrated Explanations: with Uncertainty Information and Counterfactuals](https://doi.org/10.1016/j.eswa.2024.123154). Expert Systems with Applications, 1-27.
- [Löfström, T](https://github.com/tuvelofstrom)., [Löfström, H](https://github.com/Moffran)., Johansson, U., Sönströd, C., and [Matela, R](https://github.com/rudymatela). [Calibrated Explanations for Regression](https://arxiv.org/abs/2308.16245). arXiv preprint arXiv:2308.16245. Accepted to Machine Learning. In press.
-  [Löfström, H](https://github.com/Moffran)., [Löfström, T](https://github.com/tuvelofstrom). (2024). [Conditional Calibrated Explanations: Finding a Path Between Bias and Uncertainty](https://doi.org/10.1007/978-3-031-63787-2_17). In: Longo, L., Lapuschkin, S., Seifert, C. (eds) Explainable Artificial Intelligence. xAI 2024. Communications in Computer and Information Science, vol 2153. Springer, Cham.
- [Löfström, T](https://github.com/tuvelofstrom)., [Löfström, H](https://github.com/Moffran)., Johansson, U. (2024). [Calibrated Explanations for Multi-class](https://raw.githubusercontent.com/mlresearch/v230/main/assets/lofstrom24a/lofstrom24a.pdf). <i>Proceedings of the Thirteenth Workshop on Conformal and Probabilistic Prediction and Applications</i>, in <i>Proceedings of Machine Learning Research</i>, PMLR 230:175-194. [Presentation](https://copa-conference.com/presentations/Lofstrom.pdf)

The paper that originated the idea of `calibrated-explanations` is:

- [Löfström, H.](https://github.com/Moffran), [Löfström, T.](https://github.com/tuvelofstrom), Johansson, U., & Sönströd, C. (2023). [Investigating the impact of calibration on the quality of explanations](https://link.springer.com/article/10.1007/s10472-023-09837-2). Annals of Mathematics and Artificial Intelligence, 1-18. [Code and results](https://github.com/tuvelofstrom/calibrating-explanations).

## Bibtex Entries

Bibtex for Helena Löfström's dissertation:

```bibtex
@phdthesis{lofstrom2023dissertation,
	author = {L{\"o}fstr{\"o}m, Helena},
	institution = {Jönköping University, JIBS, Informatics},
	pages = {72},
	publisher = {Jönköping University, Jönköping International Business School},
	school = {, JIBS, Informatics},
	title = {Trustworthy explanations : Improved decision support through well-calibrated uncertainty quantification},
	series = {JIBS Dissertation Series},
	ISSN = {1403-0470},
	number = {159},
	keywords = {Explainable Artificial Intelligence, Interpretable Machine Learning, Decision Support Systems, Uncertainty Estimation, Explanation Methods},
	abstract = {The use of Artificial Intelligence (AI) has transformed fields like disease diagnosis and defence. Utilising sophisticated Machine Learning (ML) models, AI predicts future events based on historical data, introducing complexity that challenges understanding and decision-making. Previous research emphasizes users’ difficulty discerning when to trust predictions due to model complexity, underscoring addressing model complexity and providing transparent explanations as pivotal for facilitating high-quality decisions. Many ML models offer probability estimates for predictions, commonly used in methods providing explanations to guide users on prediction confidence. However, these probabilities often do not accurately reflect the actual distribution in the data, leading to potential user misinterpretation of prediction trustworthiness. Additionally, most explanation methods fail to convey whether the model’s probability is linked to any uncertainty, further diminishing the reliability of the explanations. Evaluating the quality of explanations for decision support is challenging, and although highlighted as essential in research, there are no benchmark criteria for comparative evaluations. This thesis introduces an innovative explanation method that generates reliable explanations, incorporating uncertainty information supporting users in determining when to trust the model’s predictions. The thesis also outlines strategies for evaluating explanation quality and facilitating comparative evaluations. Through empirical evaluations and user studies, the thesis provides practical insights to support decision-making utilising complex ML models. },
	ISBN = {978-91-7914-031-1},
	ISBN = {978-91-7914-032-8},
	year = {2023}
}
```

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
Bibtex for the conditional paper:

```bibtex
@InProceedings{lofstrom2024ce_conditional,
	author="L{\"o}fstr{\"o}m, Helena and L{\"o}fstr{\"o}m, Tuwe",
	editor="Longo, Luca and Lapuschkin, Sebastian and Seifert, Christin",
	title="Conditional Calibrated Explanations: Finding a Path Between Bias and Uncertainty",
	booktitle="Explainable Artificial Intelligence",
	year="2024",
	publisher="Springer Nature Switzerland",
	address="Cham",
	pages="332--355",
	abstract="While Artificial Intelligence and Machine Learning models are becoming increasingly prevalent, it is essential to remember that they are not infallible or inherently objective. These models depend on the data they are trained on and the inherent bias of the chosen machine learning algorithm. Therefore, selecting and sampling data for training is crucial for a fair outcome of the model. A model predicting, e.g., whether an applicant should be taken further in the job application process, could create heavily biased predictions against women if the data used to train the model mostly contained information about men. The well-known concept of conditional categories used in Conformal Prediction can be utilised to address this type of bias in the data. The Conformal Prediction framework includes uncertainty quantification methods for classification and regression. To help meet the challenges of data sets with potential bias, conditional categories were incorporated into an existing explanation method called Calibrated Explanations, relying on conformal methods. This approach allows users to try out different settings while simultaneously having the possibility to study how the uncertainty in the predictions is affected on an individual level. Furthermore, this paper evaluated how the uncertainty changed when using conditional categories based on attributes containing potential bias. It showed that the uncertainty significantly increased, revealing that fairness came with a cost of increased uncertainty.",
	isbn="978-3-031-63787-2"
}
``` 

Bibtex for the multi-class paper:

```bibtex
@InProceedings{lofstrom2024ce_multiclass,
	title = 	 {Calibrated Explanations for Multi-class},
	author =       {L\"{o}fstr\"{o}m, Tuwe and L\"{o}fstr\"{o}m, Helena and Johansson, Ulf},
	booktitle = 	 {Proceedings of the Thirteenth Symposium on Conformal and Probabilistic Prediction with Applications},
	pages = 	 {175--194},
	year = 	 {2024},
	editor = 	 {Vantini, Simone and Fontana, Matteo and Solari, Aldo and Boström, Henrik and Carlsson, Lars},
	volume = 	 {230},
	series = 	 {Proceedings of Machine Learning Research},
	month = 	 {09--11 Sep},
	publisher =    {PMLR},
	pdf = 	 {https://raw.githubusercontent.com/mlresearch/v230/main/assets/lofstrom24a/lofstrom24a.pdf},
	url = 	 {https://proceedings.mlr.press/v230/lofstrom24a.html},
	abstract = 	 {Calibrated Explanations is a recently proposed feature importance explanation method providing uncertainty quantification. It utilises Venn-Abers to generate well-calibrated factual and counterfactual explanations for binary classification. In this paper, we extend the method to support multi-class classification. The paper includes an evaluation illustrating the calibration quality of the selected multi-class calibration approach, as well as a demonstration of how the explanations can help determine which explanations to trust.}
}
```

To cite this software, use the following bibtex entry:

```bibtex
@software{lofstrom2024ce_repository,
	author = 	{Löfström, Helena and Löfström, Tuwe and Johansson, Ulf and Sönströd, Cecilia and Matela, Rudy},
	license = 	{BSD-3-Clause},
	title = 	{Calibrated Explanations},
	url = 		{https://github.com/Moffran/calibrated_explanations},
	version = 	{v0.4.0},
	month = 	August,
	year = 		{2024}
}
```
