# Citing calibrated-explanations

If you use `calibrated-explanations` for a scientific publication, you are kindly requested to cite one of the following papers:
## Published papers
- [Löfström, H](https://github.com/Moffran). (2023). [Trustworthy explanations: Improved decision support through well-calibrated uncertainty quantification](https://www.diva-portal.org/smash/record.jsf?pid=diva2%3A1810440&dswid=6197) (Doctoral dissertation, Jönköping University, Jönköping International Business School).
- [Löfström, H](https://github.com/Moffran)., [Löfström, T](https://github.com/tuvelofstrom)., Johansson, U., and Sönströd, C. (2024). [Calibrated Explanations: with Uncertainty Information and Counterfactuals](https://doi.org/10.1016/j.eswa.2024.123154). Expert Systems with Applications, 1-27. https://doi.org/10.1016/j.eswa.2024.123154
- [Löfström, T](https://github.com/tuvelofstrom)., [Löfström, H](https://github.com/Moffran)., Johansson, U., Sönströd, C., and [Matela, R](https://github.com/rudymatela). (2025). [Calibrated Explanations for Regression](https://doi.org/10.1007/s10994-024-06642-8). Machine Learning 114, 100. https://doi.org/10.1007/s10994-024-06642-8
- [Löfström, H](https://github.com/Moffran)., [Löfström, T](https://github.com/tuvelofstrom). (2024). [Conditional Calibrated Explanations: Finding a Path Between Bias and Uncertainty](https://doi.org/10.1007/978-3-031-63787-2_17). In: Longo, L., Lapuschkin, S., Seifert, C. (eds) Explainable Artificial Intelligence. xAI 2024. Communications in Computer and Information Science, vol 2153. Springer, Cham. https://doi.org/10.1007/978-3-031-63787-2_17
- [Löfström, T](https://github.com/tuvelofstrom)., [Löfström, H](https://github.com/Moffran)., Johansson, U. (2024). [Calibrated Explanations for Multi-class](https://raw.githubusercontent.com/mlresearch/v230/main/assets/lofstrom24a/lofstrom24a.pdf). <i>Proceedings of the Thirteenth Workshop on Conformal and Probabilistic Prediction and Applications</i>, in <i>Proceedings of Machine Learning Research</i>, PMLR 230:175-194. [Presentation](https://copa-conference.com/presentations/Lofstrom.pdf)

The paper that originated the idea of `calibrated-explanations` is:

- [Löfström, H.](https://github.com/Moffran), [Löfström, T.](https://github.com/tuvelofstrom), Johansson, U., & Sönströd, C. (2023). [Investigating the impact of calibration on the quality of explanations](https://link.springer.com/article/10.1007/s10472-023-09837-2). Annals of Mathematics and Artificial Intelligence, 1-18. [Code and results](https://github.com/tuvelofstrom/calibrating-explanations).

## Preprints: 
- [Löfström, H](https://github.com/Moffran)., [Löfström, T](https://github.com/tuvelofstrom)., and [Hallberg Szabadvary, J](https://github.com/egonmedhatten). (2024). [Ensured: Explanations for Decreasing the Epistemic Uncertainty in Predictions](https://arxiv.org/abs/2410.05479). arXiv preprint arXiv:2410.05479. 
- [Löfström, T](https://github.com/tuvelofstrom)., [Rabia Yapicioglu, F](https://github.com/rabia174)., Stramiglio A., [Löfström, H](https://github.com/Moffran)., and Vitali F. (2024). [Fast Calibrated Explanations: Efficient and Uncertainty-Aware Explanations for Machine Learning Models](https://arxiv.org/abs/2410.21129). arXiv preprint arXiv:2410.21129. 

# Bibtex Entries
## Published papers

Bibtex for Helena Löfström's dissertation:

```bibtex
@phdthesis{lofstrom2023dissertation,
	author = 		{L{\"o}fstr{\"o}m, Helena},
	institution = 	{Jönköping University, JIBS, Informatics},
	pages = 		{72},
	publisher = 	{Jönköping University},
	school = 		{Jönköping International Business School, JIBS, Informatics},
	title = 		{Trustworthy explanations : Improved decision support through well-calibrated uncertainty quantification},
	series = 		{JIBS Dissertation Series},
	ISSN = 			{1403-0470},
	number = 		{159},
	keywords = 		{Explainable Artificial Intelligence, Interpretable Machine Learning, Decision Support Systems, Uncertainty Estimation, Explanation Methods},
	abstract = 		{The use of Artificial Intelligence (AI) has transformed fields like disease diagnosis and defence. Utilising sophisticated Machine Learning (ML) models, AI predicts future events based on historical data, introducing complexity that challenges understanding and decision-making. Previous research emphasizes users’ difficulty discerning when to trust predictions due to model complexity, underscoring addressing model complexity and providing transparent explanations as pivotal for facilitating high-quality decisions. Many ML models offer probability estimates for predictions, commonly used in methods providing explanations to guide users on prediction confidence. However, these probabilities often do not accurately reflect the actual distribution in the data, leading to potential user misinterpretation of prediction trustworthiness. Additionally, most explanation methods fail to convey whether the model’s probability is linked to any uncertainty, further diminishing the reliability of the explanations. Evaluating the quality of explanations for decision support is challenging, and although highlighted as essential in research, there are no benchmark criteria for comparative evaluations. This thesis introduces an innovative explanation method that generates reliable explanations, incorporating uncertainty information supporting users in determining when to trust the model’s predictions. The thesis also outlines strategies for evaluating explanation quality and facilitating comparative evaluations. Through empirical evaluations and user studies, the thesis provides practical insights to support decision-making utilising complex ML models. },
	ISBN = 			{978-91-7914-031-1},
	ISBN = 			{978-91-7914-032-8},
	year = 			{2023}
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
	doi = 		{10.1016/j.eswa.2024.123154},
	url = 		{https://www.sciencedirect.com/science/article/pii/S0957417424000198},
	author = 	{Helena Löfström and Tuwe Löfström and Ulf Johansson and Cecilia Sönströd},
	keywords = 	{Explainable AI, Feature importance, Calibrated explanations, Venn-Abers, Uncertainty quantification, Counterfactual explanations},
	abstract = 	{While local explanations for AI models can offer insights into individual predictions, such as feature importance, they are plagued by issues like instability. The unreliability of feature weights, often skewed due to poorly calibrated ML models, deepens these challenges. Moreover, the critical aspect of feature importance uncertainty remains mostly unaddressed in Explainable AI (XAI). The novel feature importance explanation method presented in this paper, called Calibrated Explanations (CE), is designed to tackle these issues head-on. Built on the foundation of Venn-Abers, CE not only calibrates the underlying model but also delivers reliable feature importance explanations with an exact definition of the feature weights. CE goes beyond conventional solutions by addressing output uncertainty. It accomplishes this by providing uncertainty quantification for both feature weights and the model’s probability estimates. Additionally, CE is model-agnostic, featuring easily comprehensible conditional rules and the ability to generate counterfactual explanations with embedded uncertainty quantification. Results from an evaluation with 25 benchmark datasets underscore the efficacy of CE, making it stand as a fast, reliable, stable, and robust solution.}
}
```

Bibtex entry for the regression paper:

```bibtex
@article{lofstrom2025ce_regression,
	title =		{Calibrated explanations for regression},
	author =	{L{\"o}fstr{\"o}m, Tuwe and L{\"o}fstr{\"o}m, Helena and Johansson, Ulf and S{\"o}nstr{\"o}d, Cecilia and Matela, Rudy},
	journal =	{Machine Learning},
	volume =	{114},
	number =	{100},
	year =		{2025},
	publisher =	{Springer Nature},
	doi = 		{10.1007/s10994-024-06642-8},
	url = 		{https://link.springer.com/article/10.1007/s10994-024-06642-8},
	abstract =  {Artificial Intelligence (AI) methods are an integral part of modern decision support systems. The best-performing predictive models used in AI-based decision support systems lack transparency. Explainable Artificial Intelligence (XAI) aims to create AI systems that can explain their rationale to human users. Local explanations in XAI can provide information about the causes of individual predictions in terms of feature importance. However, a critical drawback of existing local explanation methods is their inability to quantify the uncertainty associated with a feature’s importance. This paper introduces an extension of a feature importance explanation method, Calibrated Explanations, previously only supporting classification, with support for standard regression and probabilistic regression, i.e., the probability that the target is below an arbitrary threshold. The extension for regression keeps all the benefits of Calibrated Explanations, such as calibration of the prediction from the underlying model with confidence intervals, uncertainty quantification of feature importance, and allows both factual and counterfactual explanations. Calibrated Explanations for regression provides fast, reliable, stable, and robust explanations. Calibrated Explanations for probabilistic regression provides an entirely new way of creating probabilistic explanations from any ordinary regression model, allowing dynamic selection of thresholds. The method is model agnostic with easily understood conditional rules. An implementation in Python is freely available on GitHub and for installation using both pip and conda, making the results in this paper easily replicable.}
}
```

Bibtex for the conditional paper:

```bibtex
@InProceedings{lofstrom2024ce_conditional,
	author =	"L{\"o}fstr{\"o}m, Helena and L{\"o}fstr{\"o}m, Tuwe",
	editor =	"Longo, Luca and Lapuschkin, Sebastian and Seifert, Christin",
	title =		"Conditional Calibrated Explanations: Finding a Path Between Bias and Uncertainty",
	booktitle =	"Explainable Artificial Intelligence",
	year =		"2024",
	publisher =	"Springer Nature Switzerland",
	address =	"Cham",
	pages =		"332--355",
	abstract =	"While Artificial Intelligence and Machine Learning models are becoming increasingly prevalent, it is essential to remember that they are not infallible or inherently objective. These models depend on the data they are trained on and the inherent bias of the chosen machine learning algorithm. Therefore, selecting and sampling data for training is crucial for a fair outcome of the model. A model predicting, e.g., whether an applicant should be taken further in the job application process, could create heavily biased predictions against women if the data used to train the model mostly contained information about men. The well-known concept of conditional categories used in Conformal Prediction can be utilised to address this type of bias in the data. The Conformal Prediction framework includes uncertainty quantification methods for classification and regression. To help meet the challenges of data sets with potential bias, conditional categories were incorporated into an existing explanation method called Calibrated Explanations, relying on conformal methods. This approach allows users to try out different settings while simultaneously having the possibility to study how the uncertainty in the predictions is affected on an individual level. Furthermore, this paper evaluated how the uncertainty changed when using conditional categories based on attributes containing potential bias. It showed that the uncertainty significantly increased, revealing that fairness came with a cost of increased uncertainty.",
	isbn =		"978-3-031-63787-2"
}
``` 

Bibtex for the multi-class paper:

```bibtex
@InProceedings{lofstrom2024ce_multiclass,
	title = 	{Calibrated Explanations for Multi-class},
	author =    {L\"{o}fstr\"{o}m, Tuwe and L\"{o}fstr\"{o}m, Helena and Johansson, Ulf},
	booktitle = {Proceedings of the Thirteenth Symposium on Conformal and Probabilistic Prediction with Applications},
	pages = 	{175--194},
	year = 	 	{2024},
	editor = 	{Vantini, Simone and Fontana, Matteo and Solari, Aldo and Boström, Henrik and Carlsson, Lars},
	volume = 	{230},
	series = 	{Proceedings of Machine Learning Research},
	month = 	{09--11 Sep},
	publisher = {PMLR},
	pdf = 	 	{https://raw.githubusercontent.com/mlresearch/v230/main/assets/lofstrom24a/lofstrom24a.pdf},
	url = 	 	{https://proceedings.mlr.press/v230/lofstrom24a.html},
	abstract = 	{Calibrated Explanations is a recently proposed feature importance explanation method providing uncertainty quantification. It utilises Venn-Abers to generate well-calibrated factual and counterfactual explanations for binary classification. In this paper, we extend the method to support multi-class classification. The paper includes an evaluation illustrating the calibration quality of the selected multi-class calibration approach, as well as a demonstration of how the explanations can help determine which explanations to trust.}
}
```

## Preprints:

Bibtex entry for the ensured paper:

```bibtex
@misc{lofstrom2024ce_ensured,
	title = 	      {Ensured: Explanations for Decreasing the Epistemic Uncertainty in Predictions},
	author =          {L\"ofstr\"om, Helena and L\"ofstr\"om, Tuwe and Hallberg Szabadvary, Johan},
	year =            {2024},
	eprint =          {2410.05479},
	archivePrefix =   {arXiv},
	primaryClass =    {cs.LG}
}
```

Bibtex entry for the fast paper:

```bibtex
@misc{lofstrom2024ce_fast,
	title=			{Fast Calibrated Explanations: Efficient and Uncertainty-Aware Explanations for Machine Learning Models}, 
	author=			{Tuwe Löfström and Fatima Rabia Yapicioglu and Alessandra Stramiglio and Helena Löfström and Fabio Vitali},
	year=			{2024},
	eprint=			{2410.21129},
	archivePrefix=	{arXiv},
	primaryClass=	{cs.LG},
	url=			{https://arxiv.org/abs/2410.21129}, 
}
```

## Software
To cite this software, use the following bibtex entry:

```bibtex
@software{lofstrom2024ce_repository,
	author = 	{Löfström, Helena and Löfström, Tuwe and Johansson, Ulf and Sönströd, Cecilia and Matela, Rudy},
	license = 	{BSD-3-Clause},
	title = 	{Calibrated Explanations},
	url = 		{https://github.com/Moffran/calibrated_explanations},
	version = 	{v0.5.1},
	month = 	{November},
	year = 		{2024}
}
```
