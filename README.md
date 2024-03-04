# Robust Optimization for Adversarial Deep Learning

This repository contains the code of to the master thesis 'Robust Optimization for Adversarial Deep Learning' (https://doi.org/...) by Lars Steffen März from October 19, 2023 at TU Darmstadt published via TUprints under the

*LICENSE:*
CC BY 4.0 International - Creative Commons, Attribution, cf. https://creativecommons.org/licenses/by/4.0

*COPYRIGHT:*
The contents of this repository are partially based on https://github.com/giughi/An-Empirical-Study-of-DFO-Algorithms-for-Targeted-Black-Box-Attacks-in-DNNs.git and https://github.com/giughi/A-Model-Based-Derivative-Free-Approach-to-Black-Box-Adversarial-Examples-BOBYQA.git. All the code, stats and visuals in the directories `MATLAB` and `Python/Danskin` are written and created by Lars Steffen März.

*ABSTRACT:*
Recent results demonstrated that images can be adversarially perturbed to a visually indistinguishable extent in order to misguide classifiers with high standard accuracy into making confident misclassifications. Adversarial examples may even be targeted to a class the attacker chooses and transfer between different DNNs in a black-box setting, meaning that perturbations computed on one DNN are likely to confuse other DNNs. This poses a concrete and acute security risk in digital domains like content moderation, but also in physical contexts like facial recognition and autonomous driving where adversarial samples proved to survive printing and re-capturing. The phenomenon was first discovered in 2014 by Szegedy et al. and has been subject of hundreds of papers ever since, both from an attacker's and a defender's point of view. There seems to be no apparent end to an arms race of frequently published attacks and defenses as no universal, provable and practical prevention method has been developed yet.

In this work, we show that verifying ReLU-based DNNs against adversarial examples is NP-hard. Furthermore, we model the adversarial training problem as a distributionally robust optimization problem to provide a formal framework for two of the most promising defenses so far: Randomized FGSM-based adversarial training and randomized smoothing. Additionally, we propose two step size schemes for multi-step adversarial attacks that yield unprecedented low true-label-confidences. To make p-norm bounded attacks more comparable for different values of p, we define two norm rescaling functions before validating them on ImageNet. Moreover, we give an explanation as to why first-order adversarial training is successful from an empirical data augmentation perspective despite lacking the mathematical guarantees from Danskin's Theorem by analyzing cosine similarities of model parameter gradients on ImageNet. Finally, we give an update on the performance results from Giughi et al. of BOBYQA black-box attacks on CIFAR-10 by exposing instances of the two aforementioned state-of-the-art defenses to it.
