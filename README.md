# LionForests-Bot
LionForests Bot: Interactive dialogue between user-machine learning model via explanations

Moving into the future, we can assume that systems or models that utilize Machine Learning technologies, increasingly sophisticated and complex, will be widely used to draw useful conclusions, after consuming and processing large volumes of data. Many of these models, although showing extremely high performance, are characterized as black boxes, as they do not provide information about the rationale behind the decisions they make, thus creating a climate of distrust and avoidance. To deal with this problem, it is necessary to synthesize techniques for explaining the aforementioned models, like Random Forest. In this work, which is based on ''LionForests'', a package used for local interpretation of Random Forests, we propose our own solution to the problem by expanding the current package and presenting the ''LionForest Bot'', a chatbot that performs interactive user-Random Forest model dialogue through explanations, towards more usable and understandable explanations for the users.

## LionForests
Local Interpretation Of raNdom FORESTS. Building interpretable random forests!

Towards a future where ML systems will integrate into every aspect of people’s lives, researching methods to interpret such systems is necessary, instead of focusing exclusively on enhancing their performance. Enriching the trust between these systems and people will accelerate this integration process. Many medical and retail banking/finance applications use state-of-the-art ML techniques to predict certain aspects of new instances. Thus, explainability is a key requirement for human-centred AI approaches. Tree ensembles, like random forests, are widely acceptable solutions on these tasks, while at the same time they are avoided due to their black-box uninterpretable nature, creating an unreasonable paradox. In this paper, we provide a methodology for shedding light on the predictions of the misjudged family of tree ensemble algorithms. Using classic unsupervised learning techniques and an enhanced similarity metric, to wander among transparent trees inside a forest following breadcrumbs, the interpretable essence of tree ensembles arises. An interpretation provided by these systems using our approach, which we call “LionForests”, can be a simple, comprehensive rule.

## Instructions
Please ensure you have Flask installed. Then:
```bash
python3 app.py
```
After successfully running LFBot, please go to your browser to the proposed url (probably will be the following):
```url
http://127.0.0.1:5000/
```
Then, enjoy LF-Bot.

## Contributors on Altruist
Name | Email | Contribution
--- | --- | ---
Ioannis Chatziarapis | ichatzik@csd.auth.gr | Main 
[Ioannis Mollas](https://intelligence.csd.auth.gr/people/ioannis-mollas/) | iamollas@csd.auth.gr | Supervision
[Nick Bassiliades](https://intelligence.csd.auth.gr/people/bassiliades/) | nbassili@csd.auth.gr | Supervision

## See our Lab's work on interpretability
- [LionLearn Interpretability Library](https://github.com/intelligence-csd-auth-gr/LionLearn) containing: 
1. [LioNets](https://github.com/iamollas/LionLearn/tree/master/LioNets): Local Interpretation Of Neural nETworkS through penultimate layer decoding
2. [LionForests](https://github.com/iamollas/LionLearn/tree/master/LionForests): Local Interpretation Of raNdom FORESts through paTh Selection
- [Altruist](https://github.com/iamollas/Altruist)

## License
[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
