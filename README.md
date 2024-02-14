# Land-usage-classification-with-Python
This project involved the creation of masks for determining land usage with Python. Images and their masks were downloaded and broken down into structured data. Then using the scikit-learn library's LogisticRegression module, a classification model was created. This model was used to create masks for various satellite images.


Results:
With small training and testing sizes (4-6 images) ==> Accuracy train DataSet:  82.5% Accuracy test DataSet:  70.3%
With Larger sets, there seems to be an issue of overfitting which leads to significant reduction in test dataset accuracy ( reduced to 40%)

Actual Image
<img width="920" alt="image" src="https://github.com/Amloner/Land-usage-classification-with-Python/assets/124287518/0ac0832f-0379-45f6-9c2a-87bb87726e9f">
<img width="920" alt="image" src="https://github.com/Amloner/Land-usage-classification-with-Python/assets/124287518/e928364c-dc23-4528-ba8e-a0357f1cb13d">


