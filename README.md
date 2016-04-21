# Document Classification Reuters-21578 
Classificate documents on topics, using Reuters-21578 data.



### Requirements
Please see requirements.txt.
<br />
To install these packages, use the following command in a <a href="http://docs.python-guide.org/en/latest/dev/virtualenvs/" target="_blank"> virtualenv</a>.
```bash
$ pip install -r requirements.txt
```

### Training data
Based on Reuters-21578 files.
<br />
Available in sgm format on 
```bash
classification/data/ 
```
Trained data's topics can be found in
```bash
classification/data/all-topics-strings.lc.txt
```
To train and test, run the following 
from classification/
### Train 
```bash
$ python train_and_classify_reuters_data.py 
```
Flags
```bash
--no-stemming  # don't use stemming when transforming raw data 
# or
--no-stopwords # don't use "remove stopwords" when tranforming data 
```
Last flag, if mentioned
```
--svm         # use <a href="https://en.wikipedia.org/wiki/Support_vector_machine"> Support Vector Machine </a> classifier 
--naive-bayes # use <a href="https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Multinomial_naive_Bayes"> Naive-Bayes </a> classifier
--perceptron  # use <a href="https://en.wikipedia.org/wiki/Perceptron"> Perceptron </a>
```
