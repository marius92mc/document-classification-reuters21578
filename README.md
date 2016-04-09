# Document Classification Reuters-21578 
Classificate documents on topics, using Reuters-21578 data.


### Requirements
Please see requirements.txt.
<br />
To install these packages, use
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
To train/test, run the following scripts 
from classification/
### Train 
```bash
$ python train_and_classify_reuters_data.py 
```

