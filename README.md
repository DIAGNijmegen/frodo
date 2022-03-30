Newer releases are continued under github.com/xarion/tf_frodo
## FRODO: Free rejection of out-of-distribution samples in medical imaging
# Install
`pip install tf_frodo`
# Use
```
import model
from tf_frodo import FRODO

model_with_frodo = FRODO(model).fit(x_validation) 

results = model_with_frodo.predict(x_test)

assert results["model_outputs"] == model(x_test)
results["FRODO"] # Rejection scores for each sample in x_test
```
