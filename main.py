from sources.preprocessing.preprocessing import tweet_tokenizer
from sources.features.deepmoji_feature.deepmoji_vector import read_test, deepmoji_vector
from sources.code.ei_reg.random_forest.random_forest_deepmoji import predict
import subprocess
p = subprocess.Popen(['ls', '-a'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
out, err = p.communicate()
print(out)

# predict()
# read_test()
# deepmoji_vector()