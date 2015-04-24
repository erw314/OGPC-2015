import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier

# Convert train.csv to numpy array
training_features = []
training_labels = []
with open('train.csv') as f:
    reader = csv.reader(f)
    reader.next()
    for n in reader:
        training_features.append(n[1:-1])
        training_labels.append(n[-1])
training_features = np.array([[int(n) for n in m] for m in training_features])


# Convert test.csv to numpy array
test_features = []
with open('test.csv') as f:
    reader = csv.reader(f)
    reader.next()
    for n in reader:
        test_features.append(n[1:])
test_features = np.array([[int(n) for n in m] for m in test_features])

# Run random forest classifier
rfc = RandomForestClassifier(n_estimators=30)
rfc.fit(training_features, training_labels)
rfc_probs = rfc.predict_proba(test_features)


with open('FIRST.csv', 'wb') as fp:
    a = csv.writer(fp, delimiter=',')
    first_row = ['id','Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9']
    a.writerow(first_row)
    rfc_probs_with_id = []
    for i in range(len(rfc_probs)):
    	prob_numpy_array = rfc_probs[i]
    	probs = [prob_numpy_array[j] for j in range(len(prob_numpy_array))]
    	probs.insert(0, i+1)
    	rfc_probs_with_id.append(probs)
    a.writerows(rfc_probs_with_id)
