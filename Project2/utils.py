import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.model_selection import cross_val_score

class Utils:
    """
    Goes through predicted labels comparing to the actual labels
    too find which clusters contain missclassified points

    takes the number of clusters, the actual labels and the predicted labels as parameters
    """
    def find_missclassified_clusters(clusters, X, pred):
      indices = []
      missclass_clusters = []
      #Go through all clusters
      for i in range(clusters):
        missclassified = 0
        cluster_data = []
        iteration = 0
        #Loop through all clustered data and actual labels
        for actual, predicted in zip(X, pred):
          #Only look at points from certain cluster
          if predicted == i:
            cluster_data.append(iteration)
            #see if point in cluster have been missclassified
            if actual != predicted:
              missclassified = 1
          iteration +=1
        #Take note of which clusters have missclassified points
        if missclassified == 1:
          missclass_clusters.append(i)
        indices.append(cluster_data)
      return (indices, missclass_clusters)

    """
    Predicts the amount of clusters for a dataset using the silhouette method and the elbow method.
    Takes data and corresponding labels as parameters
    """
    def cluster_and_model(X, y):
        print("finding clusters")
        sse_list = []
        max_silhouette = -1
        for k in range(2, 15):
            mmodel = KMeans(n_clusters=k).fit(X)
            sse_list.append(mmodel.inertia_)
            s_score = silhouette_score(X, mmodel.labels_, metric='euclidean')
            if s_score > max_silhouette:
                max_silhouette = s_score
                max_silhouette_index = k
        plt.plot(list(range(2, 15)), sse_list, '.-')
        plt.show()
        input_cluster = int(input("Silhouette predicted %d clusters. Please enter elbow results:\n" %max_silhouette_index))
        #Find out what value is most accurate if user enters another value for elbow method than silhouete generated
        models = []
        scores = []
        if input_cluster <= max_silhouette_index:
            for ac in range(input_cluster, max_silhouette_index+1):
                temp = KMeans(n_clusters=ac).fit(X)
                models.append((temp, ac))
                scores.append(
                    accuracy_score(temp.labels_, y) + f1_score(temp.labels_, y, average='weighted') + precision_score(
                        temp.labels_,
                        y, average='weighted') + recall_score(
                        temp.labels_, y, average='weighted'))
        else:
            for ac in range(max_silhouette_index, input_cluster+1):
                temp = KMeans(n_clusters=ac).fit(X)
                models.append((temp, ac))
                scores.append(
                    accuracy_score(temp.labels_, y) + f1_score(temp.labels_, y, average='weighted') + precision_score(
                        temp.labels_,
                        y, average='weighted') + recall_score(
                        temp.labels_, y, average='weighted'))
        index_of_best = np.argmax(scores)
        final_model, final_k = models[index_of_best]
        print('K value with best performance:', final_k)
        return (final_k, final_model)

    """
    Retrieve a title and four arrays of performance scores to be plotted
    """
    def boxplots(title, base, oversampled, undersampled, proposed, smote):
        data = [base, oversampled, undersampled, proposed, smote]
        fig, ax = plt.subplots()
        ax.set_title(title)
        labs = ["", "Base", "Oversampled", "Downsampled", "Proposed Method", "SMOTE"]
        ax.set_xticks(np.arange(len(labs)))
        ax.set_xticklabels(labs, rotation=45)
        ax.boxplot(data)
        plt.show()
        
    """
    Retrieves a classifier, features and labels to perform cross-validation and display results
    """
    def evaluatePerformance(clf, features, labels):
        a = cross_val_score(clf, features, labels, cv=10, scoring='accuracy')
        p = cross_val_score(clf, features, labels, cv=10, scoring='precision')
        r = cross_val_score(clf, features, labels, cv=10, scoring='recall')
        f = cross_val_score(clf, features, labels, cv=10, scoring='f1')

        print("Accuracy:", a.mean(), "+/-", a.std())
        print("Precision:", p.mean(), "+/-", p.std())
        print("Recall:", r.mean(), "+/-", r.std())
        print("F1-Score:", f.mean(), "+/-", f.std())
        return f
