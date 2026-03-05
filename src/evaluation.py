from sklearn.metrics import classification_report, roc_auc_score

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    print("Classification Report:\n")

    print(classification_report(y_test, predictions))

    roc = roc_auc_score(y_test, probabilities)

    print("ROC AUC Score:", roc)