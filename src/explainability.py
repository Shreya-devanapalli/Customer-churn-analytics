import shap
import matplotlib.pyplot as plt
import pandas as pd

def shap_analysis(model, X_test):

    explainer = shap.Explainer(model)

    shap_values = explainer(X_test)

    shap.summary_plot(
        shap_values,
        X_test,
        show=False
    )

    plt.savefig("outputs/shap_summary.png")

    plt.close()


def feature_importance(model, X_test):

    importances = pd.Series(
        model.feature_importances_,
        index=X_test.columns
    )

    importances.sort_values().tail(10).plot(
        kind="barh"
    )

    plt.savefig("outputs/feature_importance.png")

    plt.close()