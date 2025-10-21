import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd

# ----------------------------------------------------------------------------------------------------------------------
# Accuracies on accepted samples (i.e., Precision) at different uncertainty thresholds
# T: correct sample; TP: accepted correct sample; FP: accepted incorrect sample; Precision=TP/(TP+FP)
# employ each uncertainty score in uncertainty_list as the threshold
def calculate_auarc(correctness_list, uncertainty_list):
    uncertainty_list = np.array(uncertainty_list)
    correctness_list = np.array(correctness_list)
    # base accuracy
    base_accuracy = np.mean(correctness_list)
    # Get the non-repeated uncertainty scores and sort them in ascending order
    non_repeated_uncertainty_list = sorted(np.unique(uncertainty_list))
    # For storing the accuracies at different uncertainty thresholds
    accuracy_list = []
    # Loop through each threshold and calculate the accuracy for accepted samples
    for threshold in non_repeated_uncertainty_list:
        # Accept samples with uncertainty less than the threshold
        accepted_indices = np.where(uncertainty_list <= threshold)[0]
        if len(accepted_indices) > 0:
            # Calculate the accuracy for the accepted samples
            accuracy = np.mean(correctness_list[accepted_indices])
        else:
            # If no samples are accepted, accuracy is undefined, so we assume 0
            accuracy = 0
        accuracy_list.append(accuracy)

    # Return a DataFrame for better presentation
    # df_output = pd.DataFrame({
    #     "Threshold": non_repeated_uncertainty_list,
    #     "Accuracy": accuracy_list
    # })
    #
    # print(df_output)

    # Compute the AUARC as the area under the Accuracy-Rejection Curve
    auarc_value = metrics.auc(non_repeated_uncertainty_list, accuracy_list)

    # Plot the Accuracy-Rejection Curve
    plt.plot(non_repeated_uncertainty_list, accuracy_list, label='Method')
    # Plot the base accuracy as a dashed line (compare the base accuracy with the calibrated accuracy)
    plt.axhline(y=base_accuracy, color='r', linestyle='--', label=f'Base Accuracy = {base_accuracy:.2f}')
    plt.xlabel('Uncertainty Thresholds')
    plt.ylabel('Accuracy on Accepted Samples')
    plt.title('Accuracy-Rejection Curve (ARC)')
    # plt.grid(True)
    plt.legend()
    plt.show()

    return auarc_value
# ----------------------------------------------------------------------------------------------------------------------
# accuracy on remaining (1 - rejection_rate) reliable samples after rejecting (rejection_rate) high-uncertainty samples
# sort uncertainty scores in ascending order and employ the (1 - rejection_rate) of them as the uncertainty threshold
def accuracy_at_rejection_rate(correctness_list, uncertainty_list, rejection_rate):
    # quantile or rejection rate
    correctness_list = np.array(correctness_list)
    uncertainty_list = np.array(uncertainty_list)
    # uncertainty threshold
    cutoff = np.quantile(uncertainty_list, (1 - rejection_rate))
    # accept pre-(1 - rejection_rate) samples
    select = uncertainty_list <= cutoff
    return np.mean(correctness_list[select])
# ----------------------------------------------------------------------------------------------------------------------
# Following Figure 3 in "GENERATING WITH CONFIDENCE: UNCERTAINTY QUANTIFICATION FOR BLACK-BOX LARGE LANGUAGE MODELS"
# accuracy on remaining (1 - rejection_rate) reliable samples after rejecting (rejection_rate) high-uncertainty samples
# set different rejection_rate and calculate the area under the accuracy-rejection curve
# (accuracy on accepted samples/Precision)
def area_under_accuracies_at_various_rejection_rates(correctness_list, uncertainty_list, num_rejection_rates):
    correctness_list = np.array(correctness_list)
    uncertainty_list = np.array(uncertainty_list)
    # base accuracy
    base_accuracy = np.mean(correctness_list)
    # quantiles
    rejection_rates = np.linspace(0, 1, num_rejection_rates)
    # accuracies at different quantiles (accepted samples)
    accuracies = np.array([accuracy_at_rejection_rate(correctness_list, uncertainty_list, r) for r in rejection_rates])

    # Plot the Accuracy-Rejection Curve
    plt.plot(rejection_rates, accuracies, label='Method')
    # Plot the base accuracy as a dashed line
    plt.axhline(y=base_accuracy, color='r', linestyle='--', label=f'Base Accuracy = {base_accuracy:.2f}')
    plt.xlabel('Rejection Rate')
    plt.ylabel('Accuracy on Accepted Samples')
    plt.title('Accuracy-Rejection Curve (ARC)')
    # plt.grid(True)
    plt.legend()
    plt.show()

    auarc_value = metrics.auc(rejection_rates, accuracies)
    return auarc_value


correctness = [1, 1, 0, 1, 1, 0, 1, 0, 0, 0]
uncertainty_scores_1 = [0.1, 0.26, 0.75, 0.99, 0.52, 0.33, 0.46, 0.52, 0.35, 0.66]
uncertainty_scores_2 = [0.5, 0.93, 0.75, 0.45, 0.52, 0.33, 0.95, 0.79, 0.15, 0.55]

# print(calculate_auarc(correctness, uncertainty_scores_1))
# print(calculate_auarc(correctness, uncertainty_scores_2))

print(area_under_accuracies_at_various_rejection_rates(correctness, uncertainty_scores_1, 11))
print(area_under_accuracies_at_various_rejection_rates(correctness, uncertainty_scores_2, 11))

# print(accuracy_at_quantile(correctness, uncertainty_scores_1, 0.4))
# print(accuracy_at_quantile(correctness, uncertainty_scores_2, 0.4))