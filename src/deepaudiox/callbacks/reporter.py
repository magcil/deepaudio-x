import numpy as np

from deepaudiox.callbacks.base_callback import BaseCallback
from deepaudiox.utils.reporting_utils import get_avg_posteriors, get_classification_report, get_confusion_matrix
from deepaudiox.utils.training_utils import get_logger


class Reporter(BaseCallback):
    """Training callback for producing training reports.
    
    Attributes:
        logger: A module for logging messages.

    """
    def __init__(self, logger: logging.Logger=None):
        """Initialize the callback.

        Args:
            logger: A module for logging messages. Defaults to None.

        """
        self.logger = logger or get_logger()

    def on_testing_end(self, evaluator):
        """When testing ends, produce report.
        
        Args:
            evaluator (evaluator.Evaluator): The evaluation module of the SDK.

        """
        # Produce classification report
        report = get_classification_report(
            y_true = evaluator.state.y_true,
            y_pred = evaluator.state.y_pred,
            class_mapping = evaluator.class_mapping
        )

        # Produce confusion matrix
        matrix = get_confusion_matrix(
            y_true = evaluator.state.y_true,
            y_pred = evaluator.state.y_pred,
            class_mapping = evaluator.class_mapping
        )

        # Calculate average posterior probability per class
        avg_posteriors = get_avg_posteriors(
            y_true = evaluator.state.y_true,
            y_pred = evaluator.state.y_pred,
            posteriors = evaluator.state.posteriors,
            class_mapping = evaluator.class_mapping
        )
        formatted_posteriors = "\n".join(
            f"{label:<20}: {val:.3f}" if not np.isnan(val) else f"{label:<20}: NaN"
            for label, val in avg_posteriors.items()
        )
        
        # Log results
        self.logger.info(f"[REPORTER] Class mapping: {evaluator.class_mapping} \n")
        self.logger.info("[REPORTER] Classification Report: \n")
        self.logger.info(report)
        self.logger.info("[REPORTER] Confusion Matrix: \n")
        self.logger.info(matrix)
        self.logger.info("[REPORTER] Average Posteriors: \n")
        self.logger.info(formatted_posteriors)

        return

    