# mlflow_client.py
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = "http://localhost:5001"
MLFLOW_EXPERIMENT_NAME = "Amazon_Sentiment_Retraining"


class MLflowDashboardClient:
    def __init__(self):
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            self.client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
            self.experiment = self.client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
            logger.info(f"✅ Connected to MLflow at {MLFLOW_TRACKING_URI}")
        except Exception as e:
            logger.error(f"❌ MLflow connection failed: {e}")
            self.experiment = None

    def get_current_model_stats(self):
        """Get the latest promoted model stats for the cards"""
        if not self.experiment:
            return None

        try:
            runs = self.client.search_runs(
                experiment_ids=[self.experiment.experiment_id],
                filter_string="tags.promoted = 'True'",
                max_results=1,
                order_by=["start_time DESC"]
            )

            if not runs:
                # Fallback to most recent run
                runs = self.client.search_runs(
                    experiment_ids=[self.experiment.experiment_id],
                    max_results=1,
                    order_by=["start_time DESC"]
                )

            if runs:
                run = runs[0]
                decision = run.data.tags.get("decision", "INITIAL")
                run_name = run.data.tags.get(
                    "run_name",
                    run.data.tags.get("mlflow.runName", f"Run {run.info.run_id[:8]}")
                )

                current_f1  = run.data.metrics.get("f1_macro", 0)
                previous_f1 = run.data.metrics.get("previous_f1", 0)
                if previous_f1 > 0:
                    improvement = round((current_f1 - previous_f1) / previous_f1 * 100, 2)
                else:
                    improvement = None  # first run

                return {
                    "run_name":           run_name,
                    "run_id":             run.info.run_id,
                    "f1_score":           round(current_f1 * 100, 1),
                    "accuracy":           round(run.data.metrics.get("accuracy", 0) * 100, 1),
                    "recall_negative":    round(run.data.metrics.get("recall_negative", 0) * 100, 1),
                    "improvement":        improvement,
                    "n_training_samples": int(run.data.params.get("n_training", 0)),
                    "retrained_at":       datetime.fromtimestamp(run.info.start_time / 1000),
                    "status":             decision,
                    "promoted":           decision == "PROMOTED",
                    "mlflow_url":         f"{MLFLOW_TRACKING_URI}/#/experiments/{self.experiment.experiment_id}/runs/{run.info.run_id}"
                }

        except Exception as e:
            logger.error(f"Error getting current model stats: {e}")

        return None

    def get_performance_history(self, limit=20):
        """Get historical performance for charts"""
        if not self.experiment:
            return []

        try:
            runs = self.client.search_runs(
                experiment_ids=[self.experiment.experiment_id],
                max_results=limit,
                order_by=["start_time DESC"]
            )

            history = []
            for run in runs:
                decision = run.data.tags.get("decision", "PENDING")
                promoted = run.data.tags.get("promoted", "False") == "True"
                run_time = datetime.fromtimestamp(run.info.start_time / 1000)

                current_f1  = run.data.metrics.get("f1_macro", 0)
                previous_f1 = run.data.metrics.get("previous_f1", 0)
                if previous_f1 > 0:
                    improvement = round((current_f1 - previous_f1) / previous_f1 * 100, 2)
                else:
                    improvement = None  # first run — display as —

                history.append({
                    "run_id":          run.info.run_id,
                    "run_name":        run.data.tags.get(
                                           "run_name",
                                           run.data.tags.get("mlflow.runName", run_time.strftime("%Y-%m-%d %H:%M"))
                                       ),
                    "time":            run_time.isoformat(),
                    "timestamp":       run.info.start_time,
                    "f1_score":        round(current_f1 * 100, 2),
                    "accuracy":        round(run.data.metrics.get("accuracy", 0) * 100, 2),
                    "recall_negative": round(run.data.metrics.get("recall_negative", 0) * 100, 2),
                    "f1_negative":     round(run.data.metrics.get("f1_negative", 0) * 100, 2),
                    "train_time":      run.data.metrics.get("train_time_seconds", 0),
                    "n_samples":       int(run.data.params.get("n_training", 0)),
                    "status":          decision,
                    "promoted":        promoted,
                    "improvement":     improvement,
                    "mlflow_url":      f"{MLFLOW_TRACKING_URI}/#/experiments/{self.experiment.experiment_id}/runs/{run.info.run_id}"
                })

            return history

        except Exception as e:
            logger.error(f"Error getting performance history: {e}")
            return []

    def get_latest_retraining(self):
        history = self.get_performance_history(limit=1)
        return history[0] if history else None