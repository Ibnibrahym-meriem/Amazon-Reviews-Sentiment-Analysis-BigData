# mlflow_client.py - Place in your project root
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# MLflow configuration
MLFLOW_TRACKING_URI = "http://localhost:5001"  # Or http://mlflow:5000 if in Docker
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
            # Search for runs with promoted tag = True
            runs = self.client.search_runs(
                experiment_ids=[self.experiment.experiment_id],
                filter_string="tags.promoted = 'True'",
                max_results=1,
                order_by=["start_time DESC"]
            )
            
            if runs:
                latest_run = runs[0]
                
                # Get run name (from tags or generate from start_time)
                run_name = latest_run.data.tags.get("run_name", 
                           latest_run.data.tags.get("mlflow.runName",
                           f"Run {latest_run.info.run_id[:8]}"))
                
                # Convert timestamp to datetime
                retrained_at = datetime.fromtimestamp(latest_run.info.start_time / 1000)
                
                # Get decision status
                decision = latest_run.data.tags.get("decision", "UNKNOWN")
                
                return {
                    "run_name": run_name,
                    "run_id": latest_run.info.run_id,
                    "f1_score": round(latest_run.data.metrics.get("f1_macro", 0) * 100, 1),
                    "accuracy": round(latest_run.data.metrics.get("accuracy", 0) * 100, 1),
                    "recall_negative": round(latest_run.data.metrics.get("recall_negative", 0) * 100, 1),
                    "f1_improvement": latest_run.data.metrics.get("f1_improvement", 0),
                    "n_training_samples": int(latest_run.data.params.get("n_training", 0)),
                    "retrained_at": retrained_at,
                    "status": decision,  # PROMOTED or REJECTED
                    "promoted": decision == "PROMOTED",
                    "mlflow_url": f"{MLFLOW_TRACKING_URI}/#/experiments/{self.experiment.experiment_id}/runs/{latest_run.info.run_id}"
                }
            else:
                # Fallback to most recent run if no promoted tag
                runs = self.client.search_runs(
                    experiment_ids=[self.experiment.experiment_id],
                    max_results=1,
                    order_by=["start_time DESC"]
                )
                if runs:
                    latest_run = runs[0]
                    return {
                        "run_name": "Initial Model",
                        "run_id": latest_run.info.run_id,
                        "f1_score": round(latest_run.data.metrics.get("f1_macro", 0) * 100, 1),
                        "accuracy": round(latest_run.data.metrics.get("accuracy", 0) * 100, 1),
                        "recall_negative": round(latest_run.data.metrics.get("recall_negative", 0) * 100, 1),
                        "f1_improvement": 0,
                        "n_training_samples": int(latest_run.data.params.get("n_training", 0)),
                        "retrained_at": datetime.fromtimestamp(latest_run.info.start_time / 1000),
                        "status": "INITIAL",
                        "promoted": True,
                        "mlflow_url": f"{MLFLOW_TRACKING_URI}/#/experiments/{self.experiment.experiment_id}/runs/{latest_run.info.run_id}"
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
                order_by=["start_time ASC"],
                max_results=limit
            )
            
            history = []
            for run in runs:
                # Get decision tag
                decision = run.data.tags.get("decision", "PENDING")
                promoted = run.data.tags.get("promoted", "False") == "True"
                
                # Convert timestamp
                run_time = datetime.fromtimestamp(run.info.start_time / 1000)
                
                history.append({
                    "run_id": run.info.run_id,
                    "run_name": run.data.tags.get("run_name", 
                                run.data.tags.get("mlflow.runName",
                                run_time.strftime("%Y-%m-%d %H:%M"))),
                    "time": run_time.isoformat(),
                    "timestamp": run.info.start_time,
                    "f1_score": round(run.data.metrics.get("f1_macro", 0) * 100, 2),
                    "accuracy": round(run.data.metrics.get("accuracy", 0) * 100, 2),
                    "recall_negative": round(run.data.metrics.get("recall_negative", 0) * 100, 2),
                    "f1_negative": round(run.data.metrics.get("f1_negative", 0) * 100, 2),
                    "train_time": run.data.metrics.get("train_time_seconds", 0),
                    "n_samples": int(run.data.params.get("n_training", 0)),
                    "status": decision,
                    "promoted": promoted,
                    "improvement": run.data.metrics.get("f1_improvement", 0),
                    "mlflow_url": f"{MLFLOW_TRACKING_URI}/#/experiments/{self.experiment.experiment_id}/runs/{run.info.run_id}"
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting performance history: {e}")
            return []
    
    def get_latest_retraining(self):
        """Get only the latest retraining info"""
        history = self.get_performance_history(limit=5)
        return history[0] if history else None