from sklearn.model_selection import BaseCrossValidator
from typing import Callable, Dict, List, Tuple, Union
from multiprocessing import Pool, cpu_count
from functools import partial
from abc import ABC
import pandas as pd
import numpy as np
import random
import time
import warnings

try:
    import cupy as cp
    import cudf

    HAS_GPU = True
except ImportError:
    import numpy as cp
    import pandas as cudf

    HAS_GPU = False

from .utils import create_cli_bar_chart, COLORS


class Climber(ABC):
    def __init__(
            self,
            objective: str,
            eval_metric: Callable,
            allow_negative_weights: bool = False,
            precision: float = 0.01,
            starting_model: str = "best",
            score_decimal_places: int = 3,
            random_state: int = 42,
            verbose: bool = True,
            n_jobs: int = -1,
            use_gpu: bool = False,
    ):
        self.objective = objective
        self.eval_metric = eval_metric
        self.allow_negative_weights = allow_negative_weights
        self.precision = precision
        self.starting_model = starting_model
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = cpu_count() if n_jobs == -1 else min(n_jobs, cpu_count())

        self.use_gpu = use_gpu and HAS_GPU
        if use_gpu and not HAS_GPU:
            warnings.warn("GPU libraries (cupy, cudf) not found. Falling back to CPU.")

        self.n_jobs = 1 if self.use_gpu else self.n_jobs

        self._score_decimal_places = score_decimal_places
        self._weight_decimal_places = max(2, int(-np.log10(self.precision)))

        self.best_score = None
        self.best_oof_preds = None
        self.history = None
        self._is_fitted = False

        self._validate_inputs()
        self._set_random_state()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'Climber':
        self._global_timer = time.time()

        X, y = self._validate_fit_inputs(X, y)

        if self.use_gpu:
            try:
                data_stream = cp.cuda.Stream()
                with data_stream:
                    X_gpu, y_gpu = cudf.DataFrame(X), cp.array(y)
                    X, y = X_gpu, y_gpu

                data_stream.synchronize()
            except Exception as e:
                warnings.warn(f"GPU data transfer failed: {str(e)}. Falling back to CPU.")
                self.use_gpu = False

        weight_range = self._get_weight_range()
        model_scores = self._get_individual_model_scores(X, y)
        first_model = self._get_starting_model(model_scores)

        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=self.objective == "maximize")
        sorted_models = [model for model, _ in sorted_models]
        X = X[sorted_models]

        self._print_header(model_scores)

        results = [first_model]
        coefs = [1]

        current_best_oof = X[first_model]
        remaining_oof_preds = X.drop(first_model, axis=1)
        initial_score = self.eval_metric(y, current_best_oof)

        iteration = 0
        stop_climbing = False
        last_score = initial_score

        iteration_times = []

        self.history = pd.DataFrame([{
            "iteration": iteration,
            "model": first_model,
            "score": float(initial_score),
            "improvement": float(initial_score),
            "time": float(time.time() - self._global_timer)
        }])

        if self.verbose:
            color = COLORS.GREEN if coefs[0] >= 0 else COLORS.RED
            print(
                f"   {color}{iteration:>{self.iter_width}}   {first_model:<{self.model_width}}   {coefs[0]:>{self.weight_width}.{self._weight_decimal_places}f}   {initial_score:>{self.score_width}.{self._score_decimal_places}f}     {'-':>{self.improvement_width}}     {'-':>{self.time_width}}{COLORS.END}")

        while not stop_climbing and remaining_oof_preds.shape[1] > 0:
            start_time_iter = time.time()
            iteration += 1

            potential_best_score = self.eval_metric(y,
                                                    current_best_oof) if self.objective == "maximize" else -self.eval_metric(
                y, current_best_oof)
            best_model, best_weight = None, None

            if self.use_gpu:
                best_model, best_weight = self._gpu_find_best_model_weight(
                    current_best_oof,
                    remaining_oof_preds,
                    y,
                    weight_range,
                    potential_best_score
                )

                if iteration % 5 == 0:
                    cp._default_memory_pool.free_all_blocks()
            else:
                for model in remaining_oof_preds.columns:
                    func_partial = partial(
                        self._compute_score,
                        current_preds=current_best_oof,
                        new_preds=remaining_oof_preds[model],
                        y_true=y
                    )

                    all_scores = self._parallelize_score_computation(
                        func_partial, list(weight_range))
                    for weight, score in all_scores:
                        if score > potential_best_score:
                            potential_best_score = score
                            best_model, best_weight = model, weight

            iter_time = time.time() - start_time_iter
            iteration_times.append(iter_time)

            if best_model is not None:
                results.append(best_model)
                coefs = [c * (1 - best_weight) for c in coefs] + [best_weight]

                if self.use_gpu:
                    with cp.cuda.Stream():
                        current_best_oof = self._array_weighted_sum(
                            current_best_oof,
                            remaining_oof_preds[best_model],
                            1 - best_weight,
                            best_weight
                        )
                else:
                    current_best_oof = (1 - best_weight) * current_best_oof + best_weight * remaining_oof_preds[
                        best_model]

                remaining_oof_preds = remaining_oof_preds.drop(best_model, axis=1)

                current_score = self.eval_metric(y, current_best_oof)
                improvement = abs(current_score - last_score)
                improvement_str = f"{improvement:.{self._score_decimal_places}f}"
                if self.verbose:
                    color = COLORS.GREEN if best_weight >= 0 else COLORS.RED
                    print(
                        f"   {color}{iteration:>{self.iter_width}}   {best_model:<{self.model_width}}   {best_weight:>{self.weight_width}.{self._weight_decimal_places}f}   {current_score:>{self.score_width}.{self._score_decimal_places}f}     {improvement_str:>{self.improvement_width}}     {iter_time:>{self.time_width}.2f}{COLORS.END}")

                last_score = current_score
                self.history = pd.concat([
                    self.history,
                    pd.DataFrame([{
                        "iteration": iteration,
                        "model": best_model,
                        "score": float(current_score),
                        "improvement": float(improvement),
                        "time": float(iter_time)
                    }])
                ], ignore_index=True)
            else:
                stop_climbing = True

        self.history["coef"] = coefs
        self.history["score"] = self.history["score"].astype(float)
        self.history["improvement"] = self.history["improvement"].astype(float)
        self.history["time"] = self.history["time"].astype(float)
        self.history["coef"] = self.history["coef"].astype(float)

        self.best_score = last_score.item() if self.use_gpu else last_score

        if self.use_gpu:
            self.best_oof_preds = cp.asnumpy(current_best_oof)
        else:
            self.best_oof_preds = current_best_oof

        self._is_fitted = True

        self._print_final_results()

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("Model must be fit before making predictions")

        if self.use_gpu and HAS_GPU:
            try:
                data_stream = cp.cuda.Stream()
                with data_stream:
                    X_gpu = cudf.DataFrame(X)
                data_stream.synchronize()

                batch_size = 50000 if X.shape[0] > 100000 else X.shape[0]
                num_batches = (X.shape[0] + batch_size - 1) // batch_size

                compute_streams = [cp.cuda.Stream() for _ in range(min(4, num_batches))]

                if num_batches > 1:
                    result = np.zeros(X.shape[0])
                    for i in range(num_batches):
                        stream_idx = i % len(compute_streams)
                        with compute_streams[stream_idx]:
                            start_idx = i * batch_size
                            end_idx = min(start_idx + batch_size, X.shape[0])

                            X_batch = X_gpu.iloc[start_idx:end_idx]
                            predictions_batch = cp.zeros(X_batch.shape[0])

                            for model, weight in zip(self.history["model"], self.history["coef"]):
                                predictions_batch += weight * cp.array(X_batch[model].values)

                            result[start_idx:end_idx] = cp.asnumpy(predictions_batch)

                    for stream in compute_streams:
                        stream.synchronize()

                    return result
                else:
                    with compute_streams[0]:
                        predictions = cp.zeros(X_gpu.shape[0])
                        model_columns = {}

                        for model in self.history["model"].unique():
                            model_columns[model] = cp.array(
                                X_gpu[model].values)

                        for model, weight in zip(self.history["model"], self.history["coef"]):
                            predictions += weight * model_columns[model]

                        compute_streams[0].synchronize()
                        return cp.asnumpy(predictions)

            except Exception as e:
                warnings.warn(f"GPU prediction failed: {str(e)}. Falling back to CPU.")
                self.use_gpu = False

        predictions = np.zeros(X.shape[0])
        for model, weight in zip(self.history["model"], self.history["coef"]):
            predictions += weight * X[model].values

        return predictions

    def _array_weighted_sum(self, a, b, weight_a, weight_b):
        if self.use_gpu:
            with cp.cuda.Stream():
                return cp.add(cp.multiply(weight_a, a), cp.multiply(weight_b, b))
        return weight_a * a + weight_b * b

    def _validate_inputs(self) -> None:
        if self.objective not in ["maximize", "minimize"]:
            raise ValueError("objective must be either 'maximize' or 'minimize'")

        if not callable(self.eval_metric):
            raise ValueError("eval_metric must be a callable function")

        if self.precision <= 0:
            raise ValueError("precision must be greater than 0")

    def _set_random_state(self) -> None:
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        if self.use_gpu:
            cp.random.seed(self.random_state)

    def _validate_fit_inputs(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> Tuple[pd.DataFrame, np.ndarray]:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise ValueError("y must be a pandas Series or numpy array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        if X.shape[1] == 0:
            raise ValueError("X must have at least one column")

        if isinstance(y, pd.Series):
            y = y.values

        return X, y

    def _get_weight_range(self) -> Union[np.ndarray, cp.ndarray]:
        weight_range = np.arange(-0.5, 0.51, self.precision) if self.allow_negative_weights else np.arange(
            self.precision, 0.51, self.precision)

        if self.use_gpu:
            return cp.array(weight_range)

        return weight_range

    def _get_individual_model_scores(self, X: Union[pd.DataFrame, cudf.DataFrame],
                                     y: Union[pd.Series, np.ndarray, cp.ndarray]) -> Dict[str, float]:
        return {
            model: self.eval_metric(y, X[model])
            for model in X.columns
        }

    def _get_starting_model(self, model_scores: Dict[str, float]) -> str:
        if self.starting_model == "best":
            return max(model_scores.items(), key=lambda x: x[1] if self.objective == "maximize" else -x[1])[0]
        elif self.starting_model == "random":
            return random.choice(list(model_scores.keys()))
        return self.starting_model

    def _print_header(self, model_scores: Dict[str, float]) -> None:
        if not self.verbose:
            return

        info = [
            ("Metric:", self.eval_metric.__name__),
            ("Objective:", self.objective),
            ("Precision:", self.precision),
            ("Allow negative weights:", self.allow_negative_weights),
            ("Starting model:", self.starting_model),
            ("Number of parallel jobs:", self.n_jobs),
            ("Number of models:", len(model_scores)),
            ("Using GPU:", self.use_gpu)
        ]

        print(f"{COLORS.BOLD}{COLORS.BLUE}Configuration{COLORS.END}\n")
        longest_label = max(len(label) for label, _ in info) + 5
        for label, value in info:
            print(f"   {label:<{longest_label}} {value}")

        print(f"\n\n{COLORS.BOLD}{COLORS.BLUE}Models{COLORS.END}\n")
        max_model_length = max(len(model) for model in model_scores.keys())

        best_model = max(model_scores.items(), key=lambda x: x[1] if self.objective == "maximize" else -x[1])[0]
        for line in create_cli_bar_chart(model_scores, self.objective, self._score_decimal_places):
            parts = line.split(' | ')
            if len(parts) == 3:
                model, score, bar = parts
                if model == best_model:
                    print(f"   {COLORS.GREEN}{model:<{max_model_length}} {score:>10} {bar} (best){COLORS.END}")
                else:
                    print(f"   {model:<{max_model_length}} {score:>10} {bar}")

        print(f"\n\n{COLORS.BOLD}{COLORS.BLUE}Running Hill Climbing{COLORS.END}\n")

        self.model_width = max_model_length
        self.iter_width = 4
        self.weight_width = 8
        self.score_width = 10
        self.improvement_width = 12
        self.time_width = 8

        self.total_width = 3
        self.total_width += self.iter_width + 3
        self.total_width += self.model_width + 3
        self.total_width += self.weight_width + 3
        self.total_width += self.score_width + 5
        self.total_width += self.improvement_width + 5
        self.total_width += self.time_width

        print(
            f"   {'Iter':>{self.iter_width}}   {'Model':<{self.model_width}}   {'Weight':>{self.weight_width}}   {'Score':>{self.score_width}}     {'Improvement':>{self.improvement_width}}     {'Time':>{self.time_width}}")
        print(f"   {'─' * (self.total_width - 3)}")

    def _compute_score(
            self,
            weight: float,
            current_preds: np.ndarray,
            new_preds: np.ndarray,
            y_true: np.ndarray
    ) -> Tuple[float, float]:
        ensemble_preds = (1 - weight) * current_preds + weight * new_preds
        score = self.eval_metric(y_true, ensemble_preds)

        return (weight, score) if self.objective == "maximize" else (weight, -score)

    def _parallelize_score_computation(
            self,
            func: Callable,
            weight_range: np.ndarray
    ) -> List[Tuple[float, float]]:
        if self.n_jobs == 1:
            if self.use_gpu and isinstance(weight_range, cp.ndarray):
                weight_range = cp.asnumpy(weight_range)
            return [func(weight) for weight in weight_range]

        if self.use_gpu and isinstance(weight_range, cp.ndarray):
            weight_range = cp.asnumpy(weight_range)

        num_cores = min(self.n_jobs, len(weight_range))
        with Pool(num_cores) as pool:
            return pool.map(func, weight_range)

    def _print_final_results(self) -> None:
        if not self.verbose:
            return

        print(f"\n\n{COLORS.BOLD}{COLORS.BLUE}Results{COLORS.END}\n")

        summary_info = [
            ("Number of models in ensemble:", f"{len(self.history)}")
        ]

        if len(self.history) > 1:
            improvement = abs(self.history["score"].iloc[-1] - self.history["score"].iloc[0])
            improvement_pct = improvement / abs(self.history["score"].iloc[0]) * 100 if self.history["score"].iloc[
                                                                                            0] != 0 else 0
            improvement_sign = "+" if improvement > 0 else ""
            improvement_str = f"{improvement_sign}{improvement:.{self._score_decimal_places}f} ({improvement_sign}{improvement_pct:.2f}%)"

            if improvement > 0:
                improvement_str = f"{COLORS.GREEN}{improvement_str}{COLORS.END}"
            elif improvement < 0:
                improvement_str = f"{COLORS.RED}{improvement_str}{COLORS.END}"

            summary_info.append((
                "Overall improvement:",
                improvement_str
            ))

        total_time = time.time() - self._global_timer
        iteration_times = self.history["time"].values
        summary_info.extend([
            ("Total time:", f"{total_time:.2f} seconds"),
            ("Average iteration time:", f"{sum(iteration_times) / len(iteration_times):.2f} seconds"),
            ("Final score:", f"{self.best_score:.{self._score_decimal_places}f}")
        ])

        longest_label = max(len(label) for label, _ in summary_info) + 5
        for label, value in summary_info:
            print(f"   {label:<{longest_label}} {value}")

    def _gpu_find_best_model_weight(self, current_preds, remaining_preds, y_true, weight_range, initial_best_score):
        if not self.use_gpu or not HAS_GPU:
            raise ValueError("GPU method called but GPU is not available or enabled")

        best_model = None
        best_weight = None
        best_score = initial_best_score
        batch_size = min(1000, len(weight_range))

        streams = [cp.cuda.Stream() for _ in range(min(4, len(remaining_preds.columns)))]

        for i, model in enumerate(remaining_preds.columns):
            with streams[i % len(streams)]:
                model_data = remaining_preds[model]

                for i in range(0, len(weight_range), batch_size):
                    batch_weights = weight_range[i:i + batch_size]
                    batch_weights_expanded = batch_weights.reshape(-1, 1)

                    all_weighted_preds = cp.zeros(
                        (len(batch_weights), len(y_true)))

                    for j, w in enumerate(batch_weights_expanded):
                        all_weighted_preds[j] = cp.add(
                            cp.multiply(1 - w, current_preds),
                            cp.multiply(w, model_data)
                        )

                    for j, w in enumerate(batch_weights):
                        pred = all_weighted_preds[j]
                        score = self.eval_metric(y_true, pred)
                        score = score if self.objective == "maximize" else -score

                        if score > best_score:
                            best_score = score
                            best_model = model
                            best_weight = float(w)

                cp.cuda.Stream.null.synchronize()

        for stream in streams:
            stream.synchronize()

        cp._default_memory_pool.free_all_blocks()

        return best_model, best_weight


class ClimberCV(Climber):
    def __init__(self, cv: BaseCrossValidator, **kwargs):
        super().__init__(**kwargs)
        self.cv = cv
        self.fold_scores = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._global_timer = time.time()

        X, y = self._validate_fit_inputs(X, y)

        if self.use_gpu:
            try:
                data_stream = cp.cuda.Stream()
                with data_stream:
                    X_gpu, y_gpu = cudf.DataFrame(X), cp.array(y)
                    X, y = X_gpu, y_gpu

                data_stream.synchronize()
            except Exception as e:
                warnings.warn(f"GPU data transfer failed: {str(e)}. Falling back to CPU.")
                self.use_gpu = False

        weight_range = self._get_weight_range()
        model_scores = self._get_individual_model_scores(X, y)

        self._print_header(model_scores)

        histories = []
        oof_preds = np.zeros(X.shape[0]) if not self.use_gpu else cp.zeros(X.shape[0])
        for fold_idx, (train_index, val_index) in enumerate(self.cv.split(X, y)):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y[train_index], y[val_index]

            val_scores = self._get_individual_model_scores(X_val, y_val)
            first_model = self._get_starting_model(val_scores)

            sorted_models = sorted(val_scores.items(), key=lambda x: x[1], reverse=self.objective == "maximize")
            sorted_models = [model for model, _ in sorted_models]
            X_train = X_train[sorted_models]
            X_val = X_val[sorted_models]

            results = [first_model]
            coefs = [1]

            best_train = X_train[first_model]
            best_val = X_val[first_model]

            remaining_train = X_train.drop(first_model, axis=1)
            remaining_val = X_val.drop(first_model, axis=1)

            initial_train_score = self.eval_metric(y_train, best_train)
            initial_val_score = self.eval_metric(y_val, best_val)

            iteration = 0
            stop_climbing = False

            iteration_times = []
            history = pd.DataFrame([{
                "iteration": iteration,
                "model": first_model,
                "train_score": float(initial_train_score),
                "val_score": float(initial_val_score),
                "time": float(time.time() - self._global_timer)
            }])

            if self.verbose:
                color = COLORS.GREEN if coefs[0] >= 0 else COLORS.RED
                print(
                    f"   {color}{fold_idx:>{self.fold_width}}   {iteration:>{self.iter_width}}   {first_model:<{self.model_width}}   {coefs[0]:>{self.weight_width}.{self._weight_decimal_places}f}   {initial_train_score:>{self.score_width}.{self._score_decimal_places}f}   {initial_val_score:>{self.score_width}.{self._score_decimal_places}f}   {'-':>{self.time_width}}{COLORS.END}")

            while not stop_climbing and remaining_train.shape[1] > 0:
                iteration += 1
                start_time_iter = time.time()

                potential_best_train_score = self.eval_metric(y_train,
                                                              best_train) if self.objective == "maximize" else -self.eval_metric(
                    y_train, best_train)
                potential_best_val_score = self.eval_metric(y_val,
                                                            best_val) if self.objective == "maximize" else -self.eval_metric(
                    y_val, best_val)
                best_model, best_weight = None, None

                if self.use_gpu:
                    for model in remaining_train.columns:
                        streams = [cp.cuda.Stream() for _ in range(min(4, len(weight_range)))]
                        batch_size = min(1000, len(weight_range))

                        for i in range(0, len(weight_range), batch_size):
                            with streams[i % len(streams)]:
                                batch_weights = weight_range[i:i + batch_size]
                                for w in batch_weights:
                                    train_pred = self._array_weighted_sum(best_train, remaining_train[model], 1 - w, w)
                                    train_score = self.eval_metric(y_train, train_pred)
                                    train_score = train_score if self.objective == "maximize" else -train_score

                                    if train_score > potential_best_train_score:
                                        val_pred = self._array_weighted_sum(best_val, remaining_val[model], 1 - w, w)
                                        val_score = self.eval_metric(y_val, val_pred)
                                        val_score = val_score if self.objective == "maximize" else -val_score

                                        potential_best_train_score = train_score
                                        potential_best_val_score = val_score
                                        best_model, best_weight = model, float(w)

                        for stream in streams:
                            stream.synchronize()

                        if iteration % 5 == 0:
                            cp._default_memory_pool.free_all_blocks()
                else:
                    for model in remaining_train.columns:
                        func_partial = partial(
                            self._compute_score,
                            current_preds=best_train,
                            new_preds=remaining_train[model],
                            y_true=y_train
                        )

                        all_scores = self._parallelize_score_computation(func_partial, weight_range)
                        for weight, score in all_scores:
                            if score > potential_best_train_score:
                                val_pred = (1 - weight) * best_val + weight * remaining_val[model]
                                val_score = self.eval_metric(y_val, val_pred)
                                val_score = val_score if self.objective == "maximize" else -val_score

                                potential_best_train_score = score
                                potential_best_val_score = val_score
                                best_model, best_weight = model, weight

                iter_time = time.time() - start_time_iter
                iteration_times.append(iter_time)

                if best_model is not None:
                    results.append(best_model)
                    coefs = [c * (1 - best_weight) for c in coefs] + [best_weight]

                    if self.use_gpu:
                        best_train = self._array_weighted_sum(best_train, remaining_train[best_model], 1 - best_weight,
                                                              best_weight)
                        best_val = self._array_weighted_sum(best_val, remaining_val[best_model], 1 - best_weight,
                                                            best_weight)
                    else:
                        best_train = (1 - best_weight) * best_train + best_weight * remaining_train[best_model]
                        best_val = (1 - best_weight) * best_val + best_weight * remaining_val[best_model]

                    remaining_train = remaining_train.drop(best_model, axis=1)
                    remaining_val = remaining_val.drop(best_model, axis=1)

                    train_score = potential_best_train_score if self.objective == "maximize" else - potential_best_train_score
                    val_score = potential_best_val_score if self.objective == "maximize" else - potential_best_val_score

                    if self.verbose:
                        color = COLORS.GREEN if best_weight >= 0 else COLORS.RED
                        print(
                            f"   {color}{fold_idx:>{self.fold_width}}   {iteration:>{self.iter_width}}   {best_model:<{self.model_width}}   {best_weight:>{self.weight_width}.{self._weight_decimal_places}f}   {train_score:>{self.score_width}.{self._score_decimal_places}f}   {val_score:>{self.score_width}.{self._score_decimal_places}f}   {iter_time:>{self.time_width}.2f}{COLORS.END}")

                    history = pd.concat([
                        history,
                        pd.DataFrame([{
                            "iteration": iteration,
                            "model": best_model,
                            "train_score": float(train_score),
                            "val_score": float(val_score),
                            "time": float(iter_time)
                        }])
                    ], ignore_index=True)

                else:
                    stop_climbing = True

            history["coef"] = coefs
            history["fold"] = fold_idx
            histories.append(history)

            if self.use_gpu:
                fold_preds = cp.zeros(len(val_index))
                for model, weight in zip(history["model"], history["coef"]):
                    fold_preds += weight * X_val[model]
                oof_preds[val_index] = fold_preds
            else:
                oof_preds[val_index] = np.zeros(X_val.shape[0])
                for model, weight in zip(history["model"], history["coef"]):
                    oof_preds[val_index] += weight * X_val[model].values

            fold_score = self.eval_metric(y_val, oof_preds[val_index])
            fold_score = fold_score.item() if self.use_gpu else fold_score
            self.fold_scores.append(fold_score)

            if self.verbose and fold_idx != self.cv.n_splits - 1:
                print(f"   {'─' * (self.total_width - 3)}")

        self.history = pd.concat(histories)

        if self.use_gpu:
            self.best_oof_preds = cp.asnumpy(oof_preds)
        else:
            self.best_oof_preds = oof_preds

        self.history["train_score"] = self.history["train_score"].astype(float)
        self.history["val_score"] = self.history["val_score"].astype(float)
        self.history["time"] = self.history["time"].astype(float)
        self.history["coef"] = self.history["coef"].astype(float)

        self.best_score = self.eval_metric(y, oof_preds)
        self.best_score = self.best_score.item() if self.use_gpu else self.best_score
        self._is_fitted = True

        self._print_final_results()

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("Model must be fit before making predictions")

        if self.use_gpu and HAS_GPU:
            try:
                data_stream = cp.cuda.Stream()
                with data_stream:
                    X_gpu = cudf.DataFrame(X)
                    X = X_gpu
                data_stream.synchronize()

                preds = cp.zeros(X.shape[0])
                for i in range(self.cv.n_splits):
                    fold_preds = cp.zeros(X.shape[0])
                    for model, weight in zip(self.history[self.history["fold"] == i]["model"],
                                             self.history[self.history["fold"] == i]["coef"]):
                        fold_preds += weight * cp.array(X[model].values)
                    preds += fold_preds

                cp._default_memory_pool.free_all_blocks()
                return cp.asnumpy(preds / self.cv.n_splits)

            except Exception as e:
                warnings.warn(f"GPU prediction failed: {str(e)}. Falling back to CPU.")
                self.use_gpu = False

        preds = np.zeros(X.shape[0])
        for i in range(self.cv.n_splits):
            fold_preds = np.zeros(X.shape[0])
            for model, weight in zip(self.history[self.history["fold"] == i]["model"],
                                     self.history[self.history["fold"] == i]["coef"]):
                fold_preds += weight * X[model].values
            preds += fold_preds
        return preds / self.cv.n_splits

    def _print_header(self, model_scores: Dict[str, float]) -> None:
        if not self.verbose:
            return

        info = [
            ("Metric:", self.eval_metric.__name__),
            ("Objective:", self.objective),
            ("Precision:", self.precision),
            ("Allow negative weights:", self.allow_negative_weights),
            ("Starting model:", self.starting_model),
            ("Number of parallel jobs:", self.n_jobs),
            ("Number of models:", len(model_scores)),
            ("Number of folds:", self.cv.n_splits)
        ]

        print(f"{COLORS.BOLD}{COLORS.BLUE}Configuration{COLORS.END}\n")
        longest_label = max(len(label) for label, _ in info) + 5
        for label, value in info:
            print(f"   {label:<{longest_label}} {value}")

        print(f"\n\n{COLORS.BOLD}{COLORS.BLUE}Models{COLORS.END}\n")
        max_model_length = max(len(model) for model in model_scores.keys())

        best_model = max(model_scores.items(), key=lambda x: x[1] if self.objective == "maximize" else -x[1])[0]
        for line in create_cli_bar_chart(model_scores, self.objective, self._score_decimal_places):
            parts = line.split(' | ')
            if len(parts) == 3:
                model, score, bar = parts
                if model == best_model:
                    print(
                        f"   {COLORS.GREEN}{model:<{max_model_length}} {score:>10} {bar} (best){COLORS.END}")
                else:
                    print(f"   {model:<{max_model_length}} {score:>10} {bar}")

        print(f"\n\n{COLORS.BOLD}{COLORS.BLUE}Running Hill Climbing{COLORS.END}\n")

        self.model_width = max_model_length
        self.iter_width = 4
        self.weight_width = 8
        self.score_width = 10
        self.improvement_width = 12
        self.time_width = 8
        self.fold_width = 4

        self.total_width = 3
        self.total_width += self.fold_width + 3
        self.total_width += self.iter_width + 3
        self.total_width += self.model_width + 3
        self.total_width += self.weight_width + 3
        self.total_width += self.score_width + 3
        self.total_width += self.score_width + 3
        self.total_width += self.time_width

        print(
            f"   {'Fold':>{self.fold_width}}   {'Iter':>{self.iter_width}}   {'Model':<{self.model_width}}   {'Weight':>{self.weight_width}}   {'Train':>{self.score_width}}   {'Val':>{self.score_width}}   {'Time':>{self.time_width}}")
        print(f"   {'─' * (self.total_width - 3)}")

    def _print_final_results(self) -> None:
        if not self.verbose:
            return

        print(f"\n\n{COLORS.BOLD}{COLORS.BLUE}Results{COLORS.END}\n")

        summary_info = [
            ("Number of models in ensemble:",
             f"{len(self.history['model'].unique())}")
        ]

        if len(self.history) > 1:
            improvement = abs(self.history["val_score"].iloc[-1] - self.history["val_score"].iloc[0])
            improvement_pct = improvement / abs(self.history["val_score"].iloc[0]) * 100 if \
                self.history["val_score"].iloc[0] != 0 else 0
            improvement_sign = "+" if improvement > 0 else ""
            improvement_str = f"{improvement_sign}{improvement:.{self._score_decimal_places}f} ({improvement_sign}{improvement_pct:.2f}%)"

            if improvement > 0:
                improvement_str = f"{COLORS.GREEN}{improvement_str}{COLORS.END}"
            elif improvement < 0:
                improvement_str = f"{COLORS.RED}{improvement_str}{COLORS.END}"

            summary_info.append((
                "Overall improvement:",
                improvement_str
            ))

        total_time = time.time() - self._global_timer
        iteration_times = self.history["time"].values
        summary_info.extend([
            ("Total time:", f"{total_time:.2f} seconds"),
            ("Average iteration time:",
             f"{sum(iteration_times) / len(iteration_times):.2f} seconds"),
            ("Final score:",
             f"{self.best_score:.{self._score_decimal_places}f}")
        ])

        longest_label = max(len(label) for label, _ in summary_info) + 5
        for label, value in summary_info:
            print(f"   {label:<{longest_label}} {value}")
