"""Helpers for multi-session task loading in figure notebooks."""

from __future__ import annotations

import numpy as np


def subset_task_params_sessions(task_params, model_session_ids):
    """
    Restrict ``task_params`` to a subset of sessions by original model indices.

    Keeps ``train_inds_list`` / ``val_inds_list`` aligned with the selected sessions so
    downstream dataset creation (e.g. SWM multi-session) uses the same train/eval splits
    as at training time.

    Args:
        task_params (dict): training-time task params containing a ``sessions`` list.
        model_session_ids (iterable): session indices in the *original* model/vae order.

    Returns:
        dict: shallow copy of task_params with filtered ``sessions`` (and optional index lists).
    """
    model_session_ids = [int(s) for s in model_session_ids]
    all_sessions = task_params["sessions"]
    n_all = len(all_sessions)
    for sid in model_session_ids:
        if sid < 0 or sid >= n_all:
            raise ValueError(
                f"model_session_id {sid} out of range for {n_all} sessions"
            )

    out = task_params.copy()
    out["sessions"] = [all_sessions[sid] for sid in model_session_ids]

    if "train_inds_list" in task_params and "val_inds_list" in task_params:
        out["train_inds_list"] = [
            task_params["train_inds_list"][sid] for sid in model_session_ids
        ]
        out["val_inds_list"] = [
            task_params["val_inds_list"][sid] for sid in model_session_ids
        ]

    return out


def attach_model_session_ids(task, model_session_ids):
    """Record original VAE session indices on a (possibly subset) task."""
    task.model_session_ids = [int(s) for s in model_session_ids]
    return task


def get_task_session(task, model_sess_id):
    """
    Return the session dataset for a model/VAE session id (handles subset tasks).

    When ``task.model_session_ids`` is set (see ``attach_model_session_ids``),
    ``model_sess_id`` is the original training/VAE index; otherwise it indexes
    ``task.sessions`` directly.
    """
    model_sess_id = int(model_sess_id)
    if hasattr(task, "model_session_ids"):
        idx = task.model_session_ids.index(model_sess_id)
    else:
        idx = model_sess_id
    return task.sessions[idx]
