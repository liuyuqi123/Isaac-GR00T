
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from torch.utils.data import Dataset
from tqdm import tqdm

from gr00t.utils.video import get_all_frames, get_frames_by_timestamps

from .embodiment_tags import EmbodimentTag
from .schema import (
    DatasetMetadata,
    DatasetStatisticalValues,
    LeRobotModalityMetadata,
    LeRobotStateActionMetadata,
    StateActionMetadata,
)
from .transform import ComposedModalityTransform

LE_ROBOT_MODALITY_FILENAME = "meta/modality.json"
LE_ROBOT_EPISODE_FILENAME = "meta/episodes.jsonl"
LE_ROBOT_TASKS_FILENAME = "meta/tasks.jsonl"
LE_ROBOT_INFO_FILENAME = "meta/info.json"
LE_ROBOT_STATS_FILENAME = "meta/stats.json"
LE_ROBOT_DATA_FILENAME = "data/*/*.parquet"

from gr00t.data.dataset import LeRobotSingleDataset, ModalityConfig


def extract_3d_array_from_series(series_data, expected_shape: tuple = None) -> np.ndarray:
    """
    从 pandas Series 中提取完整的 3 维数组 (T, dim, horizon)。

    处理多种嵌套情况：
    1. Series[ndarray(dtype=object)] -> 内层是 list/ndarray
    2. Series[str] -> 需要 literal_eval
    3. Series[list] -> 直接处理
    4. 任意深度的嵌套 list/tuple/ndarray

    Args:
        series_data: pandas Series，包含嵌套的动作数据
        expected_shape: 可选的期望形状 (T, dim, horizon)，用于验证

    Returns:
        np.ndarray: 形状为 (T, dim, horizon) 的 float32 数组

    Raises:
        ValueError: 当数据维度不一致或无法解析时
    """
    from ast import literal_eval

    def deep_unwrap(obj):
        """递归展开任意嵌套结构，直到获得纯数值"""
        # 1. 处理字符串
        if isinstance(obj, str):
            try:
                obj = literal_eval(obj)
            except (ValueError, SyntaxError):
                raise ValueError(f"无法解析字符串: {obj[:100]}...")

        # 2. 处理 numpy object 数组（最外层壳）
        if isinstance(obj, np.ndarray) and obj.dtype == object:
            if obj.size == 1:
                # 单元素 object 数组，递归展开
                return deep_unwrap(obj.item())
            else:
                # 多元素 object 数组，逐个展开
                return [deep_unwrap(item) for item in obj]

        # 3. 处理 list/tuple（中间层壳）
        if isinstance(obj, (list, tuple)):
            if len(obj) == 1 and isinstance(obj[0], (list, tuple, np.ndarray)):
                # 单元素容器，递归展开
                return deep_unwrap(obj[0])
            else:
                # 多元素容器，逐个展开
                return [deep_unwrap(item) for item in obj]

        # 4. 处理规则 numpy 数组（已是数值）
        if isinstance(obj, np.ndarray) and obj.dtype != object:
            return obj

        # 5. 已经是纯数值
        return obj

    # ==================== 主流程 ====================

    # 步骤 1: 展开所有行
    unwrapped_rows = []
    for idx, row_data in enumerate(series_data):
        try:
            unwrapped = deep_unwrap(row_data)
            unwrapped_rows.append(unwrapped)
        except Exception as e:
            raise ValueError(f"处理第 {idx} 行时出错: {e}\n原始数据: {row_data}")

    # 步骤 2: 转换为 numpy 数组
    try:
        # 尝试直接堆叠（适用于规则数组）
        array_3d = np.array(unwrapped_rows, dtype=np.float32)
    except ValueError as e:
        # 如果失败，检查每行的形状
        shapes = [np.array(row).shape for row in unwrapped_rows]
        unique_shapes = set(shapes)
        if len(unique_shapes) > 1:
            raise ValueError(
                f"数据维度不一致！\n"
                f"发现 {len(unique_shapes)} 种不同形状: {unique_shapes}\n"
                f"前 5 行形状: {shapes[:5]}"
            )
        raise ValueError(f"数组转换失败: {e}")

    # # 步骤 3: 验证维度
    # if array_3d.ndim != 3:
    #     raise ValueError(
    #         f"期望 3 维数组 (T, dim, horizon)，实际得到 {array_3d.ndim} 维: {array_3d.shape}"
    #     )
    #
    # # 步骤 4: 可选的形状验证
    # if expected_shape is not None:
    #     if array_3d.shape != expected_shape:
    #         print(
    #             f"⚠️ 警告: 形状不匹配\n"
    #             f"  期望: {expected_shape}\n"
    #             f"  实际: {array_3d.shape}"
    #         )

    return array_3d


class LeRobotUmiSingleDataset(LeRobotSingleDataset):

    def __init__(
        self,
        dataset_path: Path | str,
        modality_configs: dict[str, ModalityConfig],
        embodiment_tag: str | EmbodimentTag,
        video_backend: str = "torchcodec",
        video_backend_kwargs: dict | None = None,
        transforms: ComposedModalityTransform | None = None,
    ):

        super().__init__(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
            video_backend_kwargs=video_backend_kwargs,
            transforms=transforms,
        )

    def get_state_or_action(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Get the state or action data for a trajectory by a base index.
        If the step indices are out of range, pad with the data:
            if the data is stored in absolute format, pad with the first or last step data;
            otherwise, pad with zero.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The data for the trajectory and step indices.
        """
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]

        # this handles action.task_progress if specified
        if key == "action.task_progress":
            # Get frame_index array and apply proper bounds checking and padding
            frame_index_array = self.curr_traj_data["frame_index"].to_numpy()
            # Use retrieve_data_and_pad to handle out-of-bounds indices
            frame_index = self.retrieve_data_and_pad(
                array=frame_index_array,
                step_indices=step_indices,
                max_length=max_length,
                padding_strategy="first_last",  # Use first/last for task progress
            )
            # get the task progress by using "frame index / trajectory length"
            progress = frame_index / max_length
            progress = progress.reshape(-1, 1)
            return progress

        assert key.startswith(modality + "."), f"{key} must start with {modality + '.'}, got {key}"
        # Get the sub-key, e.g. state.joint_angles -> joint_angles
        key = key.replace(modality + ".", "")
        # Get the lerobot key
        le_state_or_action_cfg = getattr(self.lerobot_modality_meta, modality)
        le_key = le_state_or_action_cfg[key].original_key
        if le_key is None:
            le_key = key
        # Get the data array, shape: (T, D)
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert le_key in self.curr_traj_data.columns, f"No {le_key} found in {trajectory_id=}"

        # # 原始实现，不能正确读取数据维度
        # data_array: np.ndarray = np.stack(self.curr_traj_data[le_key])  # type: ignore

        # 递归剥壳
        data_array = extract_3d_array_from_series(
            self.curr_traj_data[le_key],
            expected_shape=(max_length, None, None)  # 只验证时间维度
        )
        # print(f"✅ 成功提取 {le_key}: {data_array.shape}")

        # # 原始实现
        # # 维度检查，但是在修改后可能不需要了
        # if data_array.ndim == 1:
        #     assert (
        #         data_array.shape[0] == max_length
        #     ), f"Expected 1D array with length {max_length}, got {data_array.shape} array"
        #     data_array = data_array.reshape(-1, 1)
        # assert data_array.ndim == 2, f"Expected 2D array, got {data_array.shape} array"

        # 关键步骤：根据modality中定义的索引范围，切片出对应的数据
        le_indices = np.arange(
            le_state_or_action_cfg[key].start,
            le_state_or_action_cfg[key].end,
        )

        # # 原始实现
        # data_array = data_array[:, le_indices]

        # 需要应对modality为action的情况，切片后直接返回base_index对应的数据
        data_array = data_array[..., le_indices]  # 总是切片最后一维

        # ---------- 2. action 特殊分支 ----------
        if modality == "action":
            d = data_array[base_index]
            return d

        # Get the state or action configuration
        state_or_action_cfg = getattr(self.metadata.modalities, modality)[key]

        # Pad the data
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=step_indices,
            max_length=max_length,
            padding_strategy="first_last" if state_or_action_cfg.absolute else "zero",
        )







