"""
类人脑双系统全闭环AI架构 - 端侧部署模块
Edge Deployment Module

支持：
- INT4量化
- 安卓手机部署
- 树莓派4B部署
- 离线运行
"""

from .edge_deployment import (
    EdgeDeploymentConfig,
    EdgeDeploymentManager,
    INT4Quantizer,
    EdgeHardwareAdapter,
    OfflineManager,
    deploy_to_edge
)

__all__ = [
    'EdgeDeploymentConfig',
    'EdgeDeploymentManager',
    'INT4Quantizer',
    'EdgeHardwareAdapter',
    'OfflineManager',
    'deploy_to_edge'
]
