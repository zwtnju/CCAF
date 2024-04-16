from pynvml import *
import logging

from adapters import DoubleSeqBnConfig, SeqBnConfig, ParBnConfig, AdapterConfig

logger = logging.getLogger(__name__)


def show_gpu():
    nvmlInit()
    device_count = nvmlDeviceGetCount()
    total_memory = 0
    total_used = 0

    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)

        total_memory += (info.total // 1048576)
        total_used += (info.used // 1048576)

    logger.info("name: [{}], num: [{}], total: [{} M], used: [{} M]."
                .format(nvmlDeviceGetName(nvmlDeviceGetHandleByIndex(0)), device_count, total_memory, total_used))

    nvmlShutdown()


def get_model_size(model, required=True):
    if required:
        model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        model_size = sum(p.numel() for p in model.parameters())
    return "{}M".format(round(model_size / 1e+6))


def get_adapter(adapter_type):
    if "houlsby" in adapter_type:
        adapter_config = DoubleSeqBnConfig()
    elif "pfeiffer" in adapter_type:
        adapter_config = SeqBnConfig()
    elif "parallel" in adapter_type:
        adapter_config = ParBnConfig()
    else:
        adapter_config = AdapterConfig()
    return adapter_config


ADAPTER_TYPE = ["houlsby", "pfeiffer", "parallel"]
