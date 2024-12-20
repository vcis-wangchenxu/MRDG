import os
import json
import platform
import collections

import GPUtil
import psutil

from datetime import datetime


def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def gpu_info():
    from tabulate import tabulate
    gpus = GPUtil.getGPUs()
    list_gpus = []
    for gpu in gpus:
        # get the GPU id
        gpu_id = gpu.id
        # name of GPU
        gpu_name = gpu.name
        # get % percentage of GPU usage of that GPU
        gpu_load = f"{gpu.load*100}%"
        # get free memory in MB format
        gpu_free_memory = f"{gpu.memoryFree}MB"
        # get used memory
        gpu_used_memory = f"{gpu.memoryUsed}MB"
        # get total memory
        gpu_total_memory = f"{gpu.memoryTotal}MB"
        # get GPU temperature in Celsius
        gpu_temperature = f"{gpu.temperature} °C"
        gpu_uuid = gpu.uuid
        list_gpus.append(collections.OrderedDict({
            "gpu_id": gpu_id,
            "gpu_name": gpu_name,
            "gpu_load": gpu_load,
            "gpu_free_memory": gpu_free_memory,
            "gpu_used_memory": gpu_used_memory,
            "gpu_total_memory": gpu_total_memory,
            "gpu_temperature": gpu_temperature,
            "gpu_uuid": gpu_uuid
        }))
    return list_gpus


def system_info():
    uname = platform.uname()
    res = collections.OrderedDict()
    res["system"] = uname.system
    res["node name"] = uname.node
    res["release"] = uname.release
    res["version"] = uname.version
    res["machine"] = uname.machine
    res["processor"] = uname.processor
    # Boot Time
    boot_time_timestamp = psutil.boot_time()
    bt = datetime.fromtimestamp(boot_time_timestamp)
    res["boot time"] = f"{bt.year}/{bt.month}/{bt.day} {bt.hour}:{bt.minute}:{bt.second}"
    return res


def cpu_info():
    # number of cores
    res = collections.OrderedDict()
    res["physical cores"] = psutil.cpu_count(logical=False)
    res["total cores"] = psutil.cpu_count(logical=True)
    # CPU frequencies
    cpufreq = psutil.cpu_freq()
    res["max Frequency"] = f"{cpufreq.max:.2f}Mhz"
    res["min Frequency"] = f"{cpufreq.min:.2f}Mhz"
    res["current frequency"] = f"{cpufreq.current:.2f}Mhz"
    return res


def memory_info():
    # Memory Information
    # get the memory details
    res = collections.OrderedDict()
    svmem = psutil.virtual_memory()
    res["Total"] = f"{get_size(svmem.total)}"
    res["Available"] = f"{get_size(svmem.available)}"
    # res["Used"] = f"{get_size(svmem.used)}"
    # res["Percentage"] = f"{svmem.percent}%"
    return res



def get_all_info():
    info = collections.OrderedDict()
    info['system'] = system_info()
    info['cpu'] = cpu_info()
    info['gpu'] = gpu_info()
    info['memory'] = memory_info()
    return info


def save_hardware_info(save_path, filename='hardware_info.json'):
    all_info = get_all_info()
    with open(os.path.join(save_path, filename), 'w') as f:
        json.dump(all_info, f, indent=4)
