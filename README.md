# FedMM-D

FedMM Research Project (Provably), advised by Dr. Jie Xu. You are also welcome to share your ideas by opening an issue or dropping me an email at [ypeng@miami.edu](mailto:ypeng@miami.edu).

<img src="fig/memory.jpg" width="400">

- [data](#data)
  - [LICENSE.txt](#licensetxt)
  - [micmic3..tar.gz](#micmic3targz)
  - [mimic3_demo](#mimic3_demo)
    - [10032_episode1_timeseries.csv](#10032_episode1_timeseriescsv)
    - [10042_episode1_timeseries.csv](#10042_episode1_timeseriescsv)
    - [listfile.csv](#listfilecsv)
  - [modelnet-40_demo](#modelnet-40_demo)
    - [guitar_0127.off](#guitar_0127off)
      - [guitar_0127.0.png](#guitar_01270png)
      - [guitar_0127.1.png](#guitar_01271png)
      - ...
    - [piano_0031.off](#piano_0031off)
      - [piano_0031.0.png](#piano_00310png)
      - [piano_0031.1.png](#piano_00311png)
      - ...
  - [raw](#raw)
    - [ADMISSIONS.csv](#admissionscsv)
    - [CALLOUT.csv](#calloutcsv)
    - [CAREGIVERS.csv](#caregiverscsv)
    - [PATIENTS.csv](#patientscsv)
- [fedmm](#fedmm-1)
  - [fedmm_yz_0115.py](#fedmm_yz_0115py)
  - [__init__.py](#__init__py)
  - [load_m40.py](#load_m40py)
  - [utilities](#utilities)
    - [__init__.py](#__init__py-1)
    - [utils.py](#utilspy)
  - [worker_1.py](#worker_1py)
  - [worker_2.py](#worker_2py)
- [fig](#fig)
  - [m3.jpg](#m3jpg)
  - [m40.jpg](#m40jpg)
  - [memory.jpg](#memoryjpg)
- [log](#log)
  - [fedmm_log_20240114172553.txt](#fedmm_log_20240114172553txt)
  - [memory_m40.png](#memory_m40png)
- [models](#models)
  - [__init__.py](#__init__py-2)
  - [resnet.py](#resnetpy)
  - [res.py](#respy)
- [utils](#utils)
  - [metrics.py](#metricspy)
  - [utils_fedmm.py](#utils_fedmmpy)
- [FedMM.yml](#fedmmyml)
- [README.md](#readmemd)


You can access the ModelNet-40 data [here](https://modelnet.cs.princeton.edu/).

You can access the (full) Mimic-III data [here](https://physionet.org/content/mimiciii/1.4/). Download the files using your terminal with the following command:
![FedMM-D](fig/m3.jpg)
```bash
wget -r -N -c -np --user yzpeng --ask-password https://physionet.org/files/mimiciii/1.4/
