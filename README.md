CoFLow Research Project. You are also welcome to share your ideas by opening an issue or dropping me an email at [pengy1@ufl.edu](mailto:pengy1@ufl.edu).

The experiment is divided into two parts. One part involves utilizing two large datasets and implementing them on powerful GPUs. The other part consists of testbed experiments using a small dataset, utilizing a total of 6 Jetson devices: 3 Nvidia Jetson TX2 and 3 Nvidia Jetson Xavier NX. [Watch the video](https://www.youtube.com/watch?v=WtCe4LvQ7do)
[![Watch the video](https://img.youtube.com/vi/WtCe4LvQ7do/0.jpg)](https://www.youtube.com/watch?v=WtCe4LvQ7do)

Below shows the memory usage during GPU experiments:

<img src="fig/memory.jpg" width="300">

You can access the ModelNet-40 data [here](https://modelnet.cs.princeton.edu/).

You can access the Mimic-III data [here](https://physionet.org/content/mimiciii-demo/1.4/). Download the full [files](https://physionet.org/content/mimiciii/1.4/) using your terminal with the following command (**replace username "yzpeng" and use your password**):

```bash
wget -r -N -c -np --user yzpeng --ask-password https://physionet.org/files/mimiciii/1.4/
```

<!-- Please note that although MIMIC-III is a public data set, when it comes to medical privacy information, you still need to complete a brief online course to download the raw files. Detailed data requirements can be found [here](https://physionet.org/content/mimiciii/1.4/). -->

The following command takes MIMIC-III CSVs, generates one directory per `SUBJECT_ID` and writes ICU stay information to `data/{SUBJECT_ID}/stays.csv`, diagnoses to `data/{SUBJECT_ID}/diagnoses.csv`, and events to `data/{SUBJECT_ID}/events.csv`. 
```bash
python -m mimic3benchmark.scripts.extract_subjects /a/bear.cs.fiu.edu./disk/bear-c/users/rxm1351/yz/0108fedmm/mimic3-benchmarks/physionet.org/files/mimiciii/1.4/ data/root/
```

The following command attempts to fix some issues (ICU stay ID is missing) and removes the events that have missing information. About 80% of events remain after removing all suspicious rows.

```bash
python -m mimic3benchmark.scripts.validate_events data/root/
```

The next command breaks up per-subject data into separate episodes (pertaining to ICU stays). Time series of events are stored in ```{SUBJECT_ID}/episode{#}_timeseries.csv``` (where # counts distinct episodes) while episode-level information (patient age, gender, ethnicity, height, weight) and outcomes (mortality, length of stay, diagnoses) are stores in ```{SUBJECT_ID}/episode{#}.csv```. This script requires two files, one that maps event ITEMIDs to clinical variables and another that defines valid ranges for clinical variables (for detecting outliers, etc.). Outlier detection is disabled in the current version.

```bash
python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root/
```

The next command splits the whole dataset into training and testing sets. Note that the train/test split is the same of all tasks.

```bash
python -m mimic3benchmark.scripts.split_train_and_test data/root/
```
	
The following command will generate a task-specific dataset for "in-hospital mortality".

```bash
python -m mimic3benchmark.scripts.create_in_hospital_mortality data/root/ data/in-hospital-mortality/
```

After the above commands are done, there will be a directory `data/{task}` for each created benchmark task.
These directories have two sub-directories: `train` and `test`.
Each of them contains bunch of ICU stays and one file with name `listfile.csv`, which lists all samples in that particular set.
Each row of `listfile.csv` has the following form: `icu_stay, period_length, label(s)`.
A row specifies a sample for which the input is the collection of ICU event of `icu_stay` that occurred in the first `period_length` hours of the stay and the target is/are `label(s)`.
In in-hospital mortality prediction task `period_length` is always 48 hours, so it is not listed in corresponding listfiles.
