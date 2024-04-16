This is the replication package for our paper titled *"CCAF: Learning Code Change via AdapterFusion"* including both code and datasets. 


## Requirements

You should al least install following packages to run our code:
- pytorch: 1.12.0+cu113
- transformers: 4.18.0
- adapters: 0.1.1
- ...

The full list of dependencies is listed in `requirements.txt`.

## Dataset
You can download the data for CMG and JDP tasks from this link: [ccaf_dataset.zip](https://drive.google.com/file/d/1StocfPcj4PYanHhv416HJp0R61J5KtMB/view?usp=drive_link).

## Run CCAF
For APCA, CMG and JDP tasks, we provide three shell scripts for running our methods and we use the CMG task as an example.

Execute following command at the root of the project to run CodeT5:
```shell
cd cmg/code
bash run.sh
```

Execute following command at the root of the project to run CodeT5<sub>adapter</sub>:
```shell
cd cmg/code
bash run_adapter.sh
```

Execute following command at the root of the project to run CCAF:
```shell
cd cmg/code
bash run_adapterfusion.sh
```

More hyperparameter settings can be found in each script.