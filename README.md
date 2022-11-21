
## DEEP LEARNING MINI-PROJECT:

### Team Members:
1) Deepti Preta Gouthaman
2) Nobel Dang
3) Shivansh Sharma

# **ResNet-Lite (<5 Million Parameters)**

A project aimed at improving ResNet accuracy on CIFAR10 dataset while keeping the model parameters under 5 million. The project is fully modular and device agnostic in design. New transforms can be applied from `transformers.py` and new model configurations can be loaded from `models.py` dynamically. There are multiple optional arguments to help you debug and get sense of the model.

> Best Accuracy: **0.00%**

> Parameters: **1234**

---
## **Project Stucture**

|Directory / File     | Description                                 |
|---------------------|---------------------------------------------|
|`checkpoints`        | Contains the saved network states           |
|`docs`               | Project documentaion                        |
|`logs`               | Test run logs                               |
|`plots`              | Saved plots                                 |
|`models.py`          | Resnet definitions and model configuartion  |
|`resnet.py`          | main python file                            |
|`tools.py`           | Debug functions                             |
|`transformers.py`    | Data Transform definitions                  |

---
## **Run options**

To get help execute -
```bash
python resnet.py -h
```

**Optional Arguments:**
| Arguments                | Description                  | Default
|--------------------------|------------------------------|-----------------
| --help | Show this help message and exit
| --mname | Model Name for logging purpose | None (Required)
| --lr | Learning Rate | 0.01
| --resume | Resume Training | False
| --epochs | No. of training epochs | 50
| --optimz | Optimizer: 'sgd', 'adam', 'adadelta' | 'sgd'
| --model | Model: 'ResNet10', 'ResNet14', 'ResNet14_v2' | 'ResNet14'
| --wd | Weight decay for l2 regularization | 5e-4
| --do_annealing | Enable Cosine Annealing | False
| --overwrite | Overwrite the existing model (mname) | False



**Usage**
python main.py --mname |str| [--help] [--lr <float>] [--resume] [--epochs <int>] [--optimz <str>] [--model <str>] [--wd <float>] [--do_annealing] [--overwrite]

---
## **How to run**

1. To start training with best model configuration for 100 epochs execute the following command -
    ```bash
    python resnet.py -e 100 -o adadelta -an -sc -mx -v -m project1_model
    ```

2. To resume training  for 100 epochs from best state checkpoint -
    ```bash
    python resnet.py -e 100 -o adadelta -an -sc -mx -v -m project1_model -r AA4Test
    ```
3. To save logs to a file in logs directory -
    ```bash
    python resnet.py -e 100 -m project1_model -r AA4Test >> logs/<filename>.log
    ```
    > Replace `<filename>` with your choice of filename

</br>

---
## **Documentation**

1. Project report can be found at [docs/project_report.pdf](https://github.com/95anantsingh/NYU-ResNet-On-Steroids/tree/main/docs/project_report.pdf)
2. Logs maintained at this [Google Sheet](https://docs.google.com/spreadsheets/d/1nRBr6NUiwAlOIIo7suecOdHwUBimqH-jmur7WVYfs0w/edit?usp=sharing)


