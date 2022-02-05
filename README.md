# RIAI 2019 Course Project

This project was conducted by Vincent Bardenhagen and Jim Buffat as part of the course Reliable Artificial Intelligence at ETH in the autumn term of 2019. The goal of the project was to implement convex relaxiations for robustness verification. The aim of sound and fast verification was achieved by our efficient implementation of the DeepZ approach. 


## Folder structure
In the directory `code` there are 3 files. 
File `networks.py` and `networks_test.py` contain encodings of fully connected and convolutional neural network architectures as PyTorch classes.
The architectures extend `nn.Module` object and consist of standard PyTorch layers (`Linear`, `Flatten`, `ReLU`, `Conv2d`). The first layer of each network performs normalization of the input image.

The file `verifier.py` contains the verifier algorithm. The stored networks and test cases are loaded in the `main` function. In the `analyze` function the DeepZ convex relaxation is implemented in pytorch. The implementation focuses on efficiency with some implementation tricks outlined in the `Report.pdf`. 

In folder `mnist_nets` there are 10 neural networks (5 fully connected and 5 convolutional). These networks are loaded using PyTorch in `verifier.py`.
In folder `test_cases` there are 10 subfolders. Each subfolder is associated with one of the networks, using the same name. In a subfolder corresponding to a network,there are 2 test cases for this network. 

In the folder `attack.py` the PGD attack is implemented to produce additional test cases to check the tightness as well as the soundness of the implemented approach. 

In the folder `tests.py` we analyze the behaviour of the individual transformations that we implemented to ensure proper functionality. 

## Setup instructions

The virtual environment used for the set up cann be installed with the following commands

```bash
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

## Running the verifier

The verifier should be run from the `code` directory using the command:

```bash
$ python verifier.py --net {net} --spec ../test_cases/{net}/img{test_idx}_{eps}.txt
```

In this command, `{net}` is equal to one of the following values (each representing one of the networks we want to verify): `fc1, fc2, fc3, fc4, fc5, conv1, conv2, conv3, conv4, conv5`.
`test_idx` is an integer representing index of the test case, while `eps` is perturbation that verifier should certify in this test case.

To evaluate the verifier on all networks and sample test cases, use the evaluation script.
You can run this script using the following commands:

```bash
chmod +x evaluate
./evaluate
```
