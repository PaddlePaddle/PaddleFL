# PaddleFL

PaddleFL is an open source federated learning framework based on PaddlePaddle. Researchers can easily replicate and compare different federated learning algorithms with PaddleFL. Developers can also benefit from PaddleFL in that it is easy to deploy a federated learning system in large scale distributed clusters. In PaddleFL, several federated learning strategies will be provided with application in computer vision, natural language processing, recommendation and so on. Application of traditional machine learning training strategies such as Multi-task learning, Transfer Learning in Federated Learning settings will be provided. Based on PaddlePaddle's large scale distributed training and elastic scheduling of training job on Kubernetes, PaddleFL can be easily deployed based on full-stack open sourced software.


## Overview of PaddleFL

Data is becoming more and more expensive nowadays, and sharing of raw data is very hard across organizations. Federated Learning aims to solve the problem of data isolation and secure sharing of data knowledge among organizations. The concept of federated learning is proposed by researchers in Google [1, 2, 3]. PaddleFL implements federated learning based on the PaddlePaddle framework. Application demonstrations in natural language processing, computer vision and recommendation will be provided in PaddleFL. PaddleFL supports the current two main federated learning strategies[4]: vertical federated learning and horizontal federated learning. Multi-tasking learning [7] and transfer learning [8] in federated learning will be developed and supported in PaddleFL in the future.

- **Horizontal Federated Learning**: Federated Averaging [2], Differential Privacy [6], Secure Aggregation[11]
- **Vertical Federated Learning**: Two-party training with PrivC[5], Three-party training with ABY3 [10]

<img src='images/FL-framework.png' width = "1000" height = "320" align="middle"/>

## Framework Design of PaddleFL

There are mainly two components in PaddleFL: **Data Parallel** and **Federated Learning with MPC (PFM)**.

- With Data Parallel, distributed data holders can finish their Federated Learning tasks based on common horizontal federated strategies, such as FedAvg, DPSGD, etc.

- PFM is implemented based on secure multi-party computation (MPC) to enable secure training and prediction. As a key product of PaddleFL, PFM intrinsically supports federated learning well, including horizontal, vertical and transfer learning scenarios. Users with little cryptography expertise can also train models or conduct prediction on encrypted data.

### Data Parallel

In Data Parallel, the whole process of model training is divided into two stages: Compile Time and Run Time. Components for defining a federated learning task and training a federated learning job are as follows:

<img src='images/FL-training.png' width = "1000" height = "400" align="middle"/>

#### A. Compile Time

- **FL-Strategy**: a user can define federated learning strategies with FL-Strategy such as Fed-Avg[2]

- **User-Defined-Program**: PaddlePaddle's program that defines the machine learning model structure and training strategies such as multi-task learning.

- **Distributed-Config**: In federated learning, a system should be deployed in distributed settings. Distributed Training Config defines distributed training node information.

- **FL-Job-Generator**: Given FL-Strategy, User-Defined Program and Distributed Training Config, FL-Job for federated server and worker will be generated through FL Job Generator. FL-Jobs will be sent to organizations and federated parameter server for run-time execution.

#### B. Run Time

- **FL-Server**: federated parameter server that usually runs in cloud or third-party clusters.

- **FL-Worker**: Each organization participates in federated learning will have one or more federated workers that will communicate with the federated parameter server.

- **FL-scheduler**: Decide which set of trainers can join the training before each updating cycle.

For more instructions, please refer to the [examples](./python/paddle_fl/paddle_fl/examples)

### Federated Learning with MPC

PaddleFL MPC implements secure training and inference tasks based on the underlying MPC protocol like ABY3[10] and PrivC[5], which are high efficient multi-party computing model. In PaddeFL, two-party federated learning based on PrivC mainly supports linear/logistic regression and DNN model. Three-party federated learning based on ABY3 supports linear/logistic regression, DNN model, CNN model and FM.

In PaddleFL MPC, participants can be classified into roles of Input Party (IP), Computing Party (CP) and Result Party (RP). Input Parties (e.g., the training data/model owners) encrypt and distribute data or models to Computing Parties (There exist three computing parties in ABY3 protocol while two computing parties in PrivC protocol). Computing Parties (e.g., the VM on the cloud) conduct training or inference tasks based on specific MPC protocols, being restricted to see only the encrypted data or models, and thus guarantee the data privacy. When the computation is completed, one or more Result Parties (e.g., data owners or specified third-party) receive the encrypted results from Computing Parties, and reconstruct the plaintext results. Roles can be overlapped, e.g., a data owner can also act as a computing party.

<img src='images/PFM-overview.png' width = "1000" height = "446" align="middle"/>


A full training or inference process in PFM consists of mainly three phases: data preparation, training/inference, and result reconstruction.

#### A. Data Preparation

- **Private data alignment**: PFM enables data owners (IPs) to find out records with identical keys (like UUID) without revealing private data to each other. This is especially useful in the vertical learning cases where segmented features with same keys need to be identified and aligned from all owners in a private manner before training.

- **Encryption and distribution**: In PFM, data and models from IPs will be encrypted using Secret-Sharing[9], and then be sent to CPs, via directly transmission or distributed storage like HDFS. Each CP can only obtain one share of each piece of data, and thus is unable to recover the original value in the Semi-honest model.

#### B. Training/Inference

A PFM program is exactly a PaddlePaddle program, and will be executed as normal PaddlePaddle programs. Before training/inference, user needs to choose a MPC protocol, define a machine learning model and their training strategies. Typical machine learning operators are provided in `paddle_fl.mpc` over encrypted data, of which the instances are created and run in order by Executor during run-time.

For more information of Training/inference phase, please refer to the following [doc](./docs/source/md/mpc_train.md).

#### C. Result Reconstruction

Upon completion of the secure training (or inference) job, the models (or prediction results) will be output by CPs in encrypted form. Result Parties can collect the encrypted results, decrypt them using the tools in PFM, and deliver the plaintext results to users (Currently, data sharing and reconstruction can be supported in both offline and online modes).

For more instructions, please refer to [mpc examples](./python/paddle_fl/mpc/examples)

## On Going and Future Work

- Vertical Federated Learning will support more algorithms.

- Add K8S deployment scheme for PFM.

- FL mobile simulator will be open sourced in following versions.
PaddleFL

PaddleFL is an open source federated learning framework based on PaddlePaddle. Researchers can easily replicate and compare different federated learning algorithms with PaddleFL. Developers can also benefit from PaddleFL in that it is easy to deploy a federated learning system in large scale distributed clusters. In PaddleFL, several federated learning strategies will be provided with application in computer vision, natural language processing, recommendation and so on. Application of traditional machine learning training strategies such as Multi-task learning, Transfer Learning in Federated Learning settings will be provided. Based on PaddlePaddle's large scale distributed training and elastic scheduling of training job on Kubernetes, PaddleFL can be easily deployed based on full-stack open sourced software.


## Overview of PaddleFL

Data is becoming more and more expensive nowadays, and sharing of raw data is very hard across organizations. Federated Learning aims to solve the problem of data isolation and secure sharing of data knowledge among organizations. The concept of federated learning is proposed by researchers in Google [1, 2, 3]. PaddleFL implements federated learning based on the PaddlePaddle framework. Application demonstrations in natural language processing, computer vision and recommendation will be provided in PaddleFL. PaddleFL supports the current two main federated learning strategies[4]: vertical federated learning and horizontal federated learning. Multi-tasking learning [7] and transfer learning [8] in federated learning will be developed and supported in PaddleFL in the future.

- **Horizontal Federated Learning**: Federated Averaging [2], Differential Privacy [6], Secure Aggregation[11]
- **Vertical Federated Learning**: Two-party training with PrivC[5], Three-party training with ABY3 [10]

<img src='images/FL-framework.png' width = "1000" height = "320" align="middle"/>

## Framework Design of PaddleFL

There are mainly two components in PaddleFL: **Data Parallel** and **Federated Learning with MPC (PFM)**.

- With Data Parallel, distributed data holders can finish their Federated Learning tasks based on common horizontal federated strategies, such as FedAvg, DPSGD, etc.

- PFM is implemented based on secure multi-party computation (MPC) to enable secure training and prediction. As a key product of PaddleFL, PFM intrinsically supports federated learning well, including horizontal, vertical and transfer learning scenarios. Users with little cryptography expertise can also train models or conduct prediction on encrypted data.

### Data Parallel

In Data Parallel, the whole process of model training is divided into two stages: Compile Time and Run Time. Components for defining a federated learning task and training a federated learning job are as follows:

<img src='images/FL-training.png' width = "1000" height = "400" align="middle"/>

#### A. Compile Time

- **FL-Strategy**: a user can define federated learning strategies with FL-Strategy such as Fed-Avg[2]

- **User-Defined-Program**: PaddlePaddle's program that defines the machine learning model structure and training strategies such as multi-task learning.

- **Distributed-Config**: In federated learning, a system should be deployed in distributed settings. Distributed Training Config defines distributed training node information.

- **FL-Job-Generator**: Given FL-Strategy, User-Defined Program and Distributed Training Config, FL-Job for federated server and worker will be generated through FL Job Generator. FL-Jobs will be sent to organizations and federated parameter server for run-time execution.

#### B. Run Time

- **FL-Server**: federated parameter server that usually runs in cloud or third-party clusters.

- **FL-Worker**: Each organization participates in federated learning will have one or more federated workers that will communicate with the federated parameter server.

- **FL-scheduler**: Decide which set of trainers can join the training before each updating cycle.

For more instructions, please refer to the [examples](./python/paddle_fl/paddle_fl/examples)

### Federated Learning with MPC

PaddleFL MPC implements secure training and inference tasks based on the underlying MPC protocol like ABY3[10] and PrivC[5], which are high efficient multi-party computing model. In PaddeFL, two-party federated learning based on PrivC mainly supports linear/logistic regression and DNN model. Three-party federated learning based on ABY3 supports linear/logistic regression, DNN model, CNN model and FM.

In PaddleFL MPC, participants can be classified into roles of Input Party (IP), Computing Party (CP) and Result Party (RP). Input Parties (e.g., the training data/model owners) encrypt and distribute data or models to Computing Parties (There exist three computing parties in ABY3 protocol while two computing parties in PrivC protocol). Computing Parties (e.g., the VM on the cloud) conduct training or inference tasks based on specific MPC protocols, being restricted to see only the encrypted data or models, and thus guarantee the data privacy. When the computation is completed, one or more Result Parties (e.g., data owners or specified third-party) receive the encrypted results from Computing Parties, and reconstruct the plaintext results. Roles can be overlapped, e.g., a data owner can also act as a computing party.

<img src='images/PFM-overview.png' width = "1000" height = "446" align="middle"/>


A full training or inference process in PFM consists of mainly three phases: data preparation, training/inference, and result reconstruction.

#### A. Data Preparation

- **Private data alignment**: PFM enables data owners (IPs) to find out records with identical keys (like UUID) without revealing private data to each other. This is especially useful in the vertical learning cases where segmented features with same keys need to be identified and aligned from all owners in a private manner before training.

- **Encryption and distribution**: In PFM, data and models from IPs will be encrypted using Secret-Sharing[9], and then be sent to CPs, via directly transmission or distributed storage like HDFS. Each CP can only obtain one share of each piece of data, and thus is unable to recover the original value in the Semi-honest model.

#### B. Training/Inference

A PFM program is exactly a PaddlePaddle program, and will be executed as normal PaddlePaddle programs. Before training/inference, user needs to choose a MPC protocol, define a machine learning model and their training strategies. Typical machine learning operators are provided in `paddle_fl.mpc` over encrypted data, of which the instances are created and run in order by Executor during run-time.

For more information of Training/inference phase, please refer to the following [doc](./docs/source/md/mpc_train.md).

#### C. Result Reconstruction

Upon completion of the secure training (or inference) job, the models (or prediction results) will be output by CPs in encrypted form. Result Parties can collect the encrypted results, decrypt them using the tools in PFM, and deliver the plaintext results to users (Currently, data sharing and reconstruction can be supported in both offline and online modes).

For more instructions, please refer to [mpc examples](./python/paddle_fl/mpc/examples)

## On Going and Future Work

- Vertical Federated Learning will support more algorithms.

- Add K8S deployment scheme for PFM.

- FL mobile simulator will be open sourced in following versions.
 PaddleFL

PaddleFL is an open source federated learning framework based on PaddlePaddle. Researchers can easily replicate and compare different federated learning algorithms with PaddleFL. Developers can also benefit from PaddleFL in that it is easy to deploy a federated learning system in large scale distributed clusters. In PaddleFL, serveral federated learning strategies will be provided with application in computer vision, natural language processing, recommendation and so on. Application of traditional machine learning training strategies such as Multi-task learning, Transfer Learning in Federated Learning settings will be provided. Based on PaddlePaddle's large scale distributed training and elastic scheduling of training job on Kubernetes, PaddleFL can be easily deployed based on full-stack open sourced software.

# Federated Learning

Data is becoming more and more expensive nowadays, and sharing of raw data is very hard across organizations. Federated Learning aims to solve the problem of data isolation and secure sharing of data knowledge among organizations. The concept of federated learning is proposed by researchers in Google [1, 2, 3].

## Overview of PaddleFL

<img src='_static/FL-framework.png' width = "1000" height = "320" align="middle"/>


In PaddleFL, horizontal and vertical federated learning strategies will be implemented according to the categorization given in [4]. Application demonstrations in natural language processing, computer vision and recommendation will be provided in PaddleFL.

#### A. Federated Learning Strategy

- **Vertical Federated Learning**: Logistic Regression with PrivC[5], Neural Network with MPC [11]

- **Horizontal Federated Learning**: Federated Averaging [2], Differential Privacy [6], Secure Aggregation

#### B. Training Strategy

- **Multi Task Learning** [7]

- **Transfer Learning** [8]

- **Active Learning**

There are mainly two components in PaddleFL: **Data Parallel** and **Federated Learning with MPC (PFM)**.

With Data Parallel, distributed data holders can finish their Federated Learning tasks based on common horizontal federated strategies, such as FedAvg, DPSGD, etc.

Besides, PFM is implemented based on secure multi-party computation (MPC) to enable secure training and prediction. As a key product of PaddleFL, PFM intrinsically supports federated learning well, including horizontal, vertical and transfer learning scenarios. Users with little cryptography expertise can also train models or conduct prediction on encrypted data.

## Framework design of PaddleFL

### Data Parallel

<img src='_static/FL-training.png' width = "1000" height = "400" align="middle"/>

In Data Parallel, components for defining a federated learning task and training a federated learning job are as follows:

#### A. Compile Time

- **FL-Strategy**: a user can define federated learning strategies with FL-Strategy such as Fed-Avg[2]

- **User-Defined-Program**: PaddlePaddle's program that defines the machine learning model structure and training strategies such as multi-task learning.

- **Distributed-Config**: In federated learning, a system should be deployed in distributed settings. Distributed Training Config defines distributed training node information.

- **FL-Job-Generator**: Given FL-Strategy, User-Defined Program and Distributed Training Config, FL-Job for federated server and worker will be generated through FL Job Generator. FL-Jobs will be sent to organizations and federated parameter server for run-time execution.

#### B. Run Time

- **FL-Server**: federated parameter server that usually runs in cloud or third-party clusters.

- **FL-Worker**: Each organization participates in federated learning will have one or more federated workers that will communicate with the federated parameter server.

- **FL-scheduler**: Decide which set of trainers can join the training before each updating cycle.

### Federated Learning with MPC

<img src='_static/PFM-overview.png' width = "1000" height = "446" align="middle"/>

Paddle FL MPC implements secure training and inference tasks based on the underlying MPC protocol like ABY3[11], which is a high efficient three-party computing model.

In ABY3, participants can be classified into roles of Input Party (IP), Computing Party (CP) and Result Party (RP). Input Parties (e.g., the training data/model owners) encrypt and distribute data or models to Computing Parties. Computing Parties (e.g., the VM on the cloud) conduct training or inference tasks based on specific MPC protocols, being restricted to see only the encrypted data or models, and thus guarantee the data privacy. When the computation is completed, one or more Result Parties (e.g., data owners or specified third-party) receive the encrypted results from Computing Parties, and reconstruct the plaintext results. Roles can be overlapped, e.g., a data owner can also act as a computing party.

A full training or inference process in PFM consists of mainly three phases: data preparation, training/inference, and result reconstruction.

#### A. Data preparation

- **Private data alignment**: PFM enables data owners (IPs) to find out records with identical keys (like UUID) without revealing private data to each other. This is especially useful in the vertical learning cases where segmented features with same keys need to be identified and aligned from all owners in a private manner before training.

- **Encryption and distribution**: In PFM, data and models from IPs will be encrypted using Secret-Sharing[10], and then be sent to CPs, via directly transmission or distributed storage like HDFS. Each CP can only obtain one share of each piece of data, and thus is unable to recover the original value in the Semi-honest model.

#### B. Training/inference

A PFM program is exactly a PaddlePaddle program, and will be executed as normal PaddlePaddle programs. Before training/inference, user needs to choose a MPC protocol, define a machine learning model and their training strategies. Typical machine learning operators are provided in `paddle_fl.mpc` over encrypted data, of which the instances are created and run in order by Executor during run-time.


#### C. Result reconstruction

Upon completion of the secure training (or inference) job, the models (or prediction results) will be output by CPs in encrypted form. Result Parties can collect the encrypted results, decrypt them using the tools in PFM, and deliver the plaintext results to users.

# On Going and Future Work

- Vertial Federated Learning will support more algorithms.

- Add K8S deployment scheme for Paddle Encrypted.

- FL mobile simulator will be open sourced in following versions.


