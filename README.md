# ECE695 Final Project

## Project Overview:

### Motivation:

In the medical field, data can sometimes be very limited. This raises a couple of different problems with the models you use when working with these datasets. Some of those problems include models being easily overtrained and not being able to generalize well. There are many approaches that are used to handle these problems. Some of them include transfer learning, generating data, bayesian models, etc. 

### Project Implementation:

In this project, we will be attempting to solve these issues by generating new data using the GANs and incorporating Monte Carlo dropout to add a bayesian component into our model. The model we will be utilizing is the transformer model from the paper "Attention is all you Need".

#### Modified DCGAN:

For the first part of this project I applied some of the concepts from DCGAN on ECoG data to try to generate more data. Then I utilized an svm to map the generated data onto which category the data belong to. Finally, we applied added this data into our dataset and observed the impact.

### Monte Carlo Dropout on Transformer model for Machine Translation Task:

The idea for this implementation came from the research paper called "To BAN or not to BAN" where the researchers applied Monte Carlo Dropout on BERT models in order to improve the performance of 



## Project Layout:



## How To Run:



## Presentation Slides:

https://docs.google.com/presentation/d/1PV-z4uMzpy1nJBi7GXPSeMeMgGYGsn9o1UsiJFFzqDM/edit?usp=sharing

## Refences:

@misc{radford2016unsupervised,
      title={Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks}, 
      author={Alec Radford and Luke Metz and Soumith Chintala},
      year={2016},
      eprint={1511.06434},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

@misc{miok2020ban,
      title={To BAN or not to BAN: Bayesian Attention Networks for Reliable Hate Speech Detection}, 
      author={Kristian Miok and Blaz Skrlj and Daniela Zaharie and Marko Robnik-Sikonja},
      year={2020},
      eprint={2007.05304},
      archivePrefix={arXiv},
      primaryClass={stat.AP}
}

@misc{vaswani2017attention,
      title={Attention Is All You Need}, 
      author={Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
      year={2017},
      eprint={1706.03762},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
