# GPFedRec: Graph-Guided Personalization for Federated Recommendation
Code for kdd-24 paper: [GPFedRec: Graph-Guided Personalization for Federated Recommendation](https://arxiv.org/pdf/2305.07866).

## Abatract
The federated recommendation system is an emerging AI service architecture that provides recommendation services in a privacy-preserving manner. Using user-relation graphs to enhance federated recommendations is a promising topic. However, it is still an open challenge to construct the user-relation graph while preserving data locality-based privacy protection in federated settings. Inspired by a simple motivation, similar users share a similar vision (embeddings) to the same item set, this paper proposes a novel Graph-guided Personalization for Federated Recommendation (GPFedRec). The proposed method constructs a user-relation graph from user-specific personalized item embeddings at the server without accessing the users’ interaction records. The personalized item embedding is locally fine-tuned on each device, and then a user-relation graph will be constructed by measuring the similarity among client-specific item embeddings. Without accessing users’ historical interactions, we embody the data locality-based privacy protection of vanilla federated learning. Furthermore, a graph-guided aggregation mechanism is designed to leverage the user-relation graph and federated optimization framework simultaneously. Extensive experiments on five benchmark datasets demonstrate GPFedRec’s superior performance. The in-depth study validates that GPFedRec can generally improve existing federated recommendation methods as a plugin while keeping user privacy safe. 

![](https://github.com/Zhangcx19/GPFedRec/blob/main/GPFedRec_framework.png)
**Figure:**
The proposed GPFedRec framework.

## Preparations before running the code
mkdir log

mkdir sh_result

## Running the code
python train.py

## Citation
If you find this project helpful, please consider to cite the following paper:

```
@article{zhang2023graph,
  title={Graph-guided Personalization for Federated Recommendation},
  author={Zhang, Chunxu and Long, Guodong and Zhou, Tianyi and Yan, Peng and Zhang, Zijjian and Yang, Bo},
  journal={arXiv preprint arXiv:2305.07866},
  year={2023}
}
```
