# WebBench

This repo contains the evaluation framework for the paper: [WebBench: How Far Have Multimodal LLMs Evolved in Web Page Understanding and Grounding?](https://arxiv.org)

[**🌐 Homepage**](https://webbench-mllm.github.io/) | [**🤗 Dataset**](https://huggingface.co/datasets/webbench/WebBench) | [**📖 arXiv**](https://arxiv.org)


## Introduction

We introduce **WebBench**, a multimodal benchmark designed to assess the **understanding and grounding capabilities of MLLMs in web scenarios**. WebBench consists of **seven tasks**, and comprises **1.5K** human-curated instances from **139** real websites, covering 87 sub-domains. We evaluate 14 open-source MLLMs, Gemini Pro, Claude 3, and GPT-4V(ision) on WebBench, revealing significant challenges and performance gaps. Further analysis highlights the limitations of current MLLMs, including inadequate grounding in text-rich environments and subpar performance with low-resolution image inputs. We believe WebBench will serve as a valuable resource for the research community and contribute to the creation of more powerful and versatile MLLMs for web-related applications.

![Alt text](assets/main.png)


## Benchmark Construction
We introduce WebBench, a comprehensive multimodal benchmark designed to assess the capabilities of MLLMs in the web domain. Inspired by the human interaction process with web browsers, WebBench consists of seven tasks that map to core abilities required for web tasks: captioning, webpage QA, heading OCR, element OCR, element grounding, action prediction, and action grounding, as detailed in the figure. The benchmark comprises 1.5K instances, all uniformly formulated in the QA style, making it easy to evaluate and compare the performance of different MLLMs.
![Alt text](assets/compare.png)
The proposed WebBench possesses the following features:
- **Comprehensiveness**: WebBench spans 139 websites with 1.5K samples, encompassing 12 different domains (e.g., travel, sports, hobby, lifestyle, animals, science, etc.) and 87 sub-domains.
- **Multi-granularity**: WebBench assesses MLLMs at three levels: website-level, element-level, and action-level.
- **Multi-tasks**: WebBench encompasses seven tasks designed to evaluate the understanding, OCR, grounding, and reasoning capabilities of MLLMs.
- **High quality**: Quality is ensured through careful human verification and curation efforts.
![Alt text](assets/detail.png)

## Evaluation

We provide evaluation code for GPT-4V, Claude, Gemini, and LLaVA 1.6 series.
See `run.sh` for more details.

The experimental results are as follows:
![Alt text](assets/exp.png)

## How to Add a Model
1. Implement a model adapter in `model_adapters`. See `model_adapters/llava_adapter.py` for an example.
2. Modify `run.py` to add your model.
3. Write a config file in `configs`.

## Contact
- Junpeng Liu: [jpliu@link.cuhk.edu.hk](jpliu@link.cuhk.edu.hk)
- Yifan Song: [yfsong@pku.edu.cn](yfsong@pku.edu.cn)
- Xiang Yue: [xyue2@andrew.cmu.edu](xyue2@andrew.cmu.edu)

## Citation
If you find this work helpful, please cite out paper:
```
@article{liu2024webbench,
    author={Junpeng Liu and Yifan Song and Bill Yuchen Lin and Wai Lam and Graham Neubig and Yuanzhi Li and Xiang Yue},
    title={WebBench: How Far Have Multimodal LLMs Evolved in Web Page Understanding and Grounding?},
    year={2024},
    eprint={2404.99999},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
