---
license: apache-2.0
datasets:
- openbmb/RLAIF-V-Dataset
language:
- en
---

# Model Card for RLAIF-V

[GitHub ](https://github.com/RLHF-V/RLAIF-V) | [Paper](https://arxiv.org/abs/2405.17220)

**RLAIF-V-7B** is trained based on LLaVA 1.5 7B with the novel [RLAIF-V](https://github.com/RLHF-V/RLAIF-V) framework.
By aligning with human preference via large scale [AI feedback](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset), the model achieves **super GPT-4V trustworthiness**. 
RLAIF-V maximally exploits the open-source feedback from two key perspectives, including high-quality feedback data and an online feedback learning algorithm. 


## Model Details

### Key Features

* 📈 **Most trustworthy LLaVA 1.5**: By learning from open-source AI feedback, specifically, the feedback from LLaVA-NeXT-34B, RLAIF-V-7B achieves the best trustworthiness improvement on LLaVA-v1.5 compared to other hallucination reduction methods.
* 💪 **Maintaining Well Performance on General Abilities**: On benchmarks evaluating general capabilities (e.g. LLaVA Bench, MMStar), RLAIF-V-7B also exhibits good performance.


<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/6566e0c493e30c8a60048eb3/ypXZxb4HE-jDPJU9115bi.png" alt="fig1" width="90%"/>
</p>

### Examples
<p align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/6566e0c493e30c8a60048eb3/Hyu2Et5CQtDFmxaYHKdu-.png" alt="fig2-1" width="80%"/>
  <img src="https://cdn-uploads.huggingface.co/production/uploads/6566e0c493e30c8a60048eb3/16mJpyH_-vnRfl8Ywfa6k.png" alt="fig2-1" width="80%"/>
</p>

### Model Description
- **Trained from model:** [llava-v1.5-7B](https://huggingface.co/liuhaotian/llava-v1.5-7b)
- **Trained on data:** [RLAIF-V-Dataset](https://huggingface.co/datasets/HaoyeZhang/RLAIF-V-Dataset)

## Usage
Please look at [GitHub](https://github.com/RLHF-V/RLAIF-V) for more details about usage.


## Citation

If you find our model/code/paper helpful, please consider cite our papers 📝:

```bibtex
@article{yu2023rlhf,
  title={Rlhf-v: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback},
  author={Yu, Tianyu and Yao, Yuan and Zhang, Haoye and He, Taiwen and Han, Yifeng and Cui, Ganqu and Hu, Jinyi and Liu, Zhiyuan and Zheng, Hai-Tao and Sun, Maosong and others},
  journal={arXiv preprint arXiv:2312.00849},
  year={2023}
}

@article{yu2024rlaifv,
  title={RLAIF-V: Aligning MLLMs through Open-Source AI Feedback for Super GPT-4V Trustworthiness}, 
  author={Yu, Tianyu and Zhang, Haoye and Yao, Yuan and Dang, Yunkai and Chen, Da and Lu, Xiaoman and Cui, Ganqu and He, Taiwen and Liu, Zhiyuan and Chua, Tat-Seng and Sun, Maosong},
  journal={arXiv preprint arXiv:2405.17220},
  year={2024},
}
```