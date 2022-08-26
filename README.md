# MM-Pyramid  

**[ACM MM 2022] MM-Pyramid: Multimodal Pyramid Attentional Network for Audio-Visual Event Localization and Video Parsing**

Jiashuo Yu, Ying Cheng, Rui-Wei Zhao, Rui Feng, Yuejie Zhang  

[Paper](https://arxiv.org/abs/2111.12374)  

## Requirements  

    python==3.6.9  
    torch==1.8.1  
    torchvision==0.9.0
    cuda==11.1  
    numpy==1.19.5  
    
## Data  

Please refer to [LLP](https://github.com/YapengTian/AVVP-ECCV20) and [AVE](https://github.com/YapengTian/AVE-ECCV18) for the required datasets.  

## Training

`python main_avvp.py --mode=train`  

## Testing

`python main_avvp.py  --mode=test`  

## Citation  

If you find our work interesting and useful, please consider citing it.  

    @article{yu2022mmp,
      title={MM-Pyramid: Multimodal Pyramid Attentional Network for Audio-Visual Event Localization and Video Parsing},
      author={Jiashuo Yu, Ying Cheng, Rui-Wei Zhao, Rui Feng, Yuejie Zhang},
      journal={arXiv preprint arXiv:2111.12374},
      year={2022}
    }  

## License

This project is released under the MIT License.
