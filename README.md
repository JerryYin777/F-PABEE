# F-PABEE
**This is our paper "F-PABEE: Flexible-patience-based Early Exit for Single-label and Multi-label text Classification Tasks“ ’s source code, paper is now included in EMNLP 2022 Findings**

You will see our article on the [EMNLP 2022 official website](https://2022.emnlp.org/) in a while,here's our [Paper](https://github.com/JerryYin777/F-PABEE/blob/master/F-PABEE%20Flexible-patience-based%20Early%20Exit%20for%20Single-label%20and.pdf).

## Contribution
* F-PABEE outperforms existing early exit models such as Brexit and PABEE on both single-label and multi-label tasks.

* F-PABEE performs well on different PLMs such as ALBERT.

* Ablation studies show that as a similarity measure, Jenson-Shannon divergence works best for our F-PABEE method.

## Benchmark

<div align=center><img src="https://user-images.githubusercontent.com/88324880/199749657-4e8cade1-7c7e-496d-b348-b0e3a87d047b.png" width="1000"></div>

![2](https://user-images.githubusercontent.com/88324880/199749661-407f2d6e-af5d-4631-bee9-abbd434c198b.jpg)

![3](https://user-images.githubusercontent.com/88324880/199755594-e67fed0b-964b-4de7-8200-1d586ef17f8e.jpg)
![4](https://user-images.githubusercontent.com/88324880/199755610-a1854b4d-a15c-4687-901f-aa6eac80fa7f.jpg)
![5](https://user-images.githubusercontent.com/88324880/199755616-88cde13a-cd0a-47f7-b8cf-879320e07190.jpg)
![6](https://user-images.githubusercontent.com/88324880/199755626-db0ef40a-ba92-491b-9d6e-c5aed73b2f39.jpg)
![7](https://user-images.githubusercontent.com/88324880/199755635-ae934bdc-4745-42d1-ab9a-00a5de683ee0.jpg)
![8](https://user-images.githubusercontent.com/88324880/199755642-ed11b9ad-d4c0-4e79-b88d-277ce8392670.jpg)

## Conclusion
We proposed F-PABEE, a novel and efficient method combining the PABEE with a softer cross-layer comparison strategy. F-PABEE is more flexible than PABEE since it can achieve different speed-performance tradeoffs by adjusting the similarity score threshold and the patience parameter. In addition, the similarity score is measured by divergence-based metrics like KL-divergence and JSD-divergence from the current and previous layers. Extensive experiments on SLC and MLC tasks demonstrate that: 
* Our F-PABEE performs better than the previous SOTA adaptive early exit methods for SLC tasks. 
* Our F-PABEE performs better than other early exit methods for MLC tasks. As far as we know, we are the first to investigate the early exiting method of MLC tasks.
* F-PABEE performs well on different PLMs such as BERT and ALBERT. 
* Ablation studies show that JSD-divergence’s similarity calculation method outperforms other similarity score calculation methods for our F-PABEE model.
