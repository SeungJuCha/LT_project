## Project : Long tail classification using synthesized images

If you need generated images for cifar100 data, tell me. (2.4GB)


  All the data must be like this root


  LT_project(main_folder)

 
     |---datasets
            |
            |
            |---cifar-100-ptyon (origin data)
            |
            |---cifar-100-lt (LT data)
            | 
            |---syn_cifar100 (generated data)

If you want to generate images by your own, Data_synthesizing.py
Set your outputpath before generating
```bash
python Data_syntesizing.py
```

Make LT datset into specific dsitribution use save_data.py



To train with cifar_train.py 
``` bash
python cifar_train.py --dataset cifar100 --longtail True  --loss_type CE --gpu 0 --batch-size 128 --imb_factor 0.01 -d 100
```


### Reference

```
@inproceedings{cao2019learning,
  title={Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss},
  author={Cao, Kaidi and Wei, Colin and Gaidon, Adrien and Arechiga, Nikos and Ma, Tengyu},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
```
