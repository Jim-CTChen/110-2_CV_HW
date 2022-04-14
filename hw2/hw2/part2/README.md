# CV HW2 part2
## Run code
### Training
``` 
> python3 main.py

// you can use --model to change model among {myLeNet,myResNet18,myResNet34,ResNet18,ResNet34}
// default model set to ResNet18
```

**This will generate best_model.pt at ./save_dir/ResNet18/best_model.pt**

**And it will generate learning curve plot at ./acc_log/ResNet18/*.png (including training/validation loss/accuracy)**

**It will also generate train/validation loss/accuratcy log file to ./acc_log/ for the usage of tool.py**

### Evaluation
```
> python3 eval.py
// you can use --model to change model among {myLeNet,myResNet18,myResNet34,ResNet18,ResNet34}
// default model set to ResNet18
```

**It will read model from ./save_dir/ResNet18/best_model.pt and do evaluation**

### Summary model
```
> python3 summary_model.py
// you can use --model to change model among {myLeNet,myResNet18,myResNet34,ResNet18,ResNet34}
// default model set to ResNet18
```

**It will summary the model and write model architecture to {model_name}.txt**

### Plot learning curve
```
> python3 plot.py
// you can use --model to change model among {myLeNet,myResNet18,myResNet34,ResNet18,ResNet34}
// default model set to ResNet18
```

**It will plot learning curve with log file in ./acc_log/{model_name}/\*log (including training/validation loss/accuracy)**
