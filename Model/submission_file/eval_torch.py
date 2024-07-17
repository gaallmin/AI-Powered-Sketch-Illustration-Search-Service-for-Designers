import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import gc

from sklearn.metrics import classification_report
import pandas as pd
from tqdm import tqdm

from model import get_classifier


def freeze(model):
    for p in model.parameters():
        p.requires_grad = False

    return model.eval()


def run_eval(model, loader):
    total_preds = []
    total_labels = []
    
    with torch.no_grad():
        model.cuda()
        for img, label in tqdm(loader):
            img = img.cuda()
            label = label.cuda()
            
            pred = model(img)
            total_preds.append(pred)
            total_labels.append(label)
            
        total_preds = torch.cat(total_preds).argmax(dim=1).cpu().numpy()
        total_labels = torch.cat(total_labels).cpu().numpy()

        result = classification_report(total_labels, total_preds, output_dict=True)
        
    return pd.DataFrame(result)


def check_across_seeds(accs, f1s, result_df, num_classes=20):
    accs = torch.tensor(accs)
    f1s = torch.tensor(f1s)
    
    assert torch.all(torch.abs(accs[1:] - accs[:1]) < 1e-1) and torch.all(torch.abs(f1s[1:] - f1s[:1]) < 1e-1), "test results are not compatible \n{}\n{}".format(accs, f1s)

    print("*** CLASSWISE RESULT ***")
    cwise_result = result_df.loc[['f1-score', 'recall'], [str(i) for i in range(num_classes)]]
    cwise_result = cwise_result.rename(index={'f1-score' : 'f1', 'recall' : 'acc'})
    print(cwise_result)
    
    print("\n*** AVG RESULT ***")
    avg_result = pd.Series({'f1' : result_df.loc['f1-score', 'macro avg'], 'acc' : result_df['accuracy'].values[0]})
    print(avg_result)



if __name__ == "__main__":
    SEEDS = [0, 5, 10]
    
    ''' 
    You need to implement "get_classifier" function that returns your implemented model.
    "get_classifier" should return your model defined with your model configuration.
    Also, you need to save model parameters as below.
    EX)
    torch.save(model.state_dict(), 'model.pth')
    
    '''
    CLF = get_classifier(num_classes=20)
    CLF.load_state_dict(torch.load('./model.pth'))

    ''' 
    Fill in the root directory path into DATA_DIR.
    You must write the subset directory for the specific split (train or valid).
    Under the root directory, the child folders should be "L2_3", "L2_10", ... , "L2_52"
    
    EX) if you named your valid dataset folder as "~/valid"
    then the child directory should be "~/valid/L2_3", "~/valid/L2_10", ... , "~/valid/L2_52"
    
    so you have to write as
    DATA_DIR = "~/valid"
    '''
    
    DATA_DIR = "../../train_val_test_dataset/valid" 
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
    transform = transforms.Compose([
                transforms.Resize(256), 
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
            ])
    
    
    ### run evaluation over 3 random seeds ###
    ACC_LIST = []
    F1_LIST = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        data = datasets.ImageFolder(root=DATA_DIR, transform=transform)
        loader = DataLoader(data, batch_size=32, shuffle=True)
        
        gc.collect()
        torch.cuda.empty_cache()
        
        CLF = freeze(CLF)
        RESULT_DF = run_eval(CLF, loader)
        ACC_LIST.append(RESULT_DF['accuracy'].values[0])
        F1_LIST.append(RESULT_DF.loc['f1-score', 'macro avg'])
        
    check_across_seeds(ACC_LIST, F1_LIST, RESULT_DF)
        
    
    
