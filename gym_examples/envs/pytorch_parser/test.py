from pytorch_parser_function import generate_and_train

from torchvision import datasets
from torchvision.transforms import ToTensor



train_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
)

test_data = datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)


generate_and_train([('conv',1,5,3,1,28,0,0,[]),('pool',1,0,2,2,28,0,0,[]),('conv',1,10,3,1,26,0,0,[]),('pool',1,0,2,2,28,0,0,[]),('fc',0,0,0,0,0,100,0,[]),('fc',0,0,0,0,0,10,0,[])], train_data, test_data)