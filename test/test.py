import torch
import os
os.chdir('/home/tbjag/workspace/csharding/')
file_path = os.path.join(os.getcwd(), 'data', 'dlrm_datasets', 'dlrm.pt')
#file_path = 'data/dlrm_datasets/dlrm.pt'

indices, offsets, lengths = torch.load(file_path)
tensor1_first_5_values = indices[:5]
tensor2_first_5_values = offsets[:5]

print("First 5 values of tensor1:", tensor1_first_5_values)
print("First 5 values of tensor2:", tensor2_first_5_values) # so  we get all the vals here
print(type(indices), type(offsets), type(lengths))
print(indices.size(), offsets.size(), lengths.size())

#how do we send data to processes