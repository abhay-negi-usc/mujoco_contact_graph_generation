import pandas as pd 

from sklearn.preprocessing import MinMaxScaler 
from torch.utils.data import Dataset 

class Wrench2ContactTypeDataset(Dataset): 

    def __init__(self, data_path, filter_out_no_contact=True, seed=42):   
        
        # read data 
        self.data = pd.read_csv(data_path) 
        
        # filter data 
        if filter_out_no_contact: 
            self.data = self.data[self.data["contact_type"] != -1] 

        # self.wrench_column_headings = ['FX','FY','FZ','TX','TY','TZ'] 
        # self.data['fx'] = self.data['FX'] / self.data['FZ'] 
        # self.data['fy'] = self.data['FY'] / self.data['FZ'] 
        # self.data['tx'] = self.data['TX'] / self.data['FZ'] 
        # self.data['ty'] = self.data['TY'] / self.data['FZ'] 
        # self.data['tz'] = self.data['TZ'] / self.data['FZ'] 
        # self.normalized_wrench_column_headings = ['fx','fy','tx','ty','tz']  

        self.pose_column_headings = ['X','Y','Z','QX','QY','QZ','QW'] 
        self.data = self.data[self.pose_column_headings + ['contact_type']] 
        
        
        # shuffle data 
        self.data = self.data.sample(frac=1, random_state=seed).reset_index(drop=True) 

        # scale data 
        self.scaler = MinMaxScaler() 
        # self.data[self.wrench_column_headings] = self.scaler.fit_transform(self.data[self.wrench_column_headings]) 
        # self.data[self.normalized_wrench_column_headings] = self.scaler.fit_transform(self.data[self.normalized_wrench_column_headings]) 
        self.data[self.pose_column_headings] = self.scaler.fit_transform(self.data[self.pose_column_headings]) 
        

    def __len__(self): 
        return len(self.data) 
    
    def __getitem__(self, idx): 
        if idx >= len(self.data) or idx < 0:
            raise IndexError("Index out of bounds") 
        
        # input = self.data.loc[idx, self.wrench_column_headings].values 

        # input = self.data.loc[idx, self.wrench_column_headings+self.normalized_wrench_column_headings].values 

        input = self.data.loc[idx, self.pose_column_headings].values 

        output = int(self.data.loc[idx, "contact_type"]) 

        return input, output  
            