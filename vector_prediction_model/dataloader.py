import pandas as pd 

from sklearn.preprocessing import MinMaxScaler 
from torch.utils.data import Dataset 

class Pose2ContactStateDataset(Dataset): 

    def __init__(self, data_path, filter_out_no_contact=True, seed=42):   
        
        # read data 
        self.data = pd.read_csv(data_path) 
        
        # filter data 
        if filter_out_no_contact: 
            self.data = self.data[self.data["contact_type"] != -1] 

        # self.data['fx'] = self.data['FX'] / self.data['FZ'] 
        # self.data['fy'] = self.data['FY'] / self.data['FZ'] 
        # self.data['tx'] = self.data['TX'] / self.data['FZ'] 
        # self.data['ty'] = self.data['TY'] / self.data['FZ'] 
        # self.data['tz'] = self.data['TZ'] / self.data['FZ'] 

        self.pose_column_headings = ['X','Y','Z','QX','QY','QZ','QW'] 
        # self.wrench_column_headings = ['FX','FY','FZ','TX','TY','TZ'] 
        # self.normalized_wrench_column_headings = ['fx','fy','tx','ty','tz']  

        hole_classes = ['HF1','HF2','HE1','HE2','HE3','HE4','HE5','HE6','HV1','HV2','HV3','HV4'] 
        peg_classes = ['PF1','PF2','PF3','PF4','PE1','PE2','PE3','PE4'] 
        self.peg_hole_classes = [] 
        for hole_class in hole_classes: 
            for peg_class in peg_classes: 
                self.peg_hole_classes.append(f'{hole_class}-{peg_class}') 

        self.data = self.data[self.pose_column_headings + self.peg_hole_classes] 
        
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

        output = self.data.loc[idx, self.peg_hole_classes].values 

        return input, output 

class Wrench2ContactStateDataset(Dataset): 

    def __init__(self, data_path, filter_out_no_contact=True, seed=42):   
        
        # read data 
        self.data = pd.read_csv(data_path) 
        
        # filter data 
        if filter_out_no_contact: 
            self.data = self.data[self.data["contact_type"] != -1] 

        self.data['fx'] = self.data['FX'] / self.data['FZ'] 
        self.data['fy'] = self.data['FY'] / self.data['FZ'] 
        self.data['tx'] = self.data['TX'] / self.data['FZ'] 
        self.data['ty'] = self.data['TY'] / self.data['FZ'] 
        self.data['tz'] = self.data['TZ'] / self.data['FZ'] 

        self.wrench_column_headings = ['FX','FY','FZ','TX','TY','TZ'] 
        self.normalized_wrench_column_headings = ['fx','fy','tx','ty','tz']  

        hole_classes = ['HF1','HF2','HE1','HE2','HE3','HE4','HE5','HE6','HV1','HV2','HV3','HV4'] 
        peg_classes = ['PF1','PF2','PF3','PF4','PE1','PE2','PE3','PE4'] 
        self.peg_hole_classes = [] 
        for hole_class in hole_classes: 
            for peg_class in peg_classes: 
                self.peg_hole_classes.append(f'{hole_class}-{peg_class}') 
        
        self.peg_hole_classes_nonzero = [] 
        for peg_hole_class in self.peg_hole_classes: 
            count = self.data[peg_hole_class].sum() 
            if count > 0: 
                self.peg_hole_classes_nonzero.append(peg_hole_class)

        self.data = self.data[self.wrench_column_headings + self.normalized_wrench_column_headings + self.peg_hole_classes_nonzero] 
        
        # TODO: ablations to perform 
        # self.data = self.data[self.wrench_column_headings + self.peg_hole_classes_nonzero] 
        # self.data = self.data[self.normalized_wrench_column_headings + self.peg_hole_classes_nonzero] 
        
        
        # shuffle data 
        self.data = self.data.sample(frac=1, random_state=seed).reset_index(drop=True) 

        # scale data 
        self.scaler = MinMaxScaler() 
        self.data[self.wrench_column_headings] = self.scaler.fit_transform(self.data[self.wrench_column_headings]) 
        self.data[self.normalized_wrench_column_headings] = self.scaler.fit_transform(self.data[self.normalized_wrench_column_headings]) 
        

    def __len__(self): 
        return len(self.data) 
    
    def __getitem__(self, idx): 
        if idx >= len(self.data) or idx < 0:
            raise IndexError("Index out of bounds") 
        
        input = self.data.loc[idx, self.wrench_column_headings+self.normalized_wrench_column_headings].values 

        # TODO: ablations to perform 
        # input = self.data.loc[idx, self.wrench_column_headings].values 
        # input = self.data.loc[idx, self.normalized_wrench_column_headings].values 

        output = self.data.loc[idx, self.peg_hole_classes_nonzero].values 

        return input, output  
            